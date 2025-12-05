use crate::handlers::helpers::json_response;
use bytes::Bytes;
use clap::Parser;
use moka::future::Cache;
use serde_json::Value;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use warp::http::HeaderMap;
use warp::log::Info as LogInfo;
use warp::{Filter, Rejection, Reply};

use crate::common::RequestContext;
use crate::constants::*;
use crate::handlers;
use crate::handlers::ollama::EmbeddingResponseMode;
use crate::model::ModelResolver;
use crate::storage::{blob::BlobStore, virtual_models::VirtualModelStore};
use crate::utils::{
    ProxyError, init_global_logger, is_logging_enabled, log_error, log_info, validate_config,
};

#[derive(Parser, Debug, Clone)]
#[command(name = "ollama-lmstudio-proxy")]
#[command(about = "High-performance proxy server bridging Ollama API and LM Studio")]
pub struct Config {
    #[arg(long, default_value = "0.0.0.0:11434", help = "Server listen address")]
    pub listen: String,

    #[arg(
        long,
        default_value = "http://localhost:1234",
        help = "LM Studio backend URL"
    )]
    pub lmstudio_url: String,

    #[arg(long, help = "Disable logging output")]
    pub no_log: bool,

    #[arg(
        long,
        default_value = "15",
        help = "Model loading wait timeout in seconds (after trigger)"
    )]
    pub load_timeout_seconds: u64,

    #[arg(
        long,
        default_value = "262144",
        help = "Initial buffer size in bytes for SSE message assembly (capacity hint)"
    )]
    pub max_buffer_size: usize,

    #[arg(long, help = "Enable partial chunk recovery for streams")]
    pub enable_chunk_recovery: bool,

    #[arg(
        long,
        default_value = "300",
        help = "TTL for model resolution cache in seconds"
    )]
    pub model_resolution_cache_ttl_seconds: u64,

    #[arg(
        long,
        help = "Enable debug logging (prints full request/response bodies)"
    )]
    pub debug: bool,
}

/// Enum to hold either native or legacy model resolver
#[derive(Clone)]
pub enum ModelResolverType {
    Native(Arc<ModelResolver>),
}

/// Production-ready proxy server with dual API support
#[derive(Clone)]
pub struct ProxyServer {
    pub client: reqwest::Client,
    pub config: Arc<Config>,
    pub model_resolver: ModelResolverType,
    pub virtual_models: Arc<VirtualModelStore>,
    pub blob_store: Arc<BlobStore>,
}

fn tolerant_json_body() -> impl Filter<Extract = (Value,), Error = Rejection> + Clone {
    warp::body::content_length_limit(MAX_JSON_BODY_SIZE_BYTES)
        .and(warp::body::bytes())
        .and_then(|body: Bytes| async move {
            if body.is_empty() {
                return Err(warp::reject::custom(ProxyError::bad_request(
                    "Missing JSON body",
                )));
            }
            serde_json::from_slice::<Value>(&body).map_err(|err| {
                warp::reject::custom(ProxyError::bad_request(&format!(
                    "Invalid JSON payload: {}",
                    err
                )))
            })
        })
}

impl ProxyServer {
    /// Create new proxy server instance with API selection
    pub fn new(config: Config) -> Result<Self, Box<dyn std::error::Error>> {
        validate_config(&config)?;

        let runtime_config = RuntimeConfig {
            max_buffer_size: if config.max_buffer_size > 0 {
                config.max_buffer_size
            } else {
                usize::MAX
            },
            string_buffer_size: 2048,
            enable_chunk_recovery: config.enable_chunk_recovery,
        };
        init_runtime_config(runtime_config);
        init_global_logger(!config.no_log);

        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .pool_max_idle_per_host(10)
            .build()?;

        let model_cache: Cache<String, String> = Cache::builder()
            .time_to_live(Duration::from_secs(
                config.model_resolution_cache_ttl_seconds,
            ))
            .build();

        let data_root = dirs::cache_dir()
            .unwrap_or_else(std::env::temp_dir)
            .join("ollama-lmstudio-proxy");
        let virtual_model_store = Arc::new(VirtualModelStore::load(
            data_root.join("virtual_models.json"),
        )?);
        let blob_store = Arc::new(BlobStore::new(data_root.join("blobs"))?);

        log_info("Using native LM Studio API mode");
        let model_resolver = ModelResolverType::Native(Arc::new(ModelResolver::new(
            config.lmstudio_url.clone(),
            model_cache,
        )));

        Ok(Self {
            client,
            config: Arc::new(config),
            model_resolver,
            virtual_models: virtual_model_store,
            blob_store,
        })
    }

    /// Run the proxy server
    pub async fn run(self) -> Result<(), Box<dyn std::error::Error>> {
        self.print_startup_banner();

        let addr: SocketAddr = self
            .config
            .listen
            .parse()
            .map_err(|e| format!("Invalid listen address '{}': {}", self.config.listen, e))?;

        let server_arc = Arc::new(self);

        let log_filter = warp::log::custom({
            let logging_enabled = is_logging_enabled();
            move |info: LogInfo| {
                if logging_enabled {
                    let status_icon = match info.status().as_u16() {
                        200..=299 => LOG_PREFIX_REQUEST,
                        400..=499 => LOG_PREFIX_WARNING,
                        500..=599 => LOG_PREFIX_ERROR,
                        _ => "❔",
                    };
                    crate::utils::STRING_BUFFER.with(|buf_cell| {
                        let mut buffer = buf_cell.borrow_mut();
                        buffer.clear();
                        use std::fmt::Write;
                        let _ = write!(
                            buffer,
                            "{} {} {} | {} | {}",
                            status_icon,
                            info.method(),
                            info.path(),
                            info.status(),
                            crate::utils::format_duration(info.elapsed())
                        );
                        println!("[{}] {}", chrono::Local::now().format("%H:%M:%S"), buffer);
                    });
                }
            }
        });

        let with_server_state = warp::any().map({
            let server_clone = server_arc.clone();
            move || server_clone.clone()
        });

        let ollama_tags_route = warp::path!("api" / "tags")
            .and(warp::get())
            .and(with_server_state.clone())
            .and_then(|s: Arc<ProxyServer>| async move {
                let context = RequestContext {
                    client: &s.client,
                    lmstudio_url: &s.config.lmstudio_url,
                    virtual_models: s.virtual_models.clone(),
                    blob_store: s.blob_store.clone(),
                };
                let token = CancellationToken::new();
                handlers::ollama::handle_ollama_tags(
                    context,
                    s.model_resolver.clone(),
                    token,
                    s.config.as_ref(),
                )
                .await
                .map_err(warp::reject::custom)
            });

        let ollama_chat_route = warp::path!("api" / "chat")
            .and(warp::post())
            .and(tolerant_json_body())
            .and(with_server_state.clone())
            .and_then(|body: Value, s: Arc<ProxyServer>| async move {
                let context = RequestContext {
                    client: &s.client,
                    lmstudio_url: &s.config.lmstudio_url,
                    virtual_models: s.virtual_models.clone(),
                    blob_store: s.blob_store.clone(),
                };
                let token = CancellationToken::new();
                let config_ref = s.config.as_ref();
                handlers::ollama::handle_ollama_chat(
                    context,
                    s.model_resolver.clone(),
                    body,
                    token,
                    config_ref,
                )
                .await
                .map_err(warp::reject::custom)
            });

        let ollama_generate_route = warp::path!("api" / "generate")
            .and(warp::post())
            .and(tolerant_json_body())
            .and(with_server_state.clone())
            .and_then(|body: Value, s: Arc<ProxyServer>| async move {
                let context = RequestContext {
                    client: &s.client,
                    lmstudio_url: &s.config.lmstudio_url,
                    virtual_models: s.virtual_models.clone(),
                    blob_store: s.blob_store.clone(),
                };
                let token = CancellationToken::new();
                let config_ref = s.config.as_ref();
                handlers::ollama::handle_ollama_generate(
                    context,
                    s.model_resolver.clone(),
                    body,
                    token,
                    config_ref,
                )
                .await
                .map_err(warp::reject::custom)
            });

        let embed_endpoint = warp::path!("api" / "embed").map(|| EmbeddingResponseMode::Embed);
        let embeddings_endpoint =
            warp::path!("api" / "embeddings").map(|| EmbeddingResponseMode::LegacyEmbeddings);

        let ollama_embeddings_route = embed_endpoint
            .or(embeddings_endpoint)
            .unify()
            .and(warp::post())
            .and(tolerant_json_body())
            .and(with_server_state.clone())
            .and_then(|mode, body: Value, s: Arc<ProxyServer>| async move {
                let context = RequestContext {
                    client: &s.client,
                    lmstudio_url: &s.config.lmstudio_url,
                    virtual_models: s.virtual_models.clone(),
                    blob_store: s.blob_store.clone(),
                };
                let token = CancellationToken::new();
                handlers::ollama::handle_ollama_embeddings(
                    context,
                    s.model_resolver.clone(),
                    body,
                    mode,
                    token,
                    s.config.as_ref(),
                )
                .await
                .map_err(warp::reject::custom)
            });

        let ollama_pull_route = warp::path!("api" / "pull")
            .and(warp::post())
            .and(tolerant_json_body())
            .and(with_server_state.clone())
            .and_then(|body: Value, s: Arc<ProxyServer>| async move {
                let context = RequestContext {
                    client: &s.client,
                    lmstudio_url: &s.config.lmstudio_url,
                    virtual_models: s.virtual_models.clone(),
                    blob_store: s.blob_store.clone(),
                };
                let token = CancellationToken::new();
                handlers::ollama::handle_ollama_pull(
                    context,
                    s.model_resolver.clone(),
                    body,
                    token,
                    s.config.as_ref(),
                )
                .await
                .map_err(warp::reject::custom)
            });

        let ollama_create_route = warp::path!("api" / "create")
            .and(warp::post())
            .and(tolerant_json_body())
            .and(with_server_state.clone())
            .and_then(|body: Value, s: Arc<ProxyServer>| async move {
                let context = RequestContext {
                    client: &s.client,
                    lmstudio_url: &s.config.lmstudio_url,
                    virtual_models: s.virtual_models.clone(),
                    blob_store: s.blob_store.clone(),
                };
                let token = CancellationToken::new();
                handlers::ollama::handle_ollama_create(
                    context,
                    s.model_resolver.clone(),
                    body,
                    token,
                    s.config.as_ref(),
                )
                .await
                .map_err(warp::reject::custom)
            });

        let ollama_copy_route = warp::path!("api" / "copy")
            .and(warp::post())
            .and(tolerant_json_body())
            .and(with_server_state.clone())
            .and_then(|body: Value, s: Arc<ProxyServer>| async move {
                let context = RequestContext {
                    client: &s.client,
                    lmstudio_url: &s.config.lmstudio_url,
                    virtual_models: s.virtual_models.clone(),
                    blob_store: s.blob_store.clone(),
                };
                let token = CancellationToken::new();
                handlers::ollama::handle_ollama_copy(
                    context,
                    s.model_resolver.clone(),
                    body,
                    token,
                    s.config.as_ref(),
                )
                .await
                .map_err(warp::reject::custom)
            });

        let ollama_delete_route = warp::path!("api" / "delete")
            .and(warp::delete())
            .and(tolerant_json_body())
            .and(with_server_state.clone())
            .and_then(|body: Value, s: Arc<ProxyServer>| async move {
                let context = RequestContext {
                    client: &s.client,
                    lmstudio_url: &s.config.lmstudio_url,
                    virtual_models: s.virtual_models.clone(),
                    blob_store: s.blob_store.clone(),
                };
                handlers::ollama::handle_ollama_delete(context, body, s.config.as_ref())
                    .await
                    .map_err(warp::reject::custom)
            });

        let ollama_push_route = warp::path!("api" / "push")
            .and(warp::post())
            .and(tolerant_json_body())
            .and(with_server_state.clone())
            .and_then(|body: Value, s: Arc<ProxyServer>| async move {
                let context = RequestContext {
                    client: &s.client,
                    lmstudio_url: &s.config.lmstudio_url,
                    virtual_models: s.virtual_models.clone(),
                    blob_store: s.blob_store.clone(),
                };
                let token = CancellationToken::new();
                handlers::ollama::handle_ollama_push(
                    context,
                    s.model_resolver.clone(),
                    body,
                    token,
                    s.config.as_ref(),
                )
                .await
                .map_err(warp::reject::custom)
            });

        let ollama_show_route = warp::path!("api" / "show")
            .and(warp::post())
            .and(tolerant_json_body())
            .and(with_server_state.clone())
            .and_then(|body: Value, s: Arc<ProxyServer>| async move {
                let context = RequestContext {
                    client: &s.client,
                    lmstudio_url: &s.config.lmstudio_url,
                    virtual_models: s.virtual_models.clone(),
                    blob_store: s.blob_store.clone(),
                };
                let token = CancellationToken::new();
                handlers::ollama::handle_ollama_show(
                    context,
                    s.model_resolver.clone(),
                    body,
                    token,
                    s.config.as_ref(),
                )
                .await
                .map_err(warp::reject::custom)
            });

        let ollama_ps_route = warp::path!("api" / "ps")
            .and(warp::get())
            .and(with_server_state.clone())
            .and_then(|s: Arc<ProxyServer>| async move {
                let context = RequestContext {
                    client: &s.client,
                    lmstudio_url: &s.config.lmstudio_url,
                    virtual_models: s.virtual_models.clone(),
                    blob_store: s.blob_store.clone(),
                };
                let token = CancellationToken::new();
                handlers::ollama::handle_ollama_ps(
                    context,
                    s.model_resolver.clone(),
                    token,
                    s.config.as_ref(),
                )
                .await
                .map_err(warp::reject::custom)
            });

        let ollama_version_route = warp::path!("api" / "version")
            .and(warp::get())
            .and(with_server_state.clone())
            .and_then(|s: Arc<ProxyServer>| async move {
                handlers::ollama::handle_ollama_version(s.config.as_ref())
                    .await
                    .map_err(warp::reject::custom)
            });

        let blob_head_route = warp::path!("api" / "blobs" / String)
            .and(warp::head())
            .and(with_server_state.clone())
            .and_then(|digest: String, s: Arc<ProxyServer>| async move {
                let context = RequestContext {
                    client: &s.client,
                    lmstudio_url: &s.config.lmstudio_url,
                    virtual_models: s.virtual_models.clone(),
                    blob_store: s.blob_store.clone(),
                };
                handlers::ollama::handle_blob_head(context, digest, s.config.as_ref())
                    .await
                    .map_err(warp::reject::custom)
            });

        let blob_upload_route = warp::path!("api" / "blobs" / String)
            .and(warp::post())
            .and(warp::body::stream())
            .and(with_server_state.clone())
            .and_then(|digest: String, stream, s: Arc<ProxyServer>| async move {
                let context = RequestContext {
                    client: &s.client,
                    lmstudio_url: &s.config.lmstudio_url,
                    virtual_models: s.virtual_models.clone(),
                    blob_store: s.blob_store.clone(),
                };
                handlers::ollama::handle_blob_upload(context, digest, stream, s.config.as_ref())
                    .await
                    .map_err(warp::reject::custom)
            });

        let lmstudio_passthrough_route = warp::path("v1")
            .and(warp::path::tail())
            .and(warp::method())
            .and(warp::body::bytes())
            .and(warp::header::headers_cloned())
            .and(
                warp::query::raw()
                    .map(Some)
                    .or(warp::any().map(|| None))
                    .unify(),
            )
            .and(with_server_state.clone())
            .and_then(
                |tail: warp::path::Tail,
                 method: warp::http::Method,
                 body_bytes: Bytes,
                 headers: HeaderMap,
                 query: Option<String>,
                 s: Arc<ProxyServer>| async move {
                    let context = RequestContext {
                        client: &s.client,
                        lmstudio_url: &s.config.lmstudio_url,
                        virtual_models: s.virtual_models.clone(),
                        blob_store: s.blob_store.clone(),
                    };
                    let token = CancellationToken::new();
                    let full_path = format!("/v1/{}", tail.as_str());
                    handlers::lmstudio::handle_lmstudio_passthrough(
                        context,
                        s.model_resolver.clone(),
                        handlers::lmstudio::LmStudioPassthroughRequest {
                            method,
                            endpoint: full_path,
                            body: body_bytes,
                            headers,
                            query,
                        },
                        token,
                        s.config.load_timeout_seconds,
                        s.config.debug,
                    )
                    .await
                    .map_err(warp::reject::custom)
                },
            );

        let lmstudio_native_passthrough_route = warp::path("api")
            .and(warp::path::param::<String>())
            .and(warp::path::tail())
            .and(warp::method())
            .and(warp::body::bytes())
            .and(warp::header::headers_cloned())
            .and(
                warp::query::raw()
                    .map(Some)
                    .or(warp::any().map(|| None))
                    .unify(),
            )
            .and(with_server_state.clone())
            .and_then(
                |version: String,
                 tail: warp::path::Tail,
                 method: warp::http::Method,
                 body_bytes: Bytes,
                 headers: HeaderMap,
                 query: Option<String>,
                 s: Arc<ProxyServer>| async move {
                    let context = RequestContext {
                        client: &s.client,
                        lmstudio_url: &s.config.lmstudio_url,
                        virtual_models: s.virtual_models.clone(),
                        blob_store: s.blob_store.clone(),
                    };
                    let token = CancellationToken::new();
                    if version != "v0" && version != "v1" {
                        return Err(warp::reject::not_found());
                    }
                    let base_path = format!("/api/{}", version);
                    let full_path = if tail.as_str().is_empty() {
                        base_path
                    } else {
                        format!("{}/{}", base_path, tail.as_str())
                    };
                    handlers::lmstudio::handle_lmstudio_passthrough(
                        context,
                        s.model_resolver.clone(),
                        handlers::lmstudio::LmStudioPassthroughRequest {
                            method,
                            endpoint: full_path,
                            body: body_bytes,
                            headers,
                            query,
                        },
                        token,
                        s.config.load_timeout_seconds,
                        s.config.debug,
                    )
                    .await
                    .map_err(warp::reject::custom)
                },
            );

        let health_route = warp::path("health")
            .and(warp::get())
            .and(with_server_state.clone())
            .and_then(|s: Arc<ProxyServer>| async move {
                let context = RequestContext {
                    client: &s.client,
                    lmstudio_url: &s.config.lmstudio_url,
                    virtual_models: s.virtual_models.clone(),
                    blob_store: s.blob_store.clone(),
                };
                let token = CancellationToken::new();
                match handlers::ollama::handle_health_check(context, token, s.config.as_ref()).await
                {
                    Ok(status_json) => Ok(json_response(&status_json)),
                    Err(e) => Err(warp::reject::custom(e)),
                }
            });

        let app_routes = ollama_tags_route
            .boxed()
            .or(ollama_chat_route.boxed())
            .or(ollama_generate_route.boxed())
            .or(ollama_embeddings_route.boxed())
            .or(ollama_pull_route.boxed())
            .or(ollama_create_route.boxed())
            .or(ollama_copy_route.boxed())
            .or(ollama_delete_route.boxed())
            .or(ollama_push_route.boxed())
            .or(ollama_show_route.boxed())
            .or(ollama_ps_route.boxed())
            .or(ollama_version_route.boxed())
            .or(blob_head_route.boxed())
            .or(blob_upload_route.boxed())
            .or(lmstudio_passthrough_route.boxed())
            .or(lmstudio_native_passthrough_route.boxed())
            .or(health_route.boxed())
            .boxed();

        let final_routes = app_routes.recover(handle_rejection).with(log_filter);

        warp::serve(final_routes).run(addr).await;
        Ok(())
    }

    /// Print startup banner with configuration info
    fn print_startup_banner(&self) {
        if is_logging_enabled() {
            println!();
            println!("Ollama LM Studio Proxy - Version: {}", crate::VERSION);
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

            // Configuration information
            println!("📡 Listening on: {}", self.config.listen);
            println!("🔗 LM Studio URL: {}", self.config.lmstudio_url);
            println!(
                "📝 Logging: {}",
                if is_logging_enabled() {
                    if self.config.debug {
                        "Enabled (Debug Mode)"
                    } else {
                        "Enabled"
                    }
                } else {
                    "Disabled"
                }
            );
            println!(
                "🕒 Model Load Timeout: {}s",
                self.config.load_timeout_seconds
            );
            println!(
                "🕒 Cache TTL: {}s",
                self.config.model_resolution_cache_ttl_seconds
            );
            println!(
                "📊 Initial SSE Buffer: {} bytes",
                self.config.max_buffer_size
            );
            println!(
                "🔄 Chunk Recovery: {}",
                if get_runtime_config().enable_chunk_recovery {
                    "Enabled"
                } else {
                    "Disabled"
                }
            );
            println!("🔌 API Mode: LM Studio Native REST API");
            println!("     • Requires LM Studio 0.3.6+");

            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!(
                "✅ All requests sent to {} will be converted and forwarded to LM Studio",
                self.config.listen
            );
        }
    }
}

/// Enhanced error handling with proper status codes and JSON response
async fn handle_rejection(err: Rejection) -> Result<impl Reply, Infallible> {
    let code;
    let message;
    let error_type;

    if err.is_not_found() {
        code = warp::http::StatusCode::NOT_FOUND;
        message = "Endpoint not found".to_string();
        error_type = "not_found_error".to_string();
    } else if let Some(proxy_error) = err.find::<ProxyError>() {
        code = warp::http::StatusCode::from_u16(proxy_error.status_code)
            .unwrap_or(warp::http::StatusCode::INTERNAL_SERVER_ERROR);
        message = proxy_error.message.clone();
        error_type = match proxy_error.status_code {
            400 => "bad_request_error".to_string(),
            401 => "authentication_error".to_string(),
            403 => "permission_error".to_string(),
            404 => "not_found_error".to_string(),
            413 => "payload_too_large_error".to_string(),
            429 => "rate_limit_error".to_string(),
            499 => "client_closed_request".to_string(),
            500 => "internal_server_error".to_string(),
            501 => "not_implemented_error".to_string(),
            503 => "service_unavailable_error".to_string(),
            _ => "api_error".to_string(),
        };
    } else if err.find::<warp::reject::MethodNotAllowed>().is_some() {
        code = warp::http::StatusCode::METHOD_NOT_ALLOWED;
        message = "Method Not Allowed".to_string();
        error_type = "method_not_allowed_error".to_string();
    } else if err.find::<warp::reject::PayloadTooLarge>().is_some() {
        code = warp::http::StatusCode::PAYLOAD_TOO_LARGE;
        message = "Payload Too Large (check backend or underlying HTTP server limits)".to_string();
        error_type = "payload_too_large_error".to_string();
    } else if err.find::<warp::reject::UnsupportedMediaType>().is_some() {
        code = warp::http::StatusCode::UNSUPPORTED_MEDIA_TYPE;
        message = "Unsupported Media Type. Expected application/json.".to_string();
        error_type = "unsupported_media_type_error".to_string();
    } else {
        log_error("Unhandled rejection", &format!("{:?}", err));
        code = warp::http::StatusCode::INTERNAL_SERVER_ERROR;
        message = "An unexpected internal error occurred.".to_string();
        error_type = "internal_server_error".to_string();
    }

    let json_error = serde_json::json!({
        "error": {
            "message": message,
            "type": error_type,
            "code": code.as_u16(),
            "timestamp": chrono::Utc::now().to_rfc3339()
        }
    });

    Ok(warp::reply::with_status(
        warp::reply::json(&json_error),
        code,
    ))
}

#[cfg(test)]
mod server_tests {
    use warp::Filter;

    #[tokio::test]
    async fn native_prefix_filter_matches() {
        let filter = warp::path("api")
            .and(warp::path::param::<String>())
            .and(warp::path::tail());
        let (version, tail) = warp::test::request()
            .method("GET")
            .path("/api/v1/models")
            .filter(&filter)
            .await
            .expect("filter should match");
        assert_eq!(version, "v1");
        assert_eq!(tail.as_str(), "models");
    }

    #[tokio::test]
    async fn tolerant_json_accepts_missing_content_type() {
        let filter = super::tolerant_json_body();
        let value: serde_json::Value = warp::test::request()
            .method("POST")
            .path("/")
            .body("{\"model\":\"demo\"}")
            .filter(&filter)
            .await
            .expect("JSON should parse without header");
        assert_eq!(value["model"], "demo");
    }

    #[tokio::test]
    async fn tolerant_json_rejects_invalid_payload() {
        let filter = super::tolerant_json_body();
        let result = warp::test::request()
            .method("POST")
            .path("/")
            .body("not-json")
            .filter(&filter)
            .await;
        assert!(result.is_err());
    }
}
