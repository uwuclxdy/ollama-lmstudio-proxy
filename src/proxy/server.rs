use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use moka::future::Cache;
use tokio_util::sync::CancellationToken;
use tower_http::cors::{Any, CorsLayer};

use crate::config::Config;
use crate::logging::LogConfig;
use crate::model::{LoadTracker, ModelResolver};
use crate::proxy::routes::create_router;
use crate::storage::{BlobStore, VirtualModelStore};

pub struct ProxyServer {
    pub client: reqwest::Client,
    pub config: Config,
    pub model_resolver: Arc<ModelResolver>,
    pub virtual_models: Arc<VirtualModelStore>,
    pub blob_store: Arc<BlobStore>,
    pub load_tracker: Arc<LoadTracker>,
    pub shutdown: CancellationToken,
}

impl ProxyServer {
    pub fn new(config: Config) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_with_state_dir(config, get_state_directory())
    }

    pub fn new_with_state_dir(
        config: Config,
        state_dir: PathBuf,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut client_builder = reqwest::Client::builder()
            .timeout(Duration::from_secs(300))
            .pool_max_idle_per_host(32)
            .pool_idle_timeout(Duration::from_secs(90))
            .tcp_keepalive(Duration::from_secs(60));

        if let Some(ref token) = config.lmstudio_token {
            let mut default_headers = reqwest::header::HeaderMap::new();
            let header_value = reqwest::header::HeaderValue::from_str(&format!("Bearer {}", token))
                .map_err(|e| format!("invalid lmstudio-token: {}", e))?;
            default_headers.insert(reqwest::header::AUTHORIZATION, header_value);
            client_builder = client_builder.default_headers(default_headers);
        }

        let client = client_builder.build()?;

        let cache: Cache<String, String> = Cache::builder()
            .max_capacity(1000)
            .time_to_live(Duration::from_secs(
                config.model_resolution_cache_ttl_seconds,
            ))
            .build();

        let model_resolver = Arc::new(ModelResolver::new(config.lmstudio_url.clone(), cache));

        let virtual_models_path = state_dir.join("virtual_models.json");
        let blob_dir = state_dir.join("blobs");

        let virtual_models = Arc::new(VirtualModelStore::load(virtual_models_path)?);
        let blob_store = Arc::new(BlobStore::new(blob_dir)?);
        let load_tracker = LoadTracker::new();

        Ok(Self {
            client,
            config,
            model_resolver,
            virtual_models,
            blob_store,
            load_tracker,
            shutdown: CancellationToken::new(),
        })
    }

    pub async fn run(self) -> Result<(), Box<dyn std::error::Error>> {
        let addr: SocketAddr = self.config.listen.parse()?;
        let server = Arc::new(self);

        let api_key = Arc::new(server.config.api_key.clone());

        let app = create_router(server.clone())
            .layer(axum::middleware::from_fn(access_log))
            .layer(axum::middleware::from_fn_with_state(
                api_key,
                crate::proxy::auth::api_key_gate,
            ))
            .layer(cors_layer());

        if LogConfig::get().debug_enabled {
            log::info!("starting proxy server on {} (debug mode)", addr);
        } else {
            log::info!("starting proxy server on {}", addr);
        }
        log::info!("LM Studio backend: {}", server.config.lmstudio_url);

        let listener = tokio::net::TcpListener::bind(addr).await?;

        let shutdown = server.shutdown.clone();
        tokio::spawn(async move {
            wait_for_shutdown_signal().await;
            log::info!("shutdown signal received, draining in-flight requests");
            shutdown.cancel();
        });

        axum::serve(listener, app)
            .with_graceful_shutdown(server.shutdown.clone().cancelled_owned())
            .await?;

        log::info!("server stopped");
        Ok(())
    }
}

async fn wait_for_shutdown_signal() {
    let ctrl_c = async {
        if let Err(e) = tokio::signal::ctrl_c().await {
            log::error!("failed to install ctrl-c handler: {}", e);
        }
    };

    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{SignalKind, signal};
        match signal(SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(e) => log::error!("failed to install SIGTERM handler: {}", e),
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

async fn access_log(
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let start = std::time::Instant::now();
    let response = next.run(req).await;
    let status = response.status().as_u16();
    let elapsed = crate::logging::format_duration(start.elapsed());
    if status >= 500 {
        log::error!("{} {} -> {} | {}", method, path, status, elapsed);
    } else if status >= 400 {
        log::warn!("{} {} -> {} | {}", method, path, status, elapsed);
    } else {
        log::info!("{} {} -> {} | {}", method, path, status, elapsed);
    }
    response
}

pub fn cors_layer() -> CorsLayer {
    use http::Method;
    CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([
            Method::GET,
            Method::POST,
            Method::PUT,
            Method::DELETE,
            Method::OPTIONS,
            Method::HEAD,
        ])
        .allow_headers(Any)
}

#[cfg(test)]
#[path = "../../tests/unit/auth_token.rs"]
mod tests;

fn get_state_directory() -> PathBuf {
    if let Ok(xdg_cache) = std::env::var("XDG_CACHE_HOME") {
        return PathBuf::from(xdg_cache).join("ollama-lmstudio-proxy");
    }

    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home)
            .join(".cache")
            .join("ollama-lmstudio-proxy");
    }

    std::env::temp_dir().join("ollama-lmstudio-proxy")
}
