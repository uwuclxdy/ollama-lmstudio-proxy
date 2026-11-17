// Shared helpers for wiremock-driven integration tests.
//
// `spawn_proxy()` boots a real ProxyServer bound to a random local port, wired
// to a wiremock MockServer that stands in for LM Studio. Tests register mocks
// on the returned MockServer, then hit the proxy at the returned base URL with
// reqwest just like a real client would.

#![allow(dead_code)]

use std::sync::Arc;
use std::sync::Once;

use tempfile::TempDir;
use tokio::task::JoinHandle;
use wiremock::MockServer;

use ollama_lmstudio_proxy::config::{Config, RuntimeConfig, init_runtime_config};
use ollama_lmstudio_proxy::constants::STREAM_TIMEOUT_SECONDS;
use ollama_lmstudio_proxy::logging::LogConfig;
use ollama_lmstudio_proxy::proxy::ProxyServer;
use ollama_lmstudio_proxy::proxy::routes::create_router;
use ollama_lmstudio_proxy::proxy::server::cors_layer;
use ollama_lmstudio_proxy::proxy::server::normalize_path;

static INIT_RUNTIME: Once = Once::new();

fn ensure_runtime_initialized(enable_chunk_recovery: bool) {
    INIT_RUNTIME.call_once(|| {
        init_runtime_config(RuntimeConfig {
            max_buffer_size: 262_144,
            enable_chunk_recovery,
            flash_attention: false,
            offload_kv_cache: false,
            eval_batch_size: None,
            default_context_length: None,
            auto_evict: false,
        });
        LogConfig::init(false);
    });
}

pub struct TestProxy {
    pub base_url: String,
    pub mock: MockServer,
    pub client: reqwest::Client,
    _state_dir: TempDir,
    _server_task: JoinHandle<()>,
}

impl TestProxy {
    pub fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }
}

pub async fn spawn_proxy() -> TestProxy {
    spawn_proxy_with_recovery(true).await
}

pub async fn spawn_proxy_with_recovery(enable_chunk_recovery: bool) -> TestProxy {
    spawn_proxy_inner(
        enable_chunk_recovery,
        false,
        false,
        true,
        None,
        None,
        false,
        false,
        15,
        false,
        None,
    )
    .await
}

/// Boot the production router with a response-rebuilding layer stacked
/// OUTSIDE the whole production stack, so any response extension an inner
/// error render carries is dropped before the client sees the body.
pub async fn spawn_proxy_with_rebuild_layer() -> TestProxy {
    spawn_proxy_inner(
        true, false, false, true, None, None, false, false, 15, true, None,
    )
    .await
}

/// Boot a proxy with the experimental native `/api/v1/chat` path enabled, so
/// `/api/chat` routes through LM Studio's native endpoint instead of the
/// OpenAI-compat `/api/v0/chat/completions`.
pub async fn spawn_proxy_with_native() -> TestProxy {
    spawn_proxy_inner(
        true, true, false, true, None, None, false, false, 15, false, None,
    )
    .await
}

/// Boot a proxy with the `/api/web_search` configured to forward to the mock
/// server's `/search` endpoint (with a bearer key). Mount a POST `/search`
/// mock to drive it.
pub async fn spawn_proxy_with_search() -> TestProxy {
    spawn_proxy_inner(
        true, false, true, true, None, None, false, false, 15, false, None,
    )
    .await
}

/// Boot a proxy with the web_fetch SSRF guard ENABLED (private/loopback targets
/// rejected) — i.e. `--allow-private-fetch` off.
pub async fn spawn_proxy_strict_ssrf() -> TestProxy {
    spawn_proxy_inner(
        true, false, false, false, None, None, false, false, 15, false, None,
    )
    .await
}

/// Boot a proxy requiring an inbound `Authorization: Bearer <api_key>` on every
/// request (the `--api-key` / `OLLAMA_API_KEY` gate). `None` for api_key leaves
/// the proxy open, matching the default.
pub async fn spawn_proxy_with_api_key(api_key: &str) -> TestProxy {
    spawn_proxy_inner(
        true,
        false,
        false,
        true,
        Some(api_key.to_string()),
        None,
        false,
        false,
        15,
        false,
        None,
    )
    .await
}

/// Boot a proxy requiring the inbound `api_key` AND configured with an outbound
/// LM Studio token, so a test can pin which credential reaches the backend.
pub(crate) async fn spawn_proxy_with_api_key_and_token(
    api_key: &str,
    lmstudio_token: &str,
) -> TestProxy {
    spawn_proxy_inner(
        true,
        false,
        false,
        true,
        Some(api_key.to_string()),
        Some(lmstudio_token.to_string()),
        false,
        false,
        15,
        false,
        None,
    )
    .await
}

/// Boot a proxy with `--native-chat-streaming` on: streaming `/api/chat`
/// (`stream:true`) routes through native `/api/v1/chat`, non-streaming stays on
/// the OpenAI-compat `/api/v0/chat/completions` path.
pub async fn spawn_proxy_with_native_streaming() -> TestProxy {
    spawn_proxy_inner(
        true, false, false, true, None, None, true, false, 15, false, None,
    )
    .await
}

/// Boot a proxy with `--auto-evict` on: proactively evicts other loaded models
/// before inference when the target model is not yet loaded.
pub async fn spawn_proxy_with_auto_evict() -> TestProxy {
    spawn_proxy_inner(
        true, false, false, true, None, None, false, true, 15, false, None,
    )
    .await
}

/// Boot a proxy with a custom `load_timeout_seconds` — useful for tests that
/// exercise the cold-load retry path and need a short post-trigger wait bound
/// to keep the test fast.
pub async fn spawn_proxy_with_load_timeout(load_timeout_seconds: u64) -> TestProxy {
    spawn_proxy_inner(
        true,
        false,
        false,
        true,
        None,
        None,
        false,
        false,
        load_timeout_seconds,
        false,
        None,
    )
    .await
}

/// Boot a proxy whose LM Studio base URL carries a path prefix (e.g.
/// `/model-proxy/v1`), mirroring a user-configured reverse-proxy base whose
/// path embeds classifier-relevant words; endpoint urls the proxy builds then
/// carry those words, so message-derived classification bugs reproduce.
/// `load_timeout_seconds` is pinned to 2 to keep a spurious load detour cheap.
pub async fn spawn_proxy_with_lmstudio_prefix(prefix: &str) -> TestProxy {
    spawn_proxy_inner(
        true,
        false,
        false,
        true,
        None,
        None,
        false,
        false,
        2,
        false,
        Some(prefix.to_string()),
    )
    .await
}

/// Bearer key the search-configured test proxy sends to its provider.
pub const TEST_SEARCH_API_KEY: &str = "test-search-key";

#[allow(clippy::too_many_arguments)]
#[expect(
    clippy::fn_params_excessive_bools,
    reason = "the spawn matrix reads as one flat argument list at every helper"
)]
async fn spawn_proxy_inner(
    enable_chunk_recovery: bool,
    use_native_chat: bool,
    configure_search: bool,
    allow_private_fetch: bool,
    api_key: Option<String>,
    lmstudio_token: Option<String>,
    native_chat_streaming: bool,
    auto_evict: bool,
    load_timeout_seconds: u64,
    rebuild_outer_layer: bool,
    lmstudio_path_prefix: Option<String>,
) -> TestProxy {
    ensure_runtime_initialized(enable_chunk_recovery);

    let mock = MockServer::start().await;
    let state_dir = tempfile::tempdir().expect("create temp state dir");

    let search_url = configure_search.then(|| format!("{}/search", mock.uri()));
    let search_api_key = configure_search.then(|| TEST_SEARCH_API_KEY.to_string());

    let config = Config {
        listen: "127.0.0.1:0".to_string(),
        lmstudio_url: match lmstudio_path_prefix {
            Some(prefix) => format!("{}{}", mock.uri(), prefix),
            None => mock.uri(),
        },
        log_level: "off".to_string(),
        load_timeout_seconds,
        stream_timeout_seconds: STREAM_TIMEOUT_SECONDS,
        max_buffer_size: 262_144,
        enable_chunk_recovery,
        model_resolution_cache_ttl_seconds: 1,
        lmstudio_token,
        api_key,
        use_native_chat,
        native_chat_streaming,
        flash_attention: false,
        offload_kv_cache: false,
        eval_batch_size: None,
        auto_evict,
        allow_private_fetch,
        search_url,
        search_api_key,
        ollama_version: "0.30.0".to_string(),
        default_context_length: None,
    };

    let server = ProxyServer::new_with_state_dir(config, state_dir.path().to_path_buf())
        .expect("ProxyServer::new_with_state_dir");
    let server = Arc::new(server);

    let app = router_for(server);
    let app = if rebuild_outer_layer {
        use axum::response::IntoResponse;

        app.layer(axum::middleware::from_fn(
            |req: axum::extract::Request, next: axum::middleware::Next| async move {
                // Rebuild every response: fresh status + headers + body frame,
                // so any extension the inner error render carried is dropped.
                let inner = next.run(req).await;
                let status = inner.status();
                let body = axum::body::to_bytes(inner.into_body(), usize::MAX)
                    .await
                    .expect("read inner body");
                (status, body).into_response()
            },
        ))
    } else {
        app
    };

    let (base_url, client, handle) = serve_router(app).await;

    TestProxy {
        base_url,
        mock,
        client,
        _state_dir: state_dir,
        _server_task: handle,
    }
}

/// Boots the production router with the same middleware stack
/// `ProxyServer::run` layers: api_key_gate → cors. The api_key gate is a
/// no-op when `api_key` is None, so tests without a key are unaffected.
fn router_for(server: Arc<ProxyServer>) -> axum::Router {
    let api_key = Arc::new(server.config.api_key.clone());
    create_router(server)
        .layer(axum::middleware::from_fn_with_state(
            api_key,
            ollama_lmstudio_proxy::proxy::auth::api_key_gate,
        ))
        .layer(cors_layer())
}

/// Wraps the finished router in `normalize_path` (trailing-slash handling
/// matching production), binds an ephemeral port, serves it, and returns the
/// client-facing pieces.
async fn serve_router(app: axum::Router) -> (String, reqwest::Client, JoinHandle<()>) {
    let app = normalize_path(app);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local_addr");
    let handle = tokio::spawn(async move {
        axum::serve(
            listener,
            axum::ServiceExt::<axum::extract::Request>::into_make_service(app),
        )
        .await
        .expect("axum::serve");
    });
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .expect("reqwest client");
    (format!("http://{addr}"), client, handle)
}

/// Boots a proxy whose LM Studio base URL points at an external backend (a raw
/// TCP server the test paces itself) instead of the wiremock MockServer, with
/// explicit load and stream timeout seconds. The MockServer is still started
/// to fill `TestProxy.mock`; these tests mount nothing on it.
pub(crate) async fn spawn_proxy_against_backend(
    lmstudio_url: &str,
    load_timeout_seconds: u64,
    stream_timeout_seconds: u64,
) -> TestProxy {
    ensure_runtime_initialized(true);
    let mock = MockServer::start().await;
    let state_dir = tempfile::tempdir().expect("create temp state dir");
    let config = Config {
        listen: "127.0.0.1:0".to_string(),
        lmstudio_url: lmstudio_url.to_string(),
        log_level: "off".to_string(),
        load_timeout_seconds,
        stream_timeout_seconds,
        max_buffer_size: 262_144,
        enable_chunk_recovery: true,
        model_resolution_cache_ttl_seconds: 1,
        lmstudio_token: None,
        api_key: None,
        use_native_chat: false,
        native_chat_streaming: false,
        flash_attention: false,
        offload_kv_cache: false,
        eval_batch_size: None,
        auto_evict: false,
        allow_private_fetch: true,
        search_url: None,
        search_api_key: None,
        ollama_version: "0.30.0".to_string(),
        default_context_length: None,
    };
    let server = Arc::new(
        ProxyServer::new_with_state_dir(config, state_dir.path().to_path_buf())
            .expect("ProxyServer::new_with_state_dir"),
    );
    let (base_url, client, handle) = serve_router(router_for(server)).await;
    TestProxy {
        base_url,
        mock,
        client,
        _state_dir: state_dir,
        _server_task: handle,
    }
}
