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
use ollama_lmstudio_proxy::logging::LogConfig;
use ollama_lmstudio_proxy::proxy::ProxyServer;
use ollama_lmstudio_proxy::proxy::routes::create_router;
use ollama_lmstudio_proxy::proxy::server::cors_layer;

static INIT_RUNTIME: Once = Once::new();

fn ensure_runtime_initialized(enable_chunk_recovery: bool) {
    INIT_RUNTIME.call_once(|| {
        init_runtime_config(RuntimeConfig {
            max_buffer_size: 262_144,
            enable_chunk_recovery,
            flash_attention: false,
            offload_kv_cache: false,
            eval_batch_size: None,
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
    spawn_proxy_inner(enable_chunk_recovery, false, false, true).await
}

/// Boot a proxy with the experimental native `/api/v1/chat` path enabled, so
/// `/api/chat` routes through LM Studio's native endpoint instead of the
/// OpenAI-compat `/api/v0/chat/completions`.
pub async fn spawn_proxy_with_native() -> TestProxy {
    spawn_proxy_inner(true, true, false, true).await
}

/// Boot a proxy with `/api/web_search` configured to forward to the mock
/// server's `/search` endpoint (with a bearer key). Mount a POST `/search`
/// mock to drive it.
pub async fn spawn_proxy_with_search() -> TestProxy {
    spawn_proxy_inner(true, false, true, true).await
}

/// Boot a proxy with the web_fetch SSRF guard ENABLED (private/loopback targets
/// rejected) — i.e. `--allow-private-fetch` off.
pub async fn spawn_proxy_strict_ssrf() -> TestProxy {
    spawn_proxy_inner(true, false, false, false).await
}

/// Bearer key the search-configured test proxy sends to its provider.
pub const TEST_SEARCH_API_KEY: &str = "test-search-key";

async fn spawn_proxy_inner(
    enable_chunk_recovery: bool,
    use_native_chat: bool,
    configure_search: bool,
    allow_private_fetch: bool,
) -> TestProxy {
    ensure_runtime_initialized(enable_chunk_recovery);

    let mock = MockServer::start().await;
    let state_dir = tempfile::tempdir().expect("create temp state dir");

    let search_url = configure_search.then(|| format!("{}/search", mock.uri()));
    let search_api_key = configure_search.then(|| TEST_SEARCH_API_KEY.to_string());

    let config = Config {
        listen: "127.0.0.1:0".to_string(),
        lmstudio_url: mock.uri(),
        log_level: "off".to_string(),
        load_timeout_seconds: 15,
        max_buffer_size: 262_144,
        enable_chunk_recovery,
        model_resolution_cache_ttl_seconds: 1,
        lmstudio_token: None,
        use_native_chat,
        flash_attention: false,
        offload_kv_cache: false,
        eval_batch_size: None,
        allow_private_fetch,
        search_url,
        search_api_key,
        ollama_version: "0.30.0".to_string(),
    };

    let server = ProxyServer::new_with_state_dir(config, state_dir.path().to_path_buf())
        .expect("ProxyServer::new_with_state_dir");
    let server = Arc::new(server);

    let app = create_router(server).layer(cors_layer());

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local_addr");

    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.expect("axum::serve");
    });

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .expect("reqwest client");

    TestProxy {
        base_url: format!("http://{}", addr),
        mock,
        client,
        _state_dir: state_dir,
        _server_task: handle,
    }
}
