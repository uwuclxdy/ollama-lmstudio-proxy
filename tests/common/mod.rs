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
    ensure_runtime_initialized(enable_chunk_recovery);

    let mock = MockServer::start().await;
    let state_dir = tempfile::tempdir().expect("create temp state dir");

    let config = Config {
        listen: "127.0.0.1:0".to_string(),
        lmstudio_url: mock.uri(),
        log_level: "off".to_string(),
        load_timeout_seconds: 15,
        max_buffer_size: 262_144,
        enable_chunk_recovery,
        model_resolution_cache_ttl_seconds: 1,
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
