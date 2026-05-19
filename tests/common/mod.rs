// Shared helpers for wiremock-driven integration tests.
//
// `spawn_proxy()` boots a real ProxyServer bound to a random local port, wired
// to a wiremock MockServer that stands in for LM Studio. Tests register mocks
// on the returned MockServer, then hit the proxy at the returned base URL with
// reqwest just like a real client would.

#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Once;

use tempfile::TempDir;
use tokio::task::JoinHandle;
use warp::Filter;
use wiremock::MockServer;

use ollama_lmstudio_proxy::config::{Config, RuntimeConfig, init_runtime_config};
use ollama_lmstudio_proxy::logging::LogConfig;
use ollama_lmstudio_proxy::server::{ProxyServer, handle_rejection};
use ollama_lmstudio_proxy::server::routes::create_routes;

static INIT_RUNTIME: Once = Once::new();

fn ensure_runtime_initialized() {
    INIT_RUNTIME.call_once(|| {
        init_runtime_config(RuntimeConfig {
            max_buffer_size: 262_144,
            enable_chunk_recovery: true,
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
    ensure_runtime_initialized();

    let mock = MockServer::start().await;
    let state_dir = tempfile::tempdir().expect("create temp state dir");

    let config = Config {
        listen: "127.0.0.1:0".to_string(),
        lmstudio_url: mock.uri(),
        log_level: "off".to_string(),
        load_timeout_seconds: 15,
        max_buffer_size: 262_144,
        enable_chunk_recovery: true,
        model_resolution_cache_ttl_seconds: 1,
        update: false,
    };

    let server = ProxyServer::new_with_state_dir(config, state_dir.path().to_path_buf())
        .expect("ProxyServer::new_with_state_dir");
    let server = Arc::new(server);

    let cors = warp::cors()
        .allow_any_origin()
        .allow_headers(vec![
            "Content-Type",
            "Authorization",
            "Accept",
            "Origin",
            "X-Requested-With",
        ])
        .allow_methods(vec!["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"]);

    let routes = create_routes(server)
        .recover(handle_rejection)
        .with(cors);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local_addr");

    let handle = tokio::spawn(async move {
        warp::serve(routes).incoming(listener).run().await;
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

pub fn _unused_path_marker(_p: PathBuf) {}
