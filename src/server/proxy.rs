use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use moka::future::Cache;
use warp::Filter;

use crate::config::Config;
use crate::logging::LogConfig;
use crate::model::ModelResolver;
use crate::server::routes::create_routes;
use crate::server::{ModelResolverType, handle_rejection};
use crate::storage::{BlobStore, VirtualModelStore};

pub struct ProxyServer {
    pub client: reqwest::Client,
    pub config: Config,
    pub model_resolver: ModelResolverType,
    pub virtual_models: Arc<VirtualModelStore>,
    pub blob_store: Arc<BlobStore>,
}

impl ProxyServer {
    pub fn new(config: Config) -> Result<Self, Box<dyn std::error::Error>> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(300))
            .pool_max_idle_per_host(32)
            .pool_idle_timeout(Duration::from_secs(90))
            .tcp_keepalive(Duration::from_secs(60))
            .build()?;

        let cache: Cache<String, String> = Cache::builder()
            .max_capacity(1000)
            .time_to_live(Duration::from_secs(
                config.model_resolution_cache_ttl_seconds,
            ))
            .build();

        let model_resolver = ModelResolverType::Native(Arc::new(ModelResolver::new(
            config.lmstudio_url.clone(),
            cache,
        )));

        let state_dir = get_state_directory();
        let virtual_models_path = state_dir.join("virtual_models.json");
        let blob_dir = state_dir.join("blobs");

        let virtual_models = Arc::new(VirtualModelStore::load(virtual_models_path)?);
        let blob_store = Arc::new(BlobStore::new(blob_dir)?);

        Ok(Self {
            client,
            config,
            model_resolver,
            virtual_models,
            blob_store,
        })
    }

    pub async fn run(self) -> Result<(), Box<dyn std::error::Error>> {
        let addr: SocketAddr = self.config.listen.parse()?;
        let server = Arc::new(self);

        let routes = create_routes(server.clone()).recover(handle_rejection);

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

        let routes_with_cors = routes.with(cors);

        if LogConfig::get().debug_enabled {
            log::info!("starting proxy server on {} (debug mode)", addr);
        } else {
            log::info!("starting proxy server on {}", addr);
        }
        log::info!("LM Studio backend: {}", server.config.lmstudio_url);

        warp::serve(routes_with_cors).run(addr).await;

        Ok(())
    }
}

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
