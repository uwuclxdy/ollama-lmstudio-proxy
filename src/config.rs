use std::sync::OnceLock;

use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(name = "ollama-lmstudio-proxy")]
#[command(about = "high-performance proxy server bridging ollama API and lm studio")]
pub struct Config {
    #[arg(long, default_value = "0.0.0.0:11434", help = "server listen address")]
    pub listen: String,

    #[arg(
        long,
        default_value = "http://localhost:1234",
        help = "lm studio backend url"
    )]
    pub lmstudio_url: String,

    #[arg(
        long,
        env = "RUST_LOG",
        default_value = "info",
        help = "log level (off, error, warn, info, debug, trace); also reads RUST_LOG"
    )]
    pub log_level: String,

    #[arg(
        long,
        default_value = "15",
        help = "model loading wait timeout in seconds (after trigger)"
    )]
    pub load_timeout_seconds: u64,

    #[arg(
        long,
        default_value = "262144",
        help = "initial buffer size in bytes for sse message assembly (capacity hint)"
    )]
    pub max_buffer_size: usize,

    #[arg(long, help = "enable partial chunk recovery for streams")]
    pub enable_chunk_recovery: bool,

    #[arg(
        long,
        default_value = "300",
        help = "ttl for model resolution cache in seconds"
    )]
    pub model_resolution_cache_ttl_seconds: u64,

    #[arg(
        long,
        env = "LMSTUDIO_TOKEN",
        help = "bearer token for LM Studio authentication (sets Authorization header on all outbound requests)"
    )]
    pub lmstudio_token: Option<String>,

    #[arg(
        long,
        help = "experimental: route /api/chat through LM Studio native /api/v1/chat for richer reasoning events and accurate stats"
    )]
    pub use_native_chat: bool,

    #[arg(
        long,
        help = "experimental: enable flash attention when loading models via /api/v1/models/load"
    )]
    pub flash_attention: bool,

    #[arg(
        long,
        help = "experimental: offload KV cache to GPU when loading models via /api/v1/models/load"
    )]
    pub offload_kv_cache: bool,

    #[arg(
        long,
        help = "experimental: set eval batch size when loading models via /api/v1/models/load"
    )]
    pub eval_batch_size: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub max_buffer_size: usize,
    pub enable_chunk_recovery: bool,
    pub flash_attention: bool,
    pub offload_kv_cache: bool,
    pub eval_batch_size: Option<u32>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_buffer_size: usize::MAX,
            enable_chunk_recovery: true,
            flash_attention: false,
            offload_kv_cache: false,
            eval_batch_size: None,
        }
    }
}

static RUNTIME_CONFIG: OnceLock<RuntimeConfig> = OnceLock::new();

pub fn init_runtime_config(config: RuntimeConfig) {
    RUNTIME_CONFIG.set(config).ok();
}

pub fn get_runtime_config() -> &'static RuntimeConfig {
    RUNTIME_CONFIG.get().unwrap_or_else(|| {
        static DEFAULT: OnceLock<RuntimeConfig> = OnceLock::new();
        DEFAULT.get_or_init(RuntimeConfig::default)
    })
}

pub fn validate_config(config: &Config) -> Result<(), String> {
    if config.listen.parse::<std::net::SocketAddr>().is_err() {
        return Err(format!("invalid listen address: {}", config.listen));
    }
    if !config.lmstudio_url.starts_with("http://") && !config.lmstudio_url.starts_with("https://") {
        return Err(format!(
            "invalid LM Studio URL (must start with http:// or https://): {}",
            config.lmstudio_url
        ));
    }
    if let Err(e) = url::Url::parse(&config.lmstudio_url) {
        return Err(format!("invalid LM Studio URL format: {}", e));
    }
    Ok(())
}
