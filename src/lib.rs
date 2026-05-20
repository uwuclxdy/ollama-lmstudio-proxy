pub mod api;
pub mod config;
pub mod constants;
pub mod error;
pub mod http;
pub mod lmstudio;
pub mod logging;
pub mod model;
pub mod proxy;
pub mod storage;
pub mod streaming;
pub mod update;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
