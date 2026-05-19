pub mod config;
pub mod constants;
pub mod error;
pub mod handlers;
pub mod http;
pub mod logging;
pub mod model;
pub mod server;
pub mod storage;
pub mod streaming;
pub mod update;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
