use clap::Parser;

mod config;
mod constants;
mod error;
mod handlers;
mod http;
mod logging;
mod model;
mod server;
mod storage;
mod streaming;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = config::Config::parse();

    config::validate_config(&cfg)?;

    setup_logging(&cfg.log_level)?;

    let debug_enabled =
        cfg.log_level.eq_ignore_ascii_case("debug") || cfg.log_level.eq_ignore_ascii_case("trace");
    logging::LogConfig::init(debug_enabled);

    config::init_runtime_config(config::RuntimeConfig {
        max_buffer_size: cfg.max_buffer_size,
        enable_chunk_recovery: cfg.enable_chunk_recovery,
    });

    let server = server::ProxyServer::new(cfg)?;
    server.run().await
}

fn setup_logging(log_level: &str) -> Result<(), Box<dyn std::error::Error>> {
    let level = log_level
        .to_lowercase()
        .parse::<log::LevelFilter>()
        .unwrap_or(log::LevelFilter::Info);

    fern::Dispatch::new()
        .format(|out, message, record| {
            let level_str = match record.level() {
                log::Level::Error => "\x1b[1;31merror:\x1b[0m",
                log::Level::Warn => "\x1b[1;33mwarn:\x1b[0m",
                log::Level::Info => "\x1b[1;32minfo:\x1b[0m",
                log::Level::Debug => "\x1b[1;34mdebug:\x1b[0m",
                log::Level::Trace => "\x1b[1;35mtrace:\x1b[0m",
            };
            out.finish(format_args!("{} {}", level_str, message))
        })
        .level(level)
        .chain(std::io::stdout())
        .apply()?;

    Ok(())
}
