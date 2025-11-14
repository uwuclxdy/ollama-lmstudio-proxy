/// src/main.rs - Application entry point for the Ollama-LMStudio proxy server.

// Core modules
mod common;
mod constants;
mod handlers;
mod model;
mod model_legacy;
mod server;
mod utils;

// Bring into scope for main
use clap::Parser;
use server::{Config, ProxyServer};

/// Version information for the application
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Name of the application
pub const NAME: &str = env!("CARGO_PKG_NAME");

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::parse();
    let server = ProxyServer::new(config)?;
    server.run().await?;
    Ok(())
}

#[cfg(test)]
mod tests;
