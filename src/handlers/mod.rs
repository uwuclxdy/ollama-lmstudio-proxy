pub mod helpers;
pub mod lmstudio;
pub mod ollama;
/// src/handlers/mod.rs - Module exports for API endpoint handlers with native and legacy support
pub mod retry;
pub mod streaming;

// Ollama handler exports with enhanced signatures for dual API support

// LM Studio handler exports with dual API support

// Streaming handler exports

// Retry handler exports

// Helper exports with enhanced native API support
pub use helpers::json_response;
