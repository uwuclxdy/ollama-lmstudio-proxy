pub mod blobs;
pub mod chat;
pub mod embeddings;
pub mod generate;
pub mod health;
pub mod lifecycle;
pub mod models;
pub mod resolution;
pub mod status_stream;
pub mod unload_only;

pub use blobs::{handle_blob_head, handle_blob_upload};
pub use chat::handle_ollama_chat;
pub use embeddings::{EmbeddingResponseMode, handle_ollama_embeddings};
pub use generate::handle_ollama_generate;
pub use health::{handle_health_check, handle_ollama_root, handle_ollama_version};
pub use lifecycle::{
    handle_ollama_copy, handle_ollama_create, handle_ollama_delete, handle_ollama_pull,
};
pub use models::{handle_ollama_ps, handle_ollama_show, handle_ollama_tags};
