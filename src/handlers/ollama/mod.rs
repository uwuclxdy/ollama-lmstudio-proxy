pub mod blobs;
pub mod chat;
pub mod embeddings;
pub mod generate;
pub mod health;
pub mod lifecycle;
pub mod models;
pub mod tags;
pub mod utils;

pub use blobs::{handle_blob_head, handle_blob_upload};
pub use chat::handle_ollama_chat;
pub use embeddings::{EmbeddingResponseMode, handle_ollama_embeddings};
pub use generate::handle_ollama_generate;
pub use health::{handle_health_check, handle_ollama_version};
pub use lifecycle::{
    handle_ollama_copy, handle_ollama_create, handle_ollama_delete, handle_ollama_pull,
    handle_ollama_push,
};
pub use models::{handle_ollama_ps, handle_ollama_show};
pub use tags::handle_ollama_tags;
