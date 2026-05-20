// Top-level entry for all wiremock-driven integration tests.
// Each submodule under `tests/integration/` exercises a different surface.
//
// Submodules are added by integration-test agents — keep this file in sync.

mod common;

#[path = "integration/ollama_chat.rs"]
mod ollama_chat;

#[path = "integration/ollama_generate.rs"]
mod ollama_generate;

#[path = "integration/ollama_embed.rs"]
mod ollama_embed;

#[path = "integration/ollama_tags_show_ps.rs"]
mod ollama_tags_show_ps;

#[path = "integration/ollama_pull_delete_copy.rs"]
mod ollama_pull_delete_copy;

#[path = "integration/ollama_create_blobs.rs"]
mod ollama_create_blobs;

#[path = "integration/lmstudio_openai.rs"]
mod lmstudio_openai;

#[path = "integration/lmstudio_native.rs"]
mod lmstudio_native;

#[path = "integration/streaming.rs"]
mod streaming;

#[path = "integration/server_routes.rs"]
mod server_routes;

#[path = "integration/ollama_errors.rs"]
mod ollama_errors;
