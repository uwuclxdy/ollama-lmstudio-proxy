// handlers/ollama/lifecycle.rs
//
// All public functions (handle_ollama_pull, handle_ollama_create,
// handle_ollama_copy, handle_ollama_delete, handle_ollama_push) are
// async handlers that require a live RequestContext with network access
// and a running tokio runtime with a VirtualModelStore. Their integration
// behaviour is covered by tests/integration/ollama_pull_delete_copy.rs.
