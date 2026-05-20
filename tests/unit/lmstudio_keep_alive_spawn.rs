use super::*;
// handlers/ollama/keep_alive.rs
//
// keep_alive_parse re-exports (parse_keep_alive_seconds, apply_keep_alive_ttl,
// keep_alive_requests_unload) are already covered by
// tests/unit/handlers_ollama_keep_alive_parse.rs.
//
// The remaining public function is spawn_model_unload_if_needed.
// Its early-return guard —
//   `if !matches!(keep_alive_seconds, Some(0)) { return; }`
// — is the only logic observable without a network.  When keep_alive != 0
// the function returns immediately without spawning anything.
// When keep_alive == 0 it spawns a task that tries to reach LM Studio;
// that path is an integration concern and is skipped here.

fn make_resolver() -> std::sync::Arc<crate::model::ModelResolver> {
    use moka::future::Cache;
    let cache: Cache<String, String> = Cache::builder().max_capacity(128).build();
    let resolver = crate::model::ModelResolver::new("http://127.0.0.1:0".to_string(), cache);
    std::sync::Arc::new(resolver)
}

#[test]
fn spawn_model_unload_noop_when_keep_alive_none() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(async {
        spawn_model_unload_if_needed(
            reqwest::Client::new(),
            "http://localhost:1234".to_string(),
            make_resolver(),
            "llama3".to_string(),
            None,
            0,
        );
    });
}

#[test]
fn spawn_model_unload_noop_when_keep_alive_positive() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(async {
        spawn_model_unload_if_needed(
            reqwest::Client::new(),
            "http://localhost:1234".to_string(),
            make_resolver(),
            "llama3".to_string(),
            Some(300),
            0,
        );
    });
}

#[test]
fn spawn_model_unload_noop_when_keep_alive_negative() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(async {
        spawn_model_unload_if_needed(
            reqwest::Client::new(),
            "http://localhost:1234".to_string(),
            make_resolver(),
            "llama3".to_string(),
            Some(-1),
            0,
        );
    });
}
