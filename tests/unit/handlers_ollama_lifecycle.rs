use serde_json::json;

// handlers/ollama/lifecycle.rs
//
// All public functions (handle_ollama_pull, handle_ollama_create,
// handle_ollama_copy, handle_ollama_delete, handle_ollama_push) are
// async handlers that require a live RequestContext with network access
// and a running tokio runtime with a VirtualModelStore.
//
// Pure logic embedded in the handlers that we can test directly:
//   - model name extraction from the request body (via extract_required_model_name)
//   - `source_model_name` fallback: body["from"] ?? model_name
//   - `stream` flag extraction: body["stream"].as_bool().unwrap_or(true)
//   - `quantization` extraction: body["quantization"].as_str()
//   - `source` override extraction: body["source"].as_str()
//
// These are tested by exercising the relevant serde_json patterns,
// mirroring exactly what the handlers do.

fn source_from_body<'a>(body: &'a serde_json::Value, model_name: &'a str) -> &'a str {
    body.get("from")
        .and_then(|v| v.as_str())
        .unwrap_or(model_name)
}

fn stream_from_body(body: &serde_json::Value) -> bool {
    body.get("stream").and_then(|v| v.as_bool()).unwrap_or(true)
}

fn quantization_from_body(body: &serde_json::Value) -> Option<String> {
    body.get("quantization")
        .and_then(|q| q.as_str())
        .map(|s| s.to_string())
}

fn source_override_from_body(body: &serde_json::Value) -> Option<&str> {
    body.get("source").and_then(|s| s.as_str())
}

// --- source_model_name ---

#[test]
fn source_falls_back_to_model_name_when_from_absent() {
    let body = json!({"model": "myalias"});
    assert_eq!(source_from_body(&body, "myalias"), "myalias");
}

#[test]
fn source_uses_from_field_when_present() {
    let body = json!({"model": "myalias", "from": "llama3"});
    assert_eq!(source_from_body(&body, "myalias"), "llama3");
}

#[test]
fn source_ignores_non_string_from() {
    let body = json!({"model": "alias", "from": 42});
    assert_eq!(source_from_body(&body, "alias"), "alias");
}

// --- stream flag ---

#[test]
fn stream_defaults_to_true_when_absent() {
    assert!(stream_from_body(&json!({"model": "m"})));
}

#[test]
fn stream_explicit_true() {
    assert!(stream_from_body(&json!({"stream": true})));
}

#[test]
fn stream_explicit_false() {
    assert!(!stream_from_body(&json!({"stream": false})));
}

#[test]
fn stream_non_bool_falls_back_to_true() {
    assert!(stream_from_body(&json!({"stream": "yes"})));
}

// --- quantization ---

#[test]
fn quantization_absent_returns_none() {
    assert!(quantization_from_body(&json!({"model": "m"})).is_none());
}

#[test]
fn quantization_present() {
    let q = quantization_from_body(&json!({"quantization": "Q4_K_M"}));
    assert_eq!(q.as_deref(), Some("Q4_K_M"));
}

#[test]
fn quantization_non_string_returns_none() {
    assert!(quantization_from_body(&json!({"quantization": 4})).is_none());
}

// --- source override (pull) ---

#[test]
fn source_override_absent_returns_none() {
    assert!(source_override_from_body(&json!({"model": "m"})).is_none());
}

#[test]
fn source_override_present() {
    let body = json!({"source": "hf://org/model"});
    assert_eq!(source_override_from_body(&body), Some("hf://org/model"));
}

#[test]
fn source_override_non_string_returns_none() {
    let body = json!({"source": true});
    assert!(source_override_from_body(&body).is_none());
}
