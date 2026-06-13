// Integration tests for upstream HTTP error status code forwarding and
// mid-stream pull error chunk shape — per api-docs/ollama/api/errors.md.
//
// Drift A: proxy must forward upstream 429 (Too Many Requests) and 502 (Bad
//   Gateway) instead of collapsing them to 503.
// Drift B: mid-stream pull error chunk must be bare {"error":"..."} with no
//   extra fields (status, model, etc.).

use serde_json::{Value, json};
use wiremock::matchers::{method, path, path_regex};
use wiremock::{Mock, ResponseTemplate};

use crate::common::spawn_proxy;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn loaded_model_entry(key: &str) -> Value {
    json!({
        "key": key,
        "type": "llm",
        "publisher": "meta",
        "architecture": "llama",
        "format": "gguf",
        "quantization": { "name": "Q4_K_M", "bits_per_weight": 4.5 },
        "max_context_length": 8192,
        "loaded_instances": [
            { "id": "inst-0", "config": { "context_length": 4096 } }
        ],
        "capabilities": { "vision": false, "trained_for_tool_use": false }
    })
}

fn lms_models(models: Vec<Value>) -> Value {
    json!({ "models": models })
}

fn native_model(key: &str) -> Value {
    json!({
        "key": key,
        "type": "llm",
        "publisher": "test-publisher",
        "architecture": "llama",
        "format": "gguf",
        "quantization": { "name": "Q4_K_M", "bits_per_weight": 4.5 },
        "max_context_length": 8192,
        "loaded_instances": [],
        "capabilities": { "vision": false, "trained_for_tool_use": false }
    })
}

async fn mount_model_catalog(proxy: &crate::common::TestProxy, model_key: &str) {
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(lms_models(vec![loaded_model_entry(model_key)])),
        )
        .mount(&proxy.mock)
        .await;
}

// ---------------------------------------------------------------------------
// Drift A — upstream 429 is forwarded as 429
// ---------------------------------------------------------------------------

#[tokio::test]
async fn upstream_429_is_forwarded_on_chat() {
    // When LM Studio returns 429, the proxy must not remap to 503. The Ollama
    // errors doc lists 429 as a possible status code (rate limit exceeded).
    let p = spawn_proxy().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(429).set_body_json(json!({"error": "rate limit exceeded"})),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "messages": [{ "role": "user", "content": "Hi" }],
            "stream": false
        }))
        .send()
        .await
        .expect("POST /api/chat 429");

    assert_eq!(
        resp.status(),
        429,
        "upstream 429 must be forwarded; got {}",
        resp.status()
    );
}

// ---------------------------------------------------------------------------
// Drift A — upstream 502 is forwarded as 502
// ---------------------------------------------------------------------------

#[tokio::test]
async fn upstream_502_is_forwarded_on_chat() {
    // When LM Studio returns 502 (cloud model unreachable), the proxy must
    // forward 502, not remap to 503.
    let p = spawn_proxy().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(502).set_body_json(json!({"error": "cloud model unreachable"})),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "messages": [{ "role": "user", "content": "Hi" }],
            "stream": false
        }))
        .send()
        .await
        .expect("POST /api/chat 502");

    assert_eq!(
        resp.status(),
        502,
        "upstream 502 must be forwarded; got {}",
        resp.status()
    );
}

// ---------------------------------------------------------------------------
// Drift B — mid-stream pull error chunk is bare {"error":"..."}
// ---------------------------------------------------------------------------

#[tokio::test]
async fn pull_stream_error_chunk_is_bare_error_object() {
    // When LM Studio's download endpoint fails mid-stream (e.g. status poll
    // returns an error), the proxy must emit a chunk with only the "error"
    // key — no "status", "model", or any other field.
    // Per api-docs/ollama/api/errors.md §"Errors that occur while streaming".
    let p = spawn_proxy().await;

    // Initiate download — LM Studio starts a download job.
    Mock::given(method("POST"))
        .and(path("/api/v1/models/download"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "job_id": "job-fail",
            "status": "downloading",
            "total_size_bytes": 4_000_000_000u64,
            "downloaded_bytes": 0,
            "started_at": "2026-01-01T00:00:00Z"
        })))
        .mount(&p.mock)
        .await;

    // Status poll returns a failed state.
    Mock::given(method("GET"))
        .and(path_regex(r"^/api/v1/models/download/status/.*"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "job_id": "job-fail",
            "status": "failed",
            "error": "disk full"
        })))
        .mount(&p.mock)
        .await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lms_models(vec![native_model("llama3.2:3b")])),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/pull"))
        .json(&json!({"model": "llama3.2:3b", "stream": true}))
        .send()
        .await
        .expect("POST /api/pull stream=true failure");

    // The HTTP response is 200 (stream already started).
    assert_eq!(resp.status(), 200);

    let body_bytes = resp.bytes().await.expect("body bytes");
    let text = String::from_utf8_lossy(&body_bytes);

    // Find the last non-empty NDJSON line.
    let last_line = text
        .lines()
        .rfind(|l| !l.trim().is_empty())
        .expect("at least one NDJSON line");

    let last_chunk: Value = serde_json::from_str(last_line)
        .unwrap_or_else(|e| panic!("last NDJSON line is not valid JSON: {e}; line='{last_line}'"));

    let obj = last_chunk.as_object().expect("chunk must be object");

    // Must have "error" key.
    assert!(
        obj.contains_key("error"),
        "error chunk must have 'error' key; got {last_chunk}"
    );

    // Must NOT have "status" or "model".
    assert!(
        !obj.contains_key("status"),
        "error chunk must not have 'status' key; got {last_chunk}"
    );
    assert!(
        !obj.contains_key("model"),
        "error chunk must not have 'model' key; got {last_chunk}"
    );

    // Must be exactly one key: {"error":"..."}.
    assert_eq!(
        obj.len(),
        1,
        "error chunk must have only 'error' key; got {last_chunk}"
    );
}
