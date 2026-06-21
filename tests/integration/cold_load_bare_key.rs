#![allow(clippy::unwrap_used, clippy::expect_used)]
// Regression test: the explicit-load POST on the JIT-on-error cold-load path
// must use the bare LM Studio model key ("some-model"), not the Ollama name
// with its ":latest" tag ("some-model:latest").
//
// Before the fix, `trigger_model_loading` sent the raw Ollama name to
// `POST /api/v1/models/load`. LM Studio rejects that with 404
// `model_not_found`, so cold loads silently failed whenever LM Studio's own JIT
// was off — and embedders never loaded because the explicit load is their only
// path. The fix resolves the Ollama name to the bare LM Studio key first.
//
// The test forces the cold-load path by:
//   1. Returning 400 "No models loaded" on the first chat completions call.
//   2. Expecting exactly ONE `POST /api/v1/models/load` body carrying
//      `{"model":"some-model"}` (no ":latest"). If the proxy still sent
//      ":latest" the wiremock body matcher would not fire → .expect(1) fails
//      at p.mock.verify().
//   3. Returning 200 on the retry chat completions call.
//
// `load_timeout_seconds` is set to 1 so the sleep between trigger and retry is
// 1 second, keeping the test fast.

use serde_json::{Value, json};
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, ResponseTemplate};

use crate::common::spawn_proxy_with_load_timeout;

// The LM Studio bare key has no tag — confirming the resolver strips ":latest".
const MODEL_KEY: &str = "some-model";
// The Ollama client sends the name with the implicit ":latest" tag.
const OLLAMA_MODEL: &str = "some-model:latest";

fn unloaded_model_entry() -> Value {
    json!({
        "key": MODEL_KEY,
        "type": "llm",
        "publisher": "test",
        "architecture": "llama",
        "format": "gguf",
        "quantization": { "name": "Q4_K_M", "bits_per_weight": 4.5 },
        "max_context_length": 4096,
        "loaded_instances": [],
        "capabilities": { "vision": false, "trained_for_tool_use": false }
    })
}

fn lm_chat_ok() -> Value {
    json!({
        "id": "chatcmpl-cold",
        "object": "chat.completion",
        "created": 1_700_000_000u64,
        "model": MODEL_KEY,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "loaded and ready" },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8 }
    })
}

// The cold-load-on-error path triggers when the first chat attempt returns LM
// Studio's literal "No models loaded" 400. The proxy must:
//   a) POST /api/v1/models/load with {"model":"some-model"} (bare key, no tag)
//   b) sleep load_timeout_seconds
//   c) retry POST /api/v0/chat/completions → 200
//
// The bare-key assertion is enforced by the body_partial_json matcher on the
// load mock: if ":latest" leaks in, the matcher won't fire and .expect(1) fails.
#[tokio::test]
async fn cold_load_uses_bare_key_not_latest_tag() {
    // 1-second sleep between trigger and retry → test completes in ~1-2 s.
    let p = spawn_proxy_with_load_timeout(1).await;

    // Catalog: model exists but has no loaded instances (cold).
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [unloaded_model_entry()]
        })))
        .mount(&p.mock)
        .await;

    // First chat completions attempt: LM Studio says model not loaded.
    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "error": { "message": "No models loaded. Please load a model first." }
        })))
        .up_to_n_times(1)
        .expect(1)
        .mount(&p.mock)
        .await;

    // The explicit load must carry the BARE key — no ":latest". This is the
    // regression guard: a body mismatch leaves expect(1) unsatisfied → verify()
    // panics → the test fails, catching any regression back to the raw name.
    Mock::given(method("POST"))
        .and(path("/api/v1/models/load"))
        .and(body_partial_json(json!({ "model": MODEL_KEY })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "status": "loaded",
            "instance_id": MODEL_KEY,
            "load_time_seconds": 0.1
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    // Retry chat completions: succeeds after the model is loaded.
    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lm_chat_ok()))
        .mount(&p.mock)
        .await;

    // Use a longer client timeout: proxy sleeps load_timeout_seconds (1s) plus
    // the two round-trip calls before returning, so 10s total is enough.
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .expect("reqwest client");

    let resp = client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": OLLAMA_MODEL,
            "messages": [{ "role": "user", "content": "hello" }],
            "stream": false
        }))
        .send()
        .await
        .expect("POST /api/chat cold load");

    assert_eq!(
        resp.status(),
        200,
        "proxy must return 200 after cold-load retry"
    );

    let body: Value = resp.json().await.expect("JSON body");
    assert_eq!(
        body["message"]["content"], "loaded and ready",
        "response content must come from the retry success body"
    );
    assert_eq!(body["done"], true);

    // Verifies that load mock's .expect(1) was satisfied with the bare key body.
    p.mock.verify().await;
}
