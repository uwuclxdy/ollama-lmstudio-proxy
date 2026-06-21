#![allow(clippy::unwrap_used, clippy::expect_used)]
// Integration tests for the proactive auto-evict behaviour.
//
// When `--auto-evict` is on and the target model is NOT yet loaded, the proxy
// must issue `POST /api/v1/models/unload` for every loaded instance of every
// OTHER model BEFORE the inference attempt. When the target IS already loaded,
// no unload call should be issued. When the flag is off, no extra work at all.

use serde_json::{Value, json};
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, ResponseTemplate};

use crate::common::{spawn_proxy, spawn_proxy_with_auto_evict};

// ── helpers ──────────────────────────────────────────────────────────────────

/// Two-model catalog: `target_key` is NOT loaded, `other_key` has one instance.
fn catalog_target_unloaded(target_key: &str, other_key: &str) -> Value {
    json!({
        "models": [
            {
                "key": target_key,
                "type": "llm",
                "publisher": "meta",
                "architecture": "llama",
                "format": "gguf",
                "quantization": { "name": "Q4_K_M", "bits_per_weight": 4.5 },
                "max_context_length": 8192,
                "loaded_instances": []
            },
            {
                "key": other_key,
                "type": "llm",
                "publisher": "mistral",
                "architecture": "mistral",
                "format": "gguf",
                "quantization": { "name": "Q4_K_M", "bits_per_weight": 4.5 },
                "max_context_length": 8192,
                "loaded_instances": [
                    { "id": "other-inst-0", "config": { "context_length": 4096 } }
                ]
            }
        ]
    })
}

/// Two-model catalog: `target_key` IS loaded with one instance, `other_key` also loaded.
fn catalog_target_loaded(target_key: &str, other_key: &str) -> Value {
    json!({
        "models": [
            {
                "key": target_key,
                "type": "llm",
                "publisher": "meta",
                "architecture": "llama",
                "format": "gguf",
                "quantization": { "name": "Q4_K_M", "bits_per_weight": 4.5 },
                "max_context_length": 8192,
                "loaded_instances": [
                    { "id": "target-inst-0", "config": { "context_length": 4096 } }
                ]
            },
            {
                "key": other_key,
                "type": "llm",
                "publisher": "mistral",
                "architecture": "mistral",
                "format": "gguf",
                "quantization": { "name": "Q4_K_M", "bits_per_weight": 4.5 },
                "max_context_length": 8192,
                "loaded_instances": [
                    { "id": "other-inst-0", "config": { "context_length": 4096 } }
                ]
            }
        ]
    })
}

fn lm_chat_ok() -> Value {
    json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1_700_000_000u64,
        "model": "llama3.1-8b-instruct",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "hello" },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6 },
        "stats": {}
    })
}

// ── tests ─────────────────────────────────────────────────────────────────────

// auto-evict ON, target NOT loaded, another model loaded → proxy must call
// POST /api/v1/models/unload for the other model's instance before inference.
#[tokio::test]
async fn auto_evict_unloads_other_when_target_not_loaded() {
    let p = spawn_proxy_with_auto_evict().await;

    // The proxy GETs /api/v1/models for resolution + proactive eviction.
    // Wiremock matches by order; mount with .mount (unlimited) so both the
    // resolution call and the eviction call are served.
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(catalog_target_unloaded(
                "llama3.1-8b-instruct",
                "mistral-7b-instruct",
            )),
        )
        .mount(&p.mock)
        .await;

    // The unload of the other model's instance must be called exactly once.
    Mock::given(method("POST"))
        .and(path("/api/v1/models/unload"))
        .and(body_partial_json(json!({ "instance_id": "other-inst-0" })))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({ "instance_id": "other-inst-0" })),
        )
        .expect(1)
        .mount(&p.mock)
        .await;

    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lm_chat_ok()))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "messages": [{ "role": "user", "content": "hi" }],
            "stream": false
        }))
        .send()
        .await
        .expect("POST /api/chat");

    assert_eq!(resp.status(), 200);
    p.mock.verify().await;
}

// auto-evict ON, target ALREADY loaded → no unload call issued at all.
#[tokio::test]
async fn auto_evict_skips_unload_when_target_already_loaded() {
    let p = spawn_proxy_with_auto_evict().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(catalog_target_loaded(
                "llama3.1-8b-instruct",
                "mistral-7b-instruct",
            )),
        )
        .mount(&p.mock)
        .await;

    // Must not be called.
    Mock::given(method("POST"))
        .and(path("/api/v1/models/unload"))
        .respond_with(ResponseTemplate::new(200))
        .expect(0)
        .mount(&p.mock)
        .await;

    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lm_chat_ok()))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "messages": [{ "role": "user", "content": "hi" }],
            "stream": false
        }))
        .send()
        .await
        .expect("POST /api/chat");

    assert_eq!(resp.status(), 200);
    p.mock.verify().await;
}

// auto-evict OFF → no unload call and no extra models-list GET beyond what
// model resolution already issues.
//
// The standard spawn_proxy has auto_evict:false. We mount an unload mock with
// expect(0) to assert nothing is evicted. We also verify the request succeeds
// normally (regression guard).
#[tokio::test]
async fn auto_evict_off_issues_no_unload() {
    let p = spawn_proxy().await;

    // Single-model catalog (target loaded) — standard mock used by most tests.
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [{
                "key": "llama3.1-8b-instruct",
                "type": "llm",
                "publisher": "meta",
                "architecture": "llama",
                "format": "gguf",
                "quantization": { "name": "Q4_K_M", "bits_per_weight": 4.5 },
                "max_context_length": 8192,
                "loaded_instances": [
                    { "id": "inst-0", "config": { "context_length": 4096 } }
                ]
            }]
        })))
        .mount(&p.mock)
        .await;

    Mock::given(method("POST"))
        .and(path("/api/v1/models/unload"))
        .respond_with(ResponseTemplate::new(200))
        .expect(0)
        .mount(&p.mock)
        .await;

    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lm_chat_ok()))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "messages": [{ "role": "user", "content": "hi" }],
            "stream": false
        }))
        .send()
        .await
        .expect("POST /api/chat");

    assert_eq!(resp.status(), 200);
    p.mock.verify().await;
}
