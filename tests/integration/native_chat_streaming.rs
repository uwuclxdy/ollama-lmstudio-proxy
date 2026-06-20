// Integration tests for the `--native-chat-streaming` auto mode.
//
// With the flag on, the chat handler routes based on the stream flag:
//   - `stream:true`  -> native `/api/v1/chat` (richer reasoning events, stats)
//   - `stream:false` -> the default OpenAI-compat `/api/v0/chat/completions`
//
// `use_native = use_native_chat || (native_chat_streaming && stream)` (see
// `src/api/ollama/chat.rs`). These tests pin BOTH halves of that expression:
// streaming lands on `/api/v1/chat`, and non-streaming stays on
// `/api/v0/chat/completions` instead of being pulled onto the native path.

use serde_json::{Value, json};
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, ResponseTemplate};

use crate::common::spawn_proxy_with_native_streaming;

/// GET /api/v1/models stub returning a single loaded model for resolution.
async fn mount_model_catalog(proxy: &crate::common::TestProxy, model_key: &str) {
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [{
                "key": model_key,
                "type": "llm",
                "publisher": "meta",
                "architecture": "llama",
                "format": "gguf",
                "quantization": { "name": "Q4_K_M", "bits_per_weight": 4.5 },
                "max_context_length": 8192,
                "loaded_instances": [
                    { "id": "inst-0", "config": { "context_length": 4096 } }
                ],
                "capabilities": { "vision": false, "trained_for_tool_use": true }
            }]
        })))
        .mount(&proxy.mock)
        .await;
}

/// Parse NDJSON body into a `Vec<Value>`, skipping blank lines.
fn parse_ndjson(text: &str) -> Vec<Value> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("valid JSON line"))
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// stream:true -> native /api/v1/chat
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn streaming_routes_to_native_v1_chat() {
    let p = spawn_proxy_with_native_streaming().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    // Pin the route: the streaming request must land on /api/v1/chat with
    // stream:true in the body — NOT on the OpenAI-compat v0 path.
    Mock::given(method("POST"))
        .and(path("/api/v1/chat"))
        .and(body_partial_json(json!({ "stream": true })))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(
                concat!(
                    "event: chat.start\ndata: {\"type\":\"chat.start\",\"model_instance_id\":\"llama3.1-8b-instruct\"}\n\n",
                    "event: message.delta\ndata: {\"type\":\"message.delta\",\"content\":\"Hello\"}\n\n",
                    "event: chat.end\ndata: {\"type\":\"chat.end\",\"result\":{\"model_instance_id\":\"llama3.1-8b-instruct\",\"output\":[{\"type\":\"message\",\"content\":\"Hello\"}],\"stats\":{\"input_tokens\":5,\"total_output_tokens\":1,\"tokens_per_second\":40.0,\"time_to_first_token_seconds\":0.1}}}\n\n"
                )
                .as_bytes(),
                "text/event-stream",
            ),
        )
        .expect(1)
        .mount(&p.mock)
        .await;

    // If routing misfires onto the v0 path, this mock would go unmatched and
    // the request would 404/500 — making the bug obvious.
    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"_unexpected_v0": true})))
        .expect(0)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "messages": [{ "role": "user", "content": "Hi" }],
            "stream": true
        }))
        .send()
        .await
        .expect("POST /api/chat streaming");

    assert_eq!(resp.status(), 200);
    let text = resp.text().await.expect("body text");
    let chunks = parse_ndjson(&text);
    assert!(!chunks.is_empty(), "stream must produce NDJSON chunks");

    let final_chunk = chunks.last().expect("terminal chunk");
    assert_eq!(final_chunk["done"], true, "final chunk must be done:true");
    assert!(
        final_chunk["eval_count"].is_number(),
        "terminal chunk must carry stats"
    );

    p.mock.verify().await;
}

// ═══════════════════════════════════════════════════════════════════════════
// stream:false -> stays on the default /api/v0/chat/completions
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn non_streaming_stays_on_v0_chat_completions() {
    let p = spawn_proxy_with_native_streaming().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    // Non-streaming must NOT hit the native path — expect(0) makes a misroute fail.
    Mock::given(method("POST"))
        .and(path("/api/v1/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"_unexpected_native": true})))
        .expect(0)
        .mount(&p.mock)
        .await;

    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "OK" },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3 }
        })))
        .expect(1)
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
        .expect("POST /api/chat non-streaming");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("JSON body");
    assert_eq!(body["done"], true);
    assert_eq!(body["message"]["content"], "OK");

    p.mock.verify().await;
}
