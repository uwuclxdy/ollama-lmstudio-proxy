// Integration tests for the experimental `--use-native-chat` path.
//
// With the flag on, POST /api/chat routes through LM Studio's native
// `/api/v1/chat` endpoint instead of the OpenAI-compat `/api/v0/chat/completions`.
// These tests boot a proxy via `spawn_proxy_with_native()` and assert:
//   - the request reaches the backend as a native body (model remapped, `input`
//     array) and the native `{output, stats}` response is converted to Ollama
//     shape with timing fields;
//   - a mocked native `event:`/`data:` SSE stream is converted to Ollama NDJSON
//     chunks (content + thinking), ending with a `done:true` chunk.
//
// The default (flag-off) routing to /api/v0/chat/completions stays covered by the
// existing `ollama_chat` integration suite.

use serde_json::{Value, json};
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, ResponseTemplate};

use crate::common::spawn_proxy_with_native;

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
// Non-streaming: native request body + response conversion
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn non_streaming_routes_to_native_and_converts_response() {
    let p = spawn_proxy_with_native().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    // Backend must receive the NATIVE body: resolved model id + `input` array
    // (NOT the OpenAI `messages` array). Native input items are `{type:"text",
    // content}` with no role key — LM Studio rejects `role` and `type:"message"`.
    Mock::given(method("POST"))
        .and(path("/api/v1/chat"))
        .and(body_partial_json(json!({
            "model": "llama3.1-8b-instruct",
            "stream": false,
            "input": [{ "type": "text", "content": "Hi" }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model_instance_id": "llama3.1-8b-instruct",
            "output": [
                { "type": "reasoning", "content": "thinking about it" },
                { "type": "message", "content": "Hello there!" }
            ],
            "stats": {
                "input_tokens": 12,
                "total_output_tokens": 8,
                "tokens_per_second": 40.0,
                "time_to_first_token_seconds": 0.1
            },
            "response_id": "resp_abc123"
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
        .expect("POST /api/chat (native)");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("JSON body");

    assert_eq!(body["model"], "llama3.1:8b");
    assert_eq!(body["done"], true);
    assert_eq!(body["done_reason"], "stop");
    assert_eq!(body["message"]["role"], "assistant");
    assert_eq!(body["message"]["content"], "Hello there!");
    assert_eq!(body["message"]["thinking"], "thinking about it");
    // The backend's non-Ollama `response_id` must be stripped from the response.
    assert!(body.get("response_id").is_none());
    assert!(body["total_duration"].is_number());
    assert!(
        body["prompt_eval_count"].is_number(),
        "prompt_eval_count must be present"
    );
    assert!(body["eval_count"].is_number(), "eval_count must be present");
    assert!(body["created_at"].is_string());
}

// ═══════════════════════════════════════════════════════════════════════════
// Integrations passthrough
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn integrations_forwarded_verbatim_to_native_backend() {
    let p = spawn_proxy_with_native().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    // Backend must receive the `integrations` array exactly as sent.
    Mock::given(method("POST"))
        .and(path("/api/v1/chat"))
        .and(body_partial_json(json!({
            "integrations": [
                "huggingface",
                { "type": "plugin", "id": "browser" },
                {
                    "type": "ephemeral_mcp",
                    "server_label": "hf",
                    "server_url": "https://hf.co/mcp"
                }
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model_instance_id": "llama3.1-8b-instruct",
            "output": [{ "type": "message", "content": "ok" }],
            "stats": {
                "input_tokens": 5,
                "total_output_tokens": 1,
                "tokens_per_second": 40.0,
                "time_to_first_token_seconds": 0.1
            }
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
            "stream": false,
            "integrations": [
                "huggingface",
                { "type": "plugin", "id": "browser" },
                {
                    "type": "ephemeral_mcp",
                    "server_label": "hf",
                    "server_url": "https://hf.co/mcp"
                }
            ]
        }))
        .send()
        .await
        .expect("POST /api/chat with integrations");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("JSON body");
    assert_eq!(body["message"]["content"], "ok");
}

#[tokio::test]
async fn non_array_integrations_not_forwarded() {
    let p = spawn_proxy_with_native().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    // Backend receives the request but without any `integrations` key.
    Mock::given(method("POST"))
        .and(path("/api/v1/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model_instance_id": "llama3.1-8b-instruct",
            "output": [{ "type": "message", "content": "ok" }],
            "stats": {
                "input_tokens": 5,
                "total_output_tokens": 1,
                "tokens_per_second": 40.0,
                "time_to_first_token_seconds": 0.1
            }
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    // Send `integrations` as a non-array (object) — must be silently dropped.
    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "messages": [{ "role": "user", "content": "Hi" }],
            "stream": false,
            "integrations": { "type": "plugin", "id": "browser" }
        }))
        .send()
        .await
        .expect("POST /api/chat with non-array integrations");

    assert_eq!(resp.status(), 200);
}

// ═══════════════════════════════════════════════════════════════════════════
// Streaming: native SSE events → Ollama NDJSON chunks
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn streaming_native_sse_converts_to_ollama_ndjson() {
    let p = spawn_proxy_with_native().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    // Native wire format: `event: <type>\ndata: <json>` blocks separated by \n\n.
    let sse_payload = concat!(
        "event: chat.start\ndata: {\"type\":\"chat.start\",\"model_instance_id\":\"llama3.1-8b-instruct\"}\n\n",
        "event: reasoning.delta\ndata: {\"type\":\"reasoning.delta\",\"content\":\"hmm\"}\n\n",
        "event: message.delta\ndata: {\"type\":\"message.delta\",\"content\":\"Hello\"}\n\n",
        "event: message.delta\ndata: {\"type\":\"message.delta\",\"content\":\" world\"}\n\n",
        "event: chat.end\ndata: {\"type\":\"chat.end\",\"result\":{\"model_instance_id\":\"llama3.1-8b-instruct\",\"output\":[{\"type\":\"message\",\"content\":\"Hello world\"}],\"stats\":{\"input_tokens\":5,\"total_output_tokens\":2,\"tokens_per_second\":40.0,\"time_to_first_token_seconds\":0.1},\"response_id\":\"resp_xyz\"}}\n\n"
    );

    Mock::given(method("POST"))
        .and(path("/api/v1/chat"))
        .and(body_partial_json(json!({ "stream": true })))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(sse_payload.as_bytes(), "text/event-stream"),
        )
        .expect(1)
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
        .expect("POST /api/chat streaming (native)");

    assert_eq!(resp.status(), 200);
    let text = resp.text().await.expect("body text");
    let chunks = parse_ndjson(&text);

    assert!(
        chunks.len() >= 4,
        "expected delta + final chunks, got {chunks:?}"
    );

    // Reasoning delta surfaces as a `thinking` fragment.
    let thinking: String = chunks
        .iter()
        .filter_map(|c| c["message"]["thinking"].as_str())
        .collect();
    assert_eq!(thinking, "hmm");

    // Message deltas concatenate to the full content.
    let content: String = chunks
        .iter()
        .filter_map(|c| c["message"]["content"].as_str())
        .collect();
    assert_eq!(content, "Hello world");

    // Every chunk but the last is `done:false`; the last is the terminal chunk.
    let final_chunk = chunks.last().expect("at least one chunk");
    assert_eq!(final_chunk["done"], true);
    assert_eq!(final_chunk["done_reason"], "stop");
    // The backend's non-Ollama `response_id` must be stripped from the chunk.
    assert!(final_chunk.get("response_id").is_none());
    assert!(final_chunk["total_duration"].is_number());
    assert!(final_chunk["eval_count"].is_number());

    for chunk in &chunks[..chunks.len() - 1] {
        assert_eq!(chunk["done"], false, "non-final chunk must be done:false");
    }
}
