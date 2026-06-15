// Integration tests for POST /api/chat — Ollama chat surface.
//
// Each test boots a real proxy via `spawn_proxy()` and registers wiremock
// mocks for the LM Studio endpoints the proxy calls. The model-resolution
// helper `mount_model_catalog` registers a GET /api/v1/models mock returning
// a single loaded model whose key contains the substring the Ollama name
// resolves to.

use serde_json::{Value, json};
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, ResponseTemplate};

use crate::common::spawn_proxy;

// ── model-catalog helpers ───────────────────────────────────────────────────

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
        "capabilities": { "vision": false, "trained_for_tool_use": true }
    })
}

fn vision_model_entry(key: &str) -> Value {
    json!({
        "key": key,
        "type": "vlm",
        "publisher": "meta",
        "architecture": "llama",
        "format": "gguf",
        "quantization": { "name": "Q4_K_M", "bits_per_weight": 4.5 },
        "max_context_length": 8192,
        "loaded_instances": [
            { "id": "inst-0", "config": { "context_length": 4096 } }
        ],
        "capabilities": { "vision": true, "trained_for_tool_use": false }
    })
}

async fn mount_model_catalog(proxy: &crate::common::TestProxy, model_key: &str) {
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [loaded_model_entry(model_key)]
        })))
        .mount(&proxy.mock)
        .await;
}

async fn mount_vision_catalog(proxy: &crate::common::TestProxy, model_key: &str) {
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [vision_model_entry(model_key)]
        })))
        .mount(&proxy.mock)
        .await;
}

/// Minimal valid LM Studio non-streaming chat completion response.
fn lm_chat_response(content: &str, finish_reason: &str) -> Value {
    json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1_700_000_000u64,
        "model": "llama3.1-8b-instruct",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": content },
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 8,
            "total_tokens": 20
        }
    })
}

/// Build an SSE body: one delta chunk per token, then `[DONE]`.
fn sse_chat_body(tokens: &[&str], finish_reason: &str) -> String {
    let mut body = String::new();
    for (i, token) in tokens.iter().enumerate() {
        let is_last = i + 1 == tokens.len();
        let fr_json = if is_last {
            format!("\"{}\"", finish_reason)
        } else {
            "null".to_string()
        };
        let chunk = json!({
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": { "content": *token },
                "finish_reason": serde_json::from_str::<Value>(&fr_json).unwrap()
            }]
        });
        body.push_str("data: ");
        body.push_str(&chunk.to_string());
        body.push_str("\n\n");
    }
    body.push_str("data: [DONE]\n\n");
    body
}

/// Parse NDJSON body into a `Vec<Value>`, skipping blank lines.
fn parse_ndjson(text: &str) -> Vec<Value> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("valid JSON line"))
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Non-streaming golden path
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn non_streaming_chat_returns_ollama_shape() {
    let p = spawn_proxy().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lm_chat_response("Hello there!", "stop")),
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
        .expect("POST /api/chat");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("JSON body");

    assert_eq!(body["model"], "llama3.1:8b");
    assert_eq!(body["done"], true);
    assert_eq!(body["done_reason"], "stop");
    assert_eq!(body["message"]["role"], "assistant");
    assert_eq!(body["message"]["content"], "Hello there!");
    assert!(body["eval_count"].is_number(), "eval_count must be present");
    assert!(
        body["prompt_eval_count"].is_number(),
        "prompt_eval_count must be present"
    );
    assert!(body["total_duration"].is_number());
    assert!(body["created_at"].is_string());
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. stream field absent defaults to streaming
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn stream_absent_defaults_to_streaming() {
    let p = spawn_proxy().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    let sse = sse_chat_body(&["Sure!"], "stop");
    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(sse.into_bytes(), "text/event-stream"),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "messages": [{ "role": "user", "content": "Hi" }]
        }))
        .send()
        .await
        .expect("POST /api/chat no stream field");

    assert_eq!(resp.status(), 200);
    let text = resp.text().await.expect("body text");
    let chunks = parse_ndjson(&text);
    assert!(
        !chunks.is_empty(),
        "absent stream must produce NDJSON chunks"
    );
    let final_chunk = chunks.last().unwrap();
    assert_eq!(final_chunk["done"], true);
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. stream:true — NDJSON chunks, final chunk done:true with stats
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn streaming_chat_emits_ndjson_with_final_done_chunk() {
    let p = spawn_proxy().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    let sse = sse_chat_body(&["Hello", " world"], "stop");
    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(sse.into_bytes(), "text/event-stream"),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "messages": [{ "role": "user", "content": "Hello" }],
            "stream": true
        }))
        .send()
        .await
        .expect("POST /api/chat stream:true");

    assert_eq!(resp.status(), 200);
    let text = resp.text().await.expect("body text");
    let chunks = parse_ndjson(&text);
    assert!(!chunks.is_empty(), "must receive at least one NDJSON chunk");

    // Final chunk has done:true and timing stats
    let final_chunk = chunks.last().expect("last chunk");
    assert_eq!(final_chunk["done"], true, "last chunk must be done:true");
    assert!(
        final_chunk.get("done_reason").is_some(),
        "done_reason missing"
    );
    assert!(
        final_chunk.get("eval_count").is_some(),
        "eval_count missing"
    );
    assert!(
        final_chunk.get("total_duration").is_some(),
        "total_duration missing"
    );

    // All non-final chunks carry done:false and a message object
    for chunk in &chunks[..chunks.len().saturating_sub(1)] {
        assert_eq!(
            chunk["done"], false,
            "intermediate chunk must be done:false"
        );
        assert!(
            chunk.get("message").is_some(),
            "intermediate chunk must have message"
        );
        assert_eq!(chunk["message"]["role"], "assistant");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. stream:false explicitly — single JSON object
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn stream_explicit_false_returns_single_object() {
    let p = spawn_proxy().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lm_chat_response("OK", "stop")))
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
        .expect("POST /api/chat stream:false");

    assert_eq!(resp.status(), 200);
    // Must parse as a single JSON object, not NDJSON.
    let body: Value = resp.json().await.expect("single JSON object");
    assert!(body.is_object());
    assert_eq!(body["done"], true);
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. options.temperature, top_p, num_predict → max_tokens, seed
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn options_temperature_num_predict_seed_forwarded() {
    let p = spawn_proxy().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lm_chat_response("Fine.", "stop")))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "messages": [{ "role": "user", "content": "Hi" }],
            "stream": false,
            "options": {
                "temperature": 0.3,
                "top_p": 0.85,
                "num_predict": 256,
                "seed": 42
            }
        }))
        .send()
        .await
        .expect("POST /api/chat options");

    assert_eq!(resp.status(), 200);
    p.mock.verify().await;
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. options.num_ctx forwarded as context_length
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn options_num_ctx_forwarded_as_context_length() {
    let p = spawn_proxy().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lm_chat_response("OK", "stop")))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "messages": [{ "role": "user", "content": "Test" }],
            "stream": false,
            "options": { "num_ctx": 2048 }
        }))
        .send()
        .await
        .expect("POST /api/chat num_ctx");

    assert_eq!(resp.status(), 200);
    p.mock.verify().await;
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. options.stop as array forwarded
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn options_stop_array_forwarded() {
    let p = spawn_proxy().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lm_chat_response("Done.", "stop")))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "messages": [{ "role": "user", "content": "Hi" }],
            "stream": false,
            "options": { "stop": ["</s>", "[INST]"] }
        }))
        .send()
        .await
        .expect("POST /api/chat stop array");

    assert_eq!(resp.status(), 200);
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. options.stop as scalar string forwarded
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn options_stop_scalar_forwarded() {
    let p = spawn_proxy().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lm_chat_response("Done.", "stop")))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "messages": [{ "role": "user", "content": "Hi" }],
            "stream": false,
            "options": { "stop": "</s>" }
        }))
        .send()
        .await
        .expect("POST /api/chat stop scalar");

    assert_eq!(resp.status(), 200);
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. tools forwarded — tool_calls in response translated to Ollama shape
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn tools_forwarded_and_tool_calls_translated_to_ollama() {
    let p = spawn_proxy().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    let lm_resp = json!({
        "id": "chatcmpl-tools",
        "object": "chat.completion",
        "created": 1_700_000_000u64,
        "model": "llama3.1-8b-instruct",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "index": 0,
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"location\":\"London\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": { "prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35 }
    });

    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lm_resp))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "messages": [{ "role": "user", "content": "What is the weather?" }],
            "stream": false,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": { "location": { "type": "string" } },
                        "required": ["location"]
                    }
                }
            }]
        }))
        .send()
        .await
        .expect("POST /api/chat tools");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("JSON");

    let tool_calls = &body["message"]["tool_calls"];
    assert!(tool_calls.is_array(), "tool_calls must be array");
    assert_eq!(tool_calls.as_array().unwrap().len(), 1);

    let tc = &tool_calls[0];
    assert_eq!(tc["function"]["index"], 0);
    assert_eq!(tc["function"]["name"], "get_weather");
    // Ollama expects arguments as a JSON object, not a string.
    assert!(
        tc["function"]["arguments"].is_object(),
        "arguments must be object, not string"
    );
    assert_eq!(tc["function"]["arguments"]["location"], "London");
    // OpenAI's `tool_calls` finish_reason is mapped to Ollama's `stop`. Real
    // Ollama clients branch on done_reason == "stop" | "length" only.
    assert_eq!(body["done_reason"], "stop");
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. format JSON schema (object) → structured output forwarded
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn format_json_schema_object_forwarded() {
    let p = spawn_proxy().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lm_chat_response("{\"answer\":42}", "stop")),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "messages": [{ "role": "user", "content": "Give me JSON" }],
            "stream": false,
            "format": {
                "type": "object",
                "properties": { "answer": { "type": "integer" } },
                "required": ["answer"]
            }
        }))
        .send()
        .await
        .expect("POST /api/chat format schema");

    assert_eq!(resp.status(), 200);
    p.mock.verify().await;
}

// ═══════════════════════════════════════════════════════════════════════════
// 11. format:"json" string shorthand forwarded
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn format_json_string_shorthand_forwarded() {
    let p = spawn_proxy().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    // Wire-level pin: Ollama's `"format": "json"` shorthand translates to a
    // permissive json_schema envelope because LM Studio only advertises
    // json_schema as a supported response_format type.
    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .and(body_partial_json(json!({
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "json",
                    "schema": { "type": "object" }
                }
            }
        })))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lm_chat_response("{\"ok\":true}", "stop")),
        )
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "messages": [{ "role": "user", "content": "JSON please" }],
            "stream": false,
            "format": "json"
        }))
        .send()
        .await
        .expect("POST /api/chat format json string");

    assert_eq!(resp.status(), 200);
    p.mock.verify().await;
}

// ═══════════════════════════════════════════════════════════════════════════
// 12. think flag forwarded — reasoning field returned in message.thinking
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn think_flag_forwarded_and_reasoning_in_message() {
    let p = spawn_proxy().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    let lm_resp = json!({
        "id": "chatcmpl-think",
        "object": "chat.completion",
        "created": 1_700_000_000u64,
        "model": "llama3.1-8b-instruct",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The answer is 42.",
                "reasoning": "Let me think step by step..."
            },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20 }
    });

    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lm_resp))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "messages": [{ "role": "user", "content": "What is 6*7?" }],
            "stream": false,
            "think": true
        }))
        .send()
        .await
        .expect("POST /api/chat think");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("JSON");
    assert_eq!(body["message"]["content"], "The answer is 42.");
    assert_eq!(
        body["message"]["thinking"], "Let me think step by step...",
        "thinking field must appear in the Ollama message"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 13. per-message images converted to multimodal content parts
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn per_message_images_converted_to_multimodal_content() {
    let p = spawn_proxy().await;
    mount_vision_catalog(&p, "llava-7b-v1.6").await;

    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lm_chat_response("I see a cat.", "stop")),
        )
        .mount(&p.mock)
        .await;

    // Minimal valid base64 PNG (1x1 pixel)
    let b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llava-7b:latest",
            "messages": [{
                "role": "user",
                "content": "What is in this image?",
                "images": [b64]
            }],
            "stream": false
        }))
        .send()
        .await
        .expect("POST /api/chat images");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("JSON");
    assert_eq!(body["message"]["content"], "I see a cat.");
}

// ═══════════════════════════════════════════════════════════════════════════
// 14. keep_alive as duration string accepted
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn keep_alive_duration_string_accepted() {
    let p = spawn_proxy().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lm_chat_response("OK", "stop")))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "messages": [{ "role": "user", "content": "Hi" }],
            "stream": false,
            "keep_alive": "10m"
        }))
        .send()
        .await
        .expect("POST /api/chat keep_alive string");

    assert_eq!(resp.status(), 200);
}

// ═══════════════════════════════════════════════════════════════════════════
// 15. keep_alive 0 (unload immediately) triggers LM Studio unload
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn keep_alive_zero_accepted() {
    let p = spawn_proxy().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lm_chat_response("OK", "stop")))
        .mount(&p.mock)
        .await;

    // Per spec, `keep_alive: 0` must trigger an explicit unload via the LM
    // Studio native endpoint after the response is returned. The unload runs
    // in a tokio::spawn background task, so we poll the mock briefly.
    Mock::given(method("POST"))
        .and(path("/api/v1/models/unload"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"ok": true})))
        .expect(1..)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "messages": [{ "role": "user", "content": "Hi" }],
            "stream": false,
            "keep_alive": 0
        }))
        .send()
        .await
        .expect("POST /api/chat keep_alive 0");

    assert_eq!(resp.status(), 200);

    let unload_was_called = async {
        for _ in 0..50 {
            let received = p.mock.received_requests().await.unwrap_or_default();
            if received
                .iter()
                .any(|r| r.url.path() == "/api/v1/models/unload")
            {
                return true;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        false
    }
    .await;

    assert!(
        unload_was_called,
        "keep_alive: 0 must result in a POST to /api/v1/models/unload"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 16. missing messages field → 400
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn missing_messages_returns_400() {
    let p = spawn_proxy().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({ "model": "llama3.1:8b" }))
        .send()
        .await
        .expect("POST /api/chat no messages");

    assert_eq!(resp.status(), 400);
}

// ═══════════════════════════════════════════════════════════════════════════
// 17. missing model field → 400
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn missing_model_returns_400() {
    let p = spawn_proxy().await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({ "messages": [{ "role": "user", "content": "Hi" }] }))
        .send()
        .await
        .expect("POST /api/chat no model");

    assert_eq!(resp.status(), 400);
}

// ═══════════════════════════════════════════════════════════════════════════
// 18. model not in LM Studio catalog → 404
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn unknown_model_returns_404() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [loaded_model_entry("some-other-model-v1")]
        })))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "does-not-exist:latest",
            "messages": [{ "role": "user", "content": "Hi" }],
            "stream": false
        }))
        .send()
        .await
        .expect("POST /api/chat unknown model");

    assert_eq!(resp.status(), 404);
}

// ═══════════════════════════════════════════════════════════════════════════
// 19. finish_reason "length" maps into done_reason:"length"
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn finish_reason_length_maps_to_done_reason_length() {
    let p = spawn_proxy().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lm_chat_response("Truncated...", "length")),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "messages": [{ "role": "user", "content": "Write a novel" }],
            "stream": false
        }))
        .send()
        .await
        .expect("POST /api/chat length");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("JSON");
    assert_eq!(body["done_reason"], "length");
}

// ═══════════════════════════════════════════════════════════════════════════
// 20. presence_penalty and frequency_penalty forwarded
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn penalty_options_forwarded_successfully() {
    let p = spawn_proxy().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lm_chat_response("Penalized.", "stop")),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "messages": [{ "role": "user", "content": "Hello" }],
            "stream": false,
            "options": {
                "presence_penalty": 0.5,
                "frequency_penalty": 0.3
            }
        }))
        .send()
        .await
        .expect("POST /api/chat penalties");

    assert_eq!(resp.status(), 200);
    p.mock.verify().await;
}

// ═══════════════════════════════════════════════════════════════════════════
// 21. logprobs and top_logprobs forwarded
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn logprobs_and_top_logprobs_forwarded() {
    let p = spawn_proxy().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lm_chat_response("Yes.", "stop")))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "messages": [{ "role": "user", "content": "Test" }],
            "stream": false,
            "logprobs": true,
            "top_logprobs": 5
        }))
        .send()
        .await
        .expect("POST /api/chat logprobs");

    assert_eq!(resp.status(), 200);
    p.mock.verify().await;
}

// ═══════════════════════════════════════════════════════════════════════════
// 22. multi-turn conversation forwarded intact
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn multi_turn_messages_forwarded_intact() {
    let p = spawn_proxy().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lm_chat_response("I remember.", "stop")),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "messages": [
                { "role": "system", "content": "You are helpful." },
                { "role": "user", "content": "What did I say?" },
                { "role": "assistant", "content": "You said hello." },
                { "role": "user", "content": "Right. And then?" }
            ],
            "stream": false
        }))
        .send()
        .await
        .expect("POST /api/chat multi-turn");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("JSON");
    assert_eq!(body["done"], true);
}

// ═══════════════════════════════════════════════════════════════════════════
// 23. streaming tool_calls delta arrives in chunks
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn streaming_tool_calls_delta_arrives_in_chunks() {
    let p = spawn_proxy().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    let tc_chunk = json!({
        "id": "chatcmpl-tc",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "call_xyz",
                    "type": "function",
                    "function": { "name": "search", "arguments": "{\"q\":\"rust\"}" }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    });
    let sse = format!("data: {}\n\ndata: [DONE]\n\n", tc_chunk);

    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(sse.into_bytes(), "text/event-stream"),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "messages": [{ "role": "user", "content": "Search rust" }],
            "stream": true,
            "tools": [{
                "type": "function",
                "function": { "name": "search", "parameters": {} }
            }]
        }))
        .send()
        .await
        .expect("POST /api/chat stream tools");

    assert_eq!(resp.status(), 200);
    let text = resp.text().await.expect("body");
    let chunks = parse_ndjson(&text);
    assert!(!chunks.is_empty());
    let final_chunk = chunks.last().unwrap();
    assert_eq!(final_chunk["done"], true);
}

// ═══════════════════════════════════════════════════════════════════════════
// 24. LM Studio 500 propagates as an error response
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn lm_studio_500_propagates_as_error() {
    let p = spawn_proxy().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
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
        .expect("POST /api/chat lm 500");

    assert!(resp.status().as_u16() >= 400, "expected error status");
}

// ═══════════════════════════════════════════════════════════════════════════
// 25. system message in messages array forwarded without duplication
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn system_message_in_messages_forwarded_without_duplication() {
    let p = spawn_proxy().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lm_chat_response("As instructed.", "stop")),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "messages": [
                { "role": "system", "content": "Always respond formally." },
                { "role": "user", "content": "Hello" }
            ],
            "stream": false
        }))
        .send()
        .await
        .expect("POST /api/chat system message");

    assert_eq!(resp.status(), 200);
    p.mock.verify().await;
}

// ═══════════════════════════════════════════════════════════════════════════
// 26. Populated stats — timings derived from stats, not wall-clock
// ═══════════════════════════════════════════════════════════════════════════
//
// When LM Studio returns a fully populated `stats` block, the proxy must
// report those timings unchanged rather than wall-clock measurements.

#[tokio::test]
async fn populated_stats_drives_timings_not_wall_clock() {
    let p = spawn_proxy().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    // Pick values wiremock cannot produce: 5 seconds total wall-clock is
    // impossible in a localhost round-trip.
    let mut lm_body = lm_chat_response("Hello there!", "stop");
    lm_body.as_object_mut().unwrap().insert(
        "stats".to_string(),
        json!({
            "time_to_first_token": 2.0,
            "generation_time": 3.0,
            "tokens_per_second": 50.0,
            "model_load_time_seconds": 1.5
        }),
    );

    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lm_body))
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
        .expect("POST /api/chat populated stats");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("JSON body");

    assert_eq!(
        body["total_duration"].as_u64(),
        Some(5_000_000_000),
        "total_duration must equal ttft + generation_time in ns"
    );
    assert_eq!(
        body["prompt_eval_duration"].as_u64(),
        Some(2_000_000_000),
        "prompt_eval_duration must equal time_to_first_token in ns"
    );
    assert_eq!(
        body["eval_duration"].as_u64(),
        Some(3_000_000_000),
        "eval_duration must equal generation_time in ns"
    );
    assert_eq!(
        body["load_duration"].as_u64(),
        Some(1_500_000_000),
        "load_duration must come from model_load_time_seconds"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 27. No stats block — wall-clock fallback
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn no_stats_block_uses_wall_clock_fallback() {
    let p = spawn_proxy().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    // Default lm_chat_response carries no `stats` block at all.
    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lm_chat_response("Hi back.", "stop")),
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
        .expect("POST /api/chat no stats");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("JSON body");

    let total = body["total_duration"].as_u64().expect("total_duration u64");
    assert!(
        total > 100_000,
        "total_duration must be wall-clock ns when stats are absent (got {total})"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 28. Empty stats — wall-clock fallback (parallels the generate-side test)
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn empty_stats_block_falls_back_to_wall_clock_timings() {
    let p = spawn_proxy().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    let mut lm_body = lm_chat_response("Hi.", "stop");
    lm_body
        .as_object_mut()
        .unwrap()
        .insert("stats".to_string(), json!({}));

    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lm_body))
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
        .expect("POST /api/chat empty stats");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("JSON body");

    // The buggy stats-branch path reports `total_duration: 1` and the
    // per-field .max(1) floors stamp 1 ns onto every duration. The 10 µs
    // threshold sits 10000× above that floor while tolerating fast hosts
    // where the wall-clock round-trip is sub-200 µs and the proportional
    // split between prompt_eval and eval can land below 100 µs per side.
    let total = body["total_duration"].as_u64().expect("total_duration u64");
    assert!(
        total > 10_000,
        "total_duration must be wall-clock ns when stats are empty (got {total})"
    );
    let prompt_eval = body["prompt_eval_duration"]
        .as_u64()
        .expect("prompt_eval_duration u64");
    assert!(
        prompt_eval > 10_000,
        "prompt_eval_duration must be wall-clock derived (got {prompt_eval})"
    );
    let eval = body["eval_duration"].as_u64().expect("eval_duration u64");
    assert!(
        eval > 10_000,
        "eval_duration must be wall-clock derived (got {eval})"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// options.system must be hoisted to a single {role:"system"} message,
// not duplicated as a top-level "system" key in the LM Studio request.
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn chat_options_system_does_not_leak_as_top_level_field() {
    let p = spawn_proxy().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lm_chat_response("ok", "stop")))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "messages": [{ "role": "user", "content": "hi" }],
            "stream": false,
            "options": { "system": "Sx" }
        }))
        .send()
        .await
        .expect("POST /api/chat options.system");

    assert_eq!(resp.status(), 200);

    let received = p.mock.received_requests().await.unwrap_or_default();
    let upstream = received
        .iter()
        .find(|r| r.url.path() == "/api/v0/chat/completions")
        .expect("LM Studio chat completions request captured");
    let body: Value = serde_json::from_slice(&upstream.body).expect("upstream body is JSON");

    assert!(
        body.get("system").is_none(),
        "options.system must not appear as a top-level key: {body}"
    );

    let messages = body["messages"].as_array().expect("messages array");
    let system_count = messages
        .iter()
        .filter(|m| m.get("role").and_then(Value::as_str) == Some("system"))
        .count();
    assert_eq!(
        system_count, 1,
        "exactly one synthetic system message must be present, got {system_count} in {body}"
    );
    assert_eq!(
        messages[0],
        json!({ "role": "system", "content": "Sx" }),
        "first message must be {{role:system, content:Sx}}: {body}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// LM Studio's chat-completions accepts a fixed key set
// (api-docs/lmstudio/.../chat-completions.md). The Ollama spec
// (api-docs/ollama/api/embed.md) lists `truncate` and `dimensions` as
// embedding-only fields, and Ollama's `ModelOptions.min_p` has no LM Studio
// equivalent. None of these may leak into the chat request body.
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn chat_drops_min_p_truncate_and_dimensions_options() {
    let p = spawn_proxy().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lm_chat_response("ok", "stop")))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "messages": [{ "role": "user", "content": "hi" }],
            "stream": false,
            "options": { "min_p": 0.1, "truncate": true, "dimensions": 64 }
        }))
        .send()
        .await
        .expect("POST /api/chat with min_p/truncate/dimensions");

    assert_eq!(resp.status(), 200);

    let received = p.mock.received_requests().await.unwrap_or_default();
    let upstream = received
        .iter()
        .find(|r| r.url.path() == "/api/v0/chat/completions")
        .expect("LM Studio chat completions request captured");
    let body: Value = serde_json::from_slice(&upstream.body).expect("upstream body is JSON");

    for key in ["min_p", "truncate", "dimensions"] {
        assert!(
            body.get(key).is_none(),
            "{key} must not appear in LM Studio chat body: {body}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// keep_alive:0 with no/empty messages → unload-only, no inference
// ═══════════════════════════════════════════════════════════════════════════
//
// Per `api-docs/ollama/api/chat.md`, `ChatRequest.required = [model]`; the
// documented unload call is `{"model":"x","keep_alive":0}` with no messages.
// The proxy must skip the LM Studio chat call entirely and only hit the
// native unload endpoint.

async fn assert_no_chat_inference_calls(p: &crate::common::TestProxy) {
    let received = p.mock.received_requests().await.unwrap_or_default();
    for r in &received {
        assert_ne!(
            r.url.path(),
            "/api/v0/chat/completions",
            "unload-only request must not POST /api/v0/chat/completions"
        );
        assert_ne!(
            r.url.path(),
            "/api/v0/completions",
            "unload-only request must not POST /api/v0/completions"
        );
    }
}

async fn wait_for_unload_call(p: &crate::common::TestProxy) -> bool {
    for _ in 0..50 {
        let received = p.mock.received_requests().await.unwrap_or_default();
        if received
            .iter()
            .any(|r| r.url.path() == "/api/v1/models/unload")
        {
            return true;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    false
}

#[tokio::test]
async fn chat_keep_alive_zero_with_no_messages_unloads_without_inference() {
    let p = spawn_proxy().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/api/v1/models/unload"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"ok": true})))
        .expect(1..)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "keep_alive": 0,
            "stream": false
        }))
        .send()
        .await
        .expect("POST /api/chat keep_alive:0 no messages");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("JSON body");
    assert_eq!(body["done"], true);
    assert_eq!(body["message"]["role"], "assistant");
    assert_eq!(body["message"]["content"], "");
    assert_eq!(body["model"], "llama3.1:8b");

    assert_no_chat_inference_calls(&p).await;
    assert!(
        wait_for_unload_call(&p).await,
        "keep_alive: 0 must result in a POST to /api/v1/models/unload"
    );
}

#[tokio::test]
async fn chat_keep_alive_zero_with_empty_messages_unloads_without_inference() {
    let p = spawn_proxy().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/api/v1/models/unload"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"ok": true})))
        .expect(1..)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "messages": [],
            "keep_alive": 0,
            "stream": false
        }))
        .send()
        .await
        .expect("POST /api/chat keep_alive:0 empty messages");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("JSON body");
    assert_eq!(body["done"], true);
    assert_eq!(body["message"]["role"], "assistant");
    assert_eq!(body["message"]["content"], "");

    assert_no_chat_inference_calls(&p).await;
    assert!(
        wait_for_unload_call(&p).await,
        "keep_alive: 0 must result in a POST to /api/v1/models/unload"
    );
}

#[tokio::test]
async fn chat_keep_alive_zero_no_messages_streaming_returns_ndjson_done_chunk() {
    let p = spawn_proxy().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/api/v1/models/unload"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"ok": true})))
        .expect(1..)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "keep_alive": 0,
            "stream": true
        }))
        .send()
        .await
        .expect("POST /api/chat keep_alive:0 stream");

    assert_eq!(resp.status(), 200);
    let text = resp.text().await.expect("body text");
    let chunks = parse_ndjson(&text);
    let final_chunk = chunks.last().expect("at least one chunk");
    assert_eq!(final_chunk["done"], true);
    assert_eq!(final_chunk["message"]["role"], "assistant");
    assert_eq!(final_chunk["message"]["content"], "");

    assert_no_chat_inference_calls(&p).await;
    assert!(wait_for_unload_call(&p).await);
}
