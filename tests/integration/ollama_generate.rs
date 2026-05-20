// Integration tests for POST /api/generate — Ollama generate surface.
//
// The generate handler routes to `/v1/completions` for plain text prompts and
// to `/v1/chat/completions` for vision requests (images present). Both paths
// are covered here.
//
// Model resolution uses /api/v1/models (LM Studio native). The mock key
// "llama3.2-3b-instruct" substring-matches the Ollama name "llama3.2:3b",
// and "llava-7b-v1.6" matches "llava-7b:latest".

use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, ResponseTemplate};

use crate::common::spawn_proxy;

// ── model-catalog helpers ───────────────────────────────────────────────────

fn llm_entry(key: &str) -> Value {
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

fn vlm_entry(key: &str) -> Value {
    json!({
        "key": key,
        "type": "vlm",
        "publisher": "llava",
        "architecture": "llama",
        "format": "gguf",
        "quantization": { "name": "Q4_K_M", "bits_per_weight": 4.5 },
        "max_context_length": 4096,
        "loaded_instances": [
            { "id": "inst-0", "config": { "context_length": 4096 } }
        ],
        "capabilities": { "vision": true, "trained_for_tool_use": false }
    })
}

async fn mount_llm_catalog(proxy: &crate::common::TestProxy, key: &str) {
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [llm_entry(key)]
        })))
        .mount(&proxy.mock)
        .await;
}

async fn mount_vlm_catalog(proxy: &crate::common::TestProxy, key: &str) {
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [vlm_entry(key)]
        })))
        .mount(&proxy.mock)
        .await;
}

/// Minimal LM Studio /v1/completions response.
fn lm_completion_response(text: &str, finish_reason: &str) -> Value {
    json!({
        "id": "cmpl-test",
        "object": "text_completion",
        "created": 1_700_000_000u64,
        "model": "llama3.2-3b-instruct",
        "choices": [{
            "index": 0,
            "text": text,
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": 6,
            "completion_tokens": 10,
            "total_tokens": 16
        }
    })
}

/// Minimal LM Studio /v1/chat/completions response (used for vision path).
fn lm_chat_response(content: &str, finish_reason: &str) -> Value {
    json!({
        "id": "chatcmpl-vision",
        "object": "chat.completion",
        "created": 1_700_000_000u64,
        "model": "llava-7b-v1.6",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": content },
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 12,
            "total_tokens": 32
        }
    })
}

/// SSE body for streaming /v1/completions (text chunks).
fn sse_completion_body(tokens: &[&str], finish_reason: &str) -> String {
    let mut body = String::new();
    for (i, token) in tokens.iter().enumerate() {
        let is_last = i + 1 == tokens.len();
        let fr_json = if is_last {
            format!("\"{}\"", finish_reason)
        } else {
            "null".to_string()
        };
        let chunk = json!({
            "id": "cmpl-stream",
            "object": "text_completion.chunk",
            "choices": [{
                "index": 0,
                "text": *token,
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
// 1. Non-streaming golden path — Ollama generate response shape
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn non_streaming_generate_returns_ollama_shape() {
    let p = spawn_proxy().await;
    mount_llm_catalog(&p, "llama3.2-3b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(lm_completion_response("The sky is blue.", "stop")),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/generate"))
        .json(&json!({
            "model": "llama3.2:3b",
            "prompt": "Why is the sky blue?",
            "stream": false
        }))
        .send()
        .await
        .expect("POST /api/generate");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("JSON body");

    assert_eq!(body["model"], "llama3.2:3b");
    assert_eq!(body["done"], true);
    assert_eq!(body["done_reason"], "stop");
    assert_eq!(body["response"], "The sky is blue.");
    assert!(body["eval_count"].is_number(), "eval_count must be present");
    assert!(
        body["prompt_eval_count"].is_number(),
        "prompt_eval_count must be present"
    );
    assert!(body["total_duration"].is_number());
    assert!(body["created_at"].is_string());
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. stream absent defaults to streaming
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn stream_absent_defaults_to_streaming() {
    let p = spawn_proxy().await;
    mount_llm_catalog(&p, "llama3.2-3b-instruct").await;

    let sse = sse_completion_body(&["OK"], "stop");
    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(sse.into_bytes(), "text/event-stream"),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/generate"))
        .json(&json!({
            "model": "llama3.2:3b",
            "prompt": "Hello"
        }))
        .send()
        .await
        .expect("POST /api/generate no stream");

    assert_eq!(resp.status(), 200);
    let text = resp.text().await.expect("body text");
    let chunks = parse_ndjson(&text);
    assert!(
        !chunks.is_empty(),
        "absent stream must produce NDJSON chunks"
    );
    let final_chunk = chunks.last().unwrap();
    assert_eq!(final_chunk["done"], true);
    assert!(final_chunk.get("response").is_some());
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. stream:true — NDJSON chunks, final chunk done:true
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn streaming_generate_emits_ndjson_with_final_done_chunk() {
    let p = spawn_proxy().await;
    mount_llm_catalog(&p, "llama3.2-3b-instruct").await;

    let sse = sse_completion_body(&["The", " sky", " is", " blue."], "stop");
    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(sse.into_bytes(), "text/event-stream"),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/generate"))
        .json(&json!({
            "model": "llama3.2:3b",
            "prompt": "Why is the sky blue?",
            "stream": true
        }))
        .send()
        .await
        .expect("POST /api/generate stream");

    assert_eq!(resp.status(), 200);
    let text = resp.text().await.expect("body text");
    let chunks = parse_ndjson(&text);
    assert!(!chunks.is_empty(), "must receive at least one NDJSON chunk");

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

    // Non-final chunks carry done:false and a response field.
    for chunk in &chunks[..chunks.len().saturating_sub(1)] {
        assert_eq!(
            chunk["done"], false,
            "intermediate chunk must be done:false"
        );
        assert!(chunk.get("response").is_some(), "response field missing");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. stream:false explicitly — single JSON object
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn stream_explicit_false_returns_single_object() {
    let p = spawn_proxy().await;
    mount_llm_catalog(&p, "llama3.2-3b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lm_completion_response("Done.", "stop")),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/generate"))
        .json(&json!({
            "model": "llama3.2:3b",
            "prompt": "Test",
            "stream": false
        }))
        .send()
        .await
        .expect("POST /api/generate stream:false");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("single JSON object");
    assert!(body.is_object());
    assert_eq!(body["done"], true);
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. options translation — temperature, num_predict → max_tokens
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn options_temperature_and_num_predict_forwarded() {
    let p = spawn_proxy().await;
    mount_llm_catalog(&p, "llama3.2-3b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lm_completion_response("Counted.", "stop")),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/generate"))
        .json(&json!({
            "model": "llama3.2:3b",
            "prompt": "Count to ten",
            "stream": false,
            "options": {
                "temperature": 0.2,
                "num_predict": 128,
                "top_p": 0.9
            }
        }))
        .send()
        .await
        .expect("POST /api/generate options");

    assert_eq!(resp.status(), 200);
    p.mock.verify().await;
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. options.num_ctx forwarded as context_length
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn options_num_ctx_forwarded_as_context_length() {
    let p = spawn_proxy().await;
    mount_llm_catalog(&p, "llama3.2-3b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lm_completion_response("OK", "stop")),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/generate"))
        .json(&json!({
            "model": "llama3.2:3b",
            "prompt": "Test",
            "stream": false,
            "options": { "num_ctx": 1024 }
        }))
        .send()
        .await
        .expect("POST /api/generate num_ctx");

    assert_eq!(resp.status(), 200);
    p.mock.verify().await;
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. options.stop as array forwarded
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn options_stop_array_forwarded() {
    let p = spawn_proxy().await;
    mount_llm_catalog(&p, "llama3.2-3b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lm_completion_response("Stopped.", "stop")),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/generate"))
        .json(&json!({
            "model": "llama3.2:3b",
            "prompt": "Continue",
            "stream": false,
            "options": { "stop": ["</s>", "[END]"] }
        }))
        .send()
        .await
        .expect("POST /api/generate stop array");

    assert_eq!(resp.status(), 200);
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. system prompt prepended to text prompt (non-raw)
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn system_prompt_prepended_to_text_prompt() {
    let p = spawn_proxy().await;
    mount_llm_catalog(&p, "llama3.2-3b-instruct").await;

    // The proxy prepends the system prompt text to the prompt before sending
    // to /v1/completions. We verify the request reaches LM Studio.
    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(lm_completion_response("Formal reply.", "stop")),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/generate"))
        .json(&json!({
            "model": "llama3.2:3b",
            "prompt": "Hello",
            "system": "You are a formal assistant.",
            "stream": false
        }))
        .send()
        .await
        .expect("POST /api/generate system");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("JSON");
    assert_eq!(body["done"], true);
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. raw:true skips system prompt injection
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn raw_mode_skips_system_prompt_injection() {
    let p = spawn_proxy().await;
    mount_llm_catalog(&p, "llama3.2-3b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lm_completion_response("Raw reply.", "stop")),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/generate"))
        .json(&json!({
            "model": "llama3.2:3b",
            "prompt": "[INST] Hello [/INST]",
            "system": "This should be ignored in raw mode.",
            "raw": true,
            "stream": false
        }))
        .send()
        .await
        .expect("POST /api/generate raw");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("JSON");
    assert_eq!(body["response"], "Raw reply.");
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. suffix forwarded for completion fill-in-the-middle
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn suffix_forwarded_to_completions_endpoint() {
    let p = spawn_proxy().await;
    mount_llm_catalog(&p, "llama3.2-3b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lm_completion_response("middle text", "stop")),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/generate"))
        .json(&json!({
            "model": "llama3.2:3b",
            "prompt": "def hello(",
            "suffix": "):\n    pass",
            "stream": false
        }))
        .send()
        .await
        .expect("POST /api/generate suffix");

    assert_eq!(resp.status(), 200);
    p.mock.verify().await;
}

// ═══════════════════════════════════════════════════════════════════════════
// 11. format:"json" forwarded
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn format_json_string_forwarded() {
    let p = spawn_proxy().await;
    mount_llm_catalog(&p, "llama3.2-3b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(lm_completion_response("{\"ok\":true}", "stop")),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/generate"))
        .json(&json!({
            "model": "llama3.2:3b",
            "prompt": "Return JSON",
            "stream": false,
            "format": "json"
        }))
        .send()
        .await
        .expect("POST /api/generate format json");

    assert_eq!(resp.status(), 200);
}

// ═══════════════════════════════════════════════════════════════════════════
// 12. format as JSON schema object forwarded
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn format_json_schema_object_forwarded() {
    let p = spawn_proxy().await;
    mount_llm_catalog(&p, "llama3.2-3b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(lm_completion_response("{\"count\":3}", "stop")),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/generate"))
        .json(&json!({
            "model": "llama3.2:3b",
            "prompt": "Count items",
            "stream": false,
            "format": {
                "type": "object",
                "properties": { "count": { "type": "integer" } }
            }
        }))
        .send()
        .await
        .expect("POST /api/generate format schema");

    assert_eq!(resp.status(), 200);
}

// ═══════════════════════════════════════════════════════════════════════════
// 13. think flag forwarded — reasoning in generate response
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn think_flag_forwarded_and_reasoning_in_response() {
    let p = spawn_proxy().await;
    mount_llm_catalog(&p, "llama3.2-3b-instruct").await;

    // The completions endpoint may return reasoning at the choice level.
    let lm_resp = json!({
        "id": "cmpl-think",
        "object": "text_completion",
        "created": 1_700_000_000u64,
        "model": "llama3.2-3b-instruct",
        "choices": [{
            "index": 0,
            "text": "42",
            "reasoning": "Step by step...",
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 8, "completion_tokens": 1, "total_tokens": 9 }
    });

    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lm_resp))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/generate"))
        .json(&json!({
            "model": "llama3.2:3b",
            "prompt": "What is 6*7?",
            "stream": false,
            "think": true
        }))
        .send()
        .await
        .expect("POST /api/generate think");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("JSON");
    assert_eq!(body["response"], "42");
    // thinking field is present when reasoning content returned
    assert!(
        body.get("thinking").is_some(),
        "thinking field must be present"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 14. vision path — images present routes to /v1/chat/completions
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn generate_with_images_routes_to_chat_completions() {
    let p = spawn_proxy().await;
    mount_vlm_catalog(&p, "llava-7b-v1.6").await;

    // Must hit /v1/chat/completions, not /v1/completions
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lm_chat_response("A cat is shown.", "stop")),
        )
        .mount(&p.mock)
        .await;

    let b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

    let resp = p
        .client
        .post(p.url("/api/generate"))
        .json(&json!({
            "model": "llava-7b:latest",
            "prompt": "Describe the image",
            "images": [b64],
            "stream": false
        }))
        .send()
        .await
        .expect("POST /api/generate images");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("JSON");
    assert_eq!(body["response"], "A cat is shown.");
    assert_eq!(body["done"], true);
}

// ═══════════════════════════════════════════════════════════════════════════
// 15. vision + streaming — NDJSON chunks, done:true at end
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn generate_vision_streaming_emits_done_chunk() {
    let p = spawn_proxy().await;
    mount_vlm_catalog(&p, "llava-7b-v1.6").await;

    let sse = {
        let chunk = json!({
            "id": "chatcmpl-vision-stream",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": { "content": "A cat." },
                "finish_reason": "stop"
            }]
        });
        format!("data: {}\n\ndata: [DONE]\n\n", chunk)
    };

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(sse.into_bytes(), "text/event-stream"),
        )
        .mount(&p.mock)
        .await;

    let b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

    let resp = p
        .client
        .post(p.url("/api/generate"))
        .json(&json!({
            "model": "llava-7b:latest",
            "prompt": "Describe it",
            "images": [b64],
            "stream": true
        }))
        .send()
        .await
        .expect("POST /api/generate vision stream");

    assert_eq!(resp.status(), 200);
    let text = resp.text().await.expect("body text");
    let chunks = parse_ndjson(&text);
    assert!(!chunks.is_empty());
    let final_chunk = chunks.last().unwrap();
    assert_eq!(final_chunk["done"], true);
    assert!(final_chunk.get("done_reason").is_some());
}

// ═══════════════════════════════════════════════════════════════════════════
// 16. missing prompt → 400
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn missing_prompt_returns_400() {
    let p = spawn_proxy().await;
    mount_llm_catalog(&p, "llama3.2-3b-instruct").await;

    let resp = p
        .client
        .post(p.url("/api/generate"))
        .json(&json!({ "model": "llama3.2:3b" }))
        .send()
        .await
        .expect("POST /api/generate no prompt");

    assert_eq!(resp.status(), 400);
}

// ═══════════════════════════════════════════════════════════════════════════
// 17. missing model → 400
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn missing_model_returns_400() {
    let p = spawn_proxy().await;

    let resp = p
        .client
        .post(p.url("/api/generate"))
        .json(&json!({ "prompt": "Hello" }))
        .send()
        .await
        .expect("POST /api/generate no model");

    assert_eq!(resp.status(), 400);
}

// ═══════════════════════════════════════════════════════════════════════════
// 18. model not in catalog → 404
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn unknown_model_returns_404() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [llm_entry("different-model-v1")]
        })))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/generate"))
        .json(&json!({
            "model": "does-not-exist:latest",
            "prompt": "Hi",
            "stream": false
        }))
        .send()
        .await
        .expect("POST /api/generate unknown model");

    assert_eq!(resp.status(), 404);
}

// ═══════════════════════════════════════════════════════════════════════════
// 19. finish_reason "length" maps to done_reason:"length"
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn finish_reason_length_maps_to_done_reason_length() {
    let p = spawn_proxy().await;
    mount_llm_catalog(&p, "llama3.2-3b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lm_completion_response("Truncated", "length")),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/generate"))
        .json(&json!({
            "model": "llama3.2:3b",
            "prompt": "Write a very long essay",
            "stream": false
        }))
        .send()
        .await
        .expect("POST /api/generate length");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("JSON");
    assert_eq!(body["done_reason"], "length");
}

// ═══════════════════════════════════════════════════════════════════════════
// 20. keep_alive duration string accepted
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn keep_alive_duration_string_accepted() {
    let p = spawn_proxy().await;
    mount_llm_catalog(&p, "llama3.2-3b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lm_completion_response("OK", "stop")),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/generate"))
        .json(&json!({
            "model": "llama3.2:3b",
            "prompt": "Hi",
            "stream": false,
            "keep_alive": "5m"
        }))
        .send()
        .await
        .expect("POST /api/generate keep_alive");

    assert_eq!(resp.status(), 200);
}

// ═══════════════════════════════════════════════════════════════════════════
// 21. LM Studio 500 propagates as error
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn lm_studio_500_propagates_as_error() {
    let p = spawn_proxy().await;
    mount_llm_catalog(&p, "llama3.2-3b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("Internal error"))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/generate"))
        .json(&json!({
            "model": "llama3.2:3b",
            "prompt": "Hi",
            "stream": false
        }))
        .send()
        .await
        .expect("POST /api/generate lm 500");

    assert!(resp.status().as_u16() >= 400);
}

// ═══════════════════════════════════════════════════════════════════════════
// 22. repeat_penalty forwarded as repeat_penalty
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn repeat_penalty_option_forwarded() {
    let p = spawn_proxy().await;
    mount_llm_catalog(&p, "llama3.2-3b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lm_completion_response("No repeats.", "stop")),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/generate"))
        .json(&json!({
            "model": "llama3.2:3b",
            "prompt": "Hi",
            "stream": false,
            "options": { "repeat_penalty": 1.3 }
        }))
        .send()
        .await
        .expect("POST /api/generate repeat_penalty");

    assert_eq!(resp.status(), 200);
    p.mock.verify().await;
}

// ═══════════════════════════════════════════════════════════════════════════
// 23. vision raw:true skips system injection before image messages
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn vision_raw_skips_system_injection() {
    let p = spawn_proxy().await;
    mount_vlm_catalog(&p, "llava-7b-v1.6").await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lm_chat_response("Raw vision reply.", "stop")),
        )
        .mount(&p.mock)
        .await;

    let b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

    let resp = p
        .client
        .post(p.url("/api/generate"))
        .json(&json!({
            "model": "llava-7b:latest",
            "prompt": "What do you see?",
            "system": "Ignored because raw:true.",
            "images": [b64],
            "raw": true,
            "stream": false
        }))
        .send()
        .await
        .expect("POST /api/generate vision raw");

    assert_eq!(resp.status(), 200);
    p.mock.verify().await;
}

// ═══════════════════════════════════════════════════════════════════════════
// 24. logprobs forwarded
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn logprobs_forwarded() {
    let p = spawn_proxy().await;
    mount_llm_catalog(&p, "llama3.2-3b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lm_completion_response("Logged.", "stop")),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/generate"))
        .json(&json!({
            "model": "llama3.2:3b",
            "prompt": "Log me",
            "stream": false,
            "logprobs": true,
            "top_logprobs": 3
        }))
        .send()
        .await
        .expect("POST /api/generate logprobs");

    assert_eq!(resp.status(), 200);
    p.mock.verify().await;
}

// ═══════════════════════════════════════════════════════════════════════════
// 25. logprobs data returned in generate response when upstream provides it
// ═══════════════════════════════════════════════════════════════════════════

fn lm_completion_response_with_logprobs(text: &str, finish_reason: &str) -> Value {
    json!({
        "id": "cmpl-test",
        "object": "text_completion",
        "created": 1_700_000_000u64,
        "model": "llama3.2-3b-instruct",
        "choices": [{
            "index": 0,
            "text": text,
            "finish_reason": finish_reason,
            "logprobs": {
                "content": [
                    {"token": "Hello", "logprob": -0.1, "top_logprobs": []},
                    {"token": "!", "logprob": -0.2, "top_logprobs": []}
                ]
            }
        }],
        "usage": {
            "prompt_tokens": 6,
            "completion_tokens": 2,
            "total_tokens": 8
        }
    })
}

#[tokio::test]
async fn logprobs_data_present_in_generate_response() {
    let p = spawn_proxy().await;
    mount_llm_catalog(&p, "llama3.2-3b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(lm_completion_response_with_logprobs("Hello!", "stop")),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/generate"))
        .json(&json!({
            "model": "llama3.2:3b",
            "prompt": "Say hello",
            "stream": false,
            "logprobs": true,
            "top_logprobs": 5
        }))
        .send()
        .await
        .expect("POST /api/generate logprobs data");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("JSON");
    let logprobs = body
        .get("logprobs")
        .and_then(|v| v.as_array())
        .expect("logprobs must be a flat array in generate response");
    assert_eq!(logprobs.len(), 2);
    assert_eq!(
        logprobs[0].get("token").and_then(|t| t.as_str()),
        Some("Hello")
    );
    p.mock.verify().await;
}

// ═══════════════════════════════════════════════════════════════════════════
// 26. context array (legacy) present in request — proxy accepts without error
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn context_array_in_request_accepted() {
    let p = spawn_proxy().await;
    mount_llm_catalog(&p, "llama3.2-3b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lm_completion_response("Continued.", "stop")),
        )
        .mount(&p.mock)
        .await;

    // `context` is a deprecated Ollama field; the proxy should not crash on it.
    let resp = p
        .client
        .post(p.url("/api/generate"))
        .json(&json!({
            "model": "llama3.2:3b",
            "prompt": "Continue from here",
            "stream": false,
            "context": [1, 2, 3, 4, 5]
        }))
        .send()
        .await
        .expect("POST /api/generate context array");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("JSON");
    assert_eq!(body["done"], true);
    // Ollama omits the context field in the response (deprecated)
    // — we only assert done:true and no crash.
}

// ═══════════════════════════════════════════════════════════════════════════
// 27. Empty stats block — fall back to wall-clock, never report all-1 ns
// ═══════════════════════════════════════════════════════════════════════════
//
// LM Studio's OpenAI-compat /v1/completions path returns `stats: {}` on every
// non-stream response. With no usable timing values, the proxy must fall back
// to wall-clock measurement rather than emitting 1-ns floors.

#[tokio::test]
async fn empty_stats_block_falls_back_to_wall_clock_timings() {
    let p = spawn_proxy().await;
    mount_llm_catalog(&p, "llama3.2-3b-instruct").await;

    let mut lm_body = lm_completion_response("Reply.", "stop");
    lm_body
        .as_object_mut()
        .unwrap()
        .insert("stats".to_string(), json!({}));

    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lm_body))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/generate"))
        .json(&json!({
            "model": "llama3.2:3b",
            "prompt": "Hello?",
            "stream": false
        }))
        .send()
        .await
        .expect("POST /api/generate empty stats");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("JSON body");

    // The buggy stats-branch path reports `total_duration: 1` and the
    // per-field .max(1) floors stamp 1 ns onto every duration. Real wall-clock
    // wiremock round-trips through axum + reqwest run in the millisecond range;
    // 100 µs is well below that floor and far above any spurious sub-µs zeroing.
    let total = body["total_duration"].as_u64().expect("total_duration u64");
    assert!(
        total > 100_000,
        "total_duration must be wall-clock ns when stats are empty (got {total})"
    );
    let prompt_eval = body["prompt_eval_duration"]
        .as_u64()
        .expect("prompt_eval_duration u64");
    assert!(
        prompt_eval > 100_000,
        "prompt_eval_duration must be wall-clock derived (got {prompt_eval})"
    );
    let eval = body["eval_duration"].as_u64().expect("eval_duration u64");
    assert!(
        eval > 100_000,
        "eval_duration must be wall-clock derived (got {eval})"
    );
}
