// Integration tests for /v1/* OpenAI-compat passthrough.
//
// The proxy forwards every /v1/* request verbatim to the LM Studio backend and
// returns the response unchanged. These tests assert that the forwarding is
// truly transparent: method, path, body, and response headers all round-trip
// without modification.

use serde_json::json;
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, ResponseTemplate};

use crate::common::spawn_proxy;

/// Mount a GET /api/v1/models stub returning a single model whose key contains `model_key`.
async fn mount_native_models(p: &crate::common::TestProxy, model_key: &str) {
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [{"key": model_key, "type": "llm", "publisher": "meta",
                        "architecture": "llama", "format": "gguf",
                        "quantization": {"name": "Q4_K_M", "bits_per_weight": 4.5},
                        "max_context_length": 8192, "loaded_instances": [],
                        "capabilities": {"vision": false, "trained_for_tool_use": false}}]
        })))
        .mount(&p.mock)
        .await;
}

// ── GET /v1/models ────────────────────────────────────────────────────────────

#[tokio::test]
async fn openai_models_list_forwarded() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-lmstudio-backend", "yes")
                .set_body_json(json!({
                    "object": "list",
                    "data": [
                        { "id": "lmstudio-community/meta-llama-3.1-8b", "object": "model" }
                    ]
                })),
        )
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .get(p.url("/v1/models"))
        .send()
        .await
        .expect("GET /v1/models");

    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers().get("x-lmstudio-backend").unwrap(),
        "yes",
        "custom response header must pass through"
    );
    let body: serde_json::Value = resp.json().await.expect("json body");
    assert_eq!(body["object"], "list");
}

#[tokio::test]
async fn openai_models_list_backend_error_propagated() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(503).set_body_string("service unavailable"))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .get(p.url("/v1/models"))
        .send()
        .await
        .expect("GET /v1/models 503");

    assert_eq!(resp.status(), 503);
}

// ── POST /v1/chat/completions (non-streaming) ─────────────────────────────────

#[tokio::test]
async fn openai_chat_completions_non_streaming_forwarded() {
    let p = spawn_proxy().await;
    mount_native_models(&p, "lmstudio-community/meta-llama-3.1-8b").await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({
            "model": "lmstudio-community/meta-llama-3.1-8b",
            "messages": [{ "role": "user", "content": "Hello" }]
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-lmstudio-request-id", "req-abc")
                .set_body_json(json!({
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "choices": [{ "message": { "role": "assistant", "content": "Hi!" } }]
                })),
        )
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/v1/chat/completions"))
        .json(&json!({
            "model": "lmstudio-community/meta-llama-3.1-8b",
            "messages": [{ "role": "user", "content": "Hello" }],
            "stream": false
        }))
        .send()
        .await
        .expect("POST /v1/chat/completions");

    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers().get("x-lmstudio-request-id").unwrap(),
        "req-abc"
    );
    let body: serde_json::Value = resp.json().await.expect("json body");
    assert_eq!(body["object"], "chat.completion");
}

#[tokio::test]
async fn openai_chat_completions_backend_error_propagated() {
    let p = spawn_proxy().await;
    mount_native_models(&p, "missing").await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "error": { "message": "model not found", "type": "invalid_request_error" }
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/v1/chat/completions"))
        .json(&json!({ "model": "missing", "messages": [] }))
        .send()
        .await
        .expect("POST /v1/chat/completions 400");

    assert_eq!(resp.status(), 400);
    let body: serde_json::Value = resp.json().await.expect("json body");
    assert_eq!(body["error"]["type"], "invalid_request_error");
}

// ── POST /v1/chat/completions (streaming SSE) ─────────────────────────────────

#[tokio::test]
async fn openai_chat_completions_streaming_bytes_roundtrip() {
    let p = spawn_proxy().await;
    mount_native_models(&p, "lmstudio-community/meta-llama-3.1-8b").await;

    let sse_payload = concat!(
        "data: {\"id\":\"chatcmpl-2\",\"object\":\"chat.completion.chunk\",",
        "\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n",
        "data: [DONE]\n\n"
    );

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({ "stream": true })))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(sse_payload.as_bytes(), "text/event-stream"),
        )
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/v1/chat/completions"))
        .json(&json!({
            "model": "lmstudio-community/meta-llama-3.1-8b",
            "messages": [{ "role": "user", "content": "Hello" }],
            "stream": true
        }))
        .send()
        .await
        .expect("POST /v1/chat/completions streaming");

    assert_eq!(resp.status(), 200);
    let bytes = resp.bytes().await.expect("bytes");
    assert_eq!(bytes.as_ref(), sse_payload.as_bytes());
}

// ── POST /v1/completions (non-streaming) ──────────────────────────────────────

#[tokio::test]
async fn openai_completions_non_streaming_forwarded() {
    let p = spawn_proxy().await;
    mount_native_models(&p, "lmstudio-community/meta-llama-3.1-8b").await;

    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .and(body_partial_json(json!({
            "model": "lmstudio-community/meta-llama-3.1-8b",
            "prompt": "Once upon a time"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "cmpl-1",
            "object": "text_completion",
            "choices": [{ "text": " there was a fox." }]
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/v1/completions"))
        .json(&json!({
            "model": "lmstudio-community/meta-llama-3.1-8b",
            "prompt": "Once upon a time"
        }))
        .send()
        .await
        .expect("POST /v1/completions");

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.expect("json body");
    assert_eq!(body["object"], "text_completion");
}

// ── POST /v1/completions (streaming SSE) ──────────────────────────────────────

#[tokio::test]
async fn openai_completions_streaming_bytes_roundtrip() {
    let p = spawn_proxy().await;
    mount_native_models(&p, "lmstudio-community/meta-llama-3.1-8b").await;

    let sse_payload = concat!(
        "data: {\"id\":\"cmpl-2\",\"object\":\"text_completion.chunk\",",
        "\"choices\":[{\"text\":\"fox\"}]}\n\n",
        "data: [DONE]\n\n"
    );

    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .and(body_partial_json(json!({ "stream": true })))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(sse_payload.as_bytes(), "text/event-stream"),
        )
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/v1/completions"))
        .json(&json!({
            "model": "lmstudio-community/meta-llama-3.1-8b",
            "prompt": "Once upon",
            "stream": true
        }))
        .send()
        .await
        .expect("POST /v1/completions streaming");

    assert_eq!(resp.status(), 200);
    let bytes = resp.bytes().await.expect("bytes");
    assert_eq!(bytes.as_ref(), sse_payload.as_bytes());
}

// ── POST /v1/embeddings ───────────────────────────────────────────────────────

#[tokio::test]
async fn openai_embeddings_forwarded() {
    let p = spawn_proxy().await;
    mount_native_models(&p, "text-embedding-nomic-embed-text-v1.5").await;

    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .and(body_partial_json(json!({
            "model": "text-embedding-nomic-embed-text-v1.5",
            "input": ["hello world"]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [{ "object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3] }],
            "model": "text-embedding-nomic-embed-text-v1.5"
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/v1/embeddings"))
        .json(&json!({
            "model": "text-embedding-nomic-embed-text-v1.5",
            "input": ["hello world"]
        }))
        .send()
        .await
        .expect("POST /v1/embeddings");

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.expect("json body");
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"][0]["object"], "embedding");
}

#[tokio::test]
async fn openai_embeddings_backend_error_propagated() {
    let p = spawn_proxy().await;
    mount_native_models(&p, "nomic").await;

    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(422).set_body_json(json!({
            "error": { "message": "invalid input", "type": "invalid_request_error" }
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/v1/embeddings"))
        .json(&json!({ "model": "nomic", "input": [] }))
        .send()
        .await
        .expect("POST /v1/embeddings 422");

    assert_eq!(resp.status(), 422);
}

// ── POST /v1/messages (Anthropic-compat passthrough) ─────────────────────────

#[tokio::test]
async fn anthropic_messages_model_remapped_and_forwarded() {
    let p = spawn_proxy().await;
    // LM Studio key is the full publisher/name form; client sends a short Ollama-style name.
    mount_native_models(&p, "ibm/granite-4-micro").await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_partial_json(json!({
            "model": "ibm/granite-4-micro",
            "messages": [{ "role": "user", "content": "Hello" }],
            "output_config": { "effort": "high" }
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-request-id", "msg-001")
                .set_body_json(json!({
                    "id": "msg_001",
                    "type": "message",
                    "role": "assistant",
                    "content": [{ "type": "text", "text": "Hi there!" }],
                    "model": "ibm/granite-4-micro",
                    "stop_reason": "end_turn",
                    "usage": { "input_tokens": 5, "output_tokens": 3 }
                })),
        )
        .expect(1)
        .mount(&p.mock)
        .await;

    // Client uses a short Ollama-style name; proxy must resolve and remap it.
    let resp = p
        .client
        .post(p.url("/v1/messages"))
        .json(&json!({
            "model": "granite-4-micro",
            "max_tokens": 256,
            "messages": [{ "role": "user", "content": "Hello" }],
            "output_config": { "effort": "high" }
        }))
        .send()
        .await
        .expect("POST /v1/messages");

    assert_eq!(resp.status(), 200);
    assert_eq!(resp.headers().get("x-request-id").unwrap(), "msg-001");
    let body: serde_json::Value = resp.json().await.expect("json body");
    assert_eq!(body["type"], "message");
    assert_eq!(body["role"], "assistant");
}

#[tokio::test]
async fn anthropic_messages_backend_error_propagated() {
    let p = spawn_proxy().await;
    mount_native_models(&p, "ibm/granite-4-micro").await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "type": "error",
            "error": { "type": "invalid_request_error", "message": "missing max_tokens" }
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/v1/messages"))
        .json(&json!({ "model": "granite-4-micro", "messages": [] }))
        .send()
        .await
        .expect("POST /v1/messages 400");

    assert_eq!(resp.status(), 400);
    let body: serde_json::Value = resp.json().await.expect("json body");
    assert_eq!(body["error"]["type"], "invalid_request_error");
}

// ── POST /v1/responses (LM Studio extension) ──────────────────────────────────

#[tokio::test]
async fn openai_responses_non_streaming_forwarded() {
    let p = spawn_proxy().await;
    mount_native_models(&p, "openai/gpt-oss-20b").await;

    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .and(body_partial_json(json!({
            "model": "openai/gpt-oss-20b",
            "input": "Provide a prime number less than 50"
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-response-id", "resp-001")
                .set_body_json(json!({
                    "id": "resp_001",
                    "object": "response",
                    "output": [{ "type": "message", "content": "47" }]
                })),
        )
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/v1/responses"))
        .json(&json!({
            "model": "openai/gpt-oss-20b",
            "input": "Provide a prime number less than 50"
        }))
        .send()
        .await
        .expect("POST /v1/responses");

    assert_eq!(resp.status(), 200);
    assert_eq!(resp.headers().get("x-response-id").unwrap(), "resp-001");
    let body: serde_json::Value = resp.json().await.expect("json body");
    assert_eq!(body["object"], "response");
}

#[tokio::test]
async fn openai_responses_streaming_bytes_roundtrip() {
    let p = spawn_proxy().await;
    mount_native_models(&p, "openai/gpt-oss-20b").await;

    let sse_payload = concat!(
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_002\"}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"47\"}\n\n",
        "data: {\"type\":\"response.completed\"}\n\n"
    );

    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .and(body_partial_json(json!({ "stream": true })))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(sse_payload.as_bytes(), "text/event-stream"),
        )
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/v1/responses"))
        .json(&json!({
            "model": "openai/gpt-oss-20b",
            "input": "Hello",
            "stream": true
        }))
        .send()
        .await
        .expect("POST /v1/responses streaming");

    assert_eq!(resp.status(), 200);
    let bytes = resp.bytes().await.expect("bytes");
    assert_eq!(bytes.as_ref(), sse_payload.as_bytes());
}

// ── POST /v1/responses — model name remapping ────────────────────────────────

#[tokio::test]
async fn openai_responses_model_remapped_and_forwarded() {
    let p = spawn_proxy().await;
    // Full LM Studio key; client will send only the short tail.
    mount_native_models(&p, "openai/gpt-oss-20b").await;

    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .and(body_partial_json(json!({
            "model": "openai/gpt-oss-20b",
            "input": "Say one word"
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-response-id", "resp-remap-001")
                .set_body_json(json!({
                    "id": "resp_remap_001",
                    "object": "response",
                    "output": [{ "type": "message", "content": "Sure" }]
                })),
        )
        .expect(1)
        .mount(&p.mock)
        .await;

    // Client uses a short Ollama-style name; proxy must resolve and remap it.
    let resp = p
        .client
        .post(p.url("/v1/responses"))
        .json(&json!({
            "model": "gpt-oss-20b",
            "input": "Say one word"
        }))
        .send()
        .await
        .expect("POST /v1/responses model remap");

    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers().get("x-response-id").unwrap(),
        "resp-remap-001"
    );
    let body: serde_json::Value = resp.json().await.expect("json body");
    assert_eq!(body["object"], "response");
}

// ── POST /v1/chat/completions — structured output (response_format) ───────────

#[tokio::test]
async fn openai_chat_completions_structured_output_forwarded() {
    let p = spawn_proxy().await;
    mount_native_models(&p, "lmstudio-community/meta-llama-3.1-8b").await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({
            "response_format": { "type": "json_schema" }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-3",
            "object": "chat.completion",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "{\"name\":\"Alice\",\"age\":30}"
                }
            }]
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/v1/chat/completions"))
        .json(&json!({
            "model": "lmstudio-community/meta-llama-3.1-8b",
            "messages": [{ "role": "user", "content": "Give me a person" }],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "person",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "name": { "type": "string" },
                            "age":  { "type": "integer" }
                        },
                        "required": ["name", "age"]
                    }
                }
            }
        }))
        .send()
        .await
        .expect("POST /v1/chat/completions structured output");

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.expect("json body");
    assert_eq!(body["object"], "chat.completion");
}

// ── POST /v1/chat/completions — function calling (tools) ──────────────────────

#[tokio::test]
async fn openai_chat_completions_tools_forwarded() {
    let p = spawn_proxy().await;
    mount_native_models(&p, "lmstudio-community/meta-llama-3.1-8b").await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({ "tool_choice": "auto" })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-4",
            "object": "chat.completion",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"Paris\"}"
                        }
                    }]
                }
            }]
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/v1/chat/completions"))
        .json(&json!({
            "model": "lmstudio-community/meta-llama-3.1-8b",
            "messages": [{ "role": "user", "content": "What is the weather in Paris?" }],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": { "city": { "type": "string" } },
                        "required": ["city"]
                    }
                }
            }],
            "tool_choice": "auto"
        }))
        .send()
        .await
        .expect("POST /v1/chat/completions tools");

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.expect("json body");
    assert_eq!(
        body["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "get_weather"
    );
}

// ── Request header forwarding ─────────────────────────────────────────────────

#[tokio::test]
async fn openai_request_authorization_header_forwarded() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": []
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .get(p.url("/v1/models"))
        .header("authorization", "Bearer test-key")
        .send()
        .await
        .expect("GET /v1/models with auth header");

    assert_eq!(resp.status(), 200);
}

// ── Virtual alias resolution on the /v1 passthrough ──────────────────────────
//
// A `/api/copy` alias must resolve on /v1 just like an Ollama-style short name:
// the alias maps to its `target_model_id` before the request reaches the backend.
// The passthrough forwards the client body otherwise verbatim, so only the model
// identifier is rewritten — virtual metadata (system prompt, etc.) is a /api
// concern, not a raw /v1 translation.

/// Seed a `/api/copy` alias `alias` -> `target_key` against the mounted catalog.
async fn create_copy_alias(p: &crate::common::TestProxy, alias: &str, target_key: &str) {
    let resp = p
        .client
        .post(p.url("/api/copy"))
        .json(&json!({ "source": target_key, "destination": alias }))
        .send()
        .await
        .expect("POST /api/copy");
    assert_eq!(resp.status(), 200, "alias creation must succeed");
}

#[tokio::test]
async fn openai_chat_completions_resolves_virtual_alias_in_body() {
    let p = spawn_proxy().await;
    let target = "lmstudio-community/meta-llama-3.1-8b";
    mount_native_models(&p, target).await;
    create_copy_alias(&p, "my-alias", target).await;

    // Backend must receive the RESOLVED target id, not the alias.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({ "model": target })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-alias",
            "object": "chat.completion",
            "choices": [{ "message": { "role": "assistant", "content": "ok" } }]
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/v1/chat/completions"))
        .json(&json!({
            "model": "my-alias",
            "messages": [{ "role": "user", "content": "Hello" }],
            "stream": false
        }))
        .send()
        .await
        .expect("POST /v1/chat/completions alias");

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.expect("json body");
    assert_eq!(body["object"], "chat.completion");
}

#[tokio::test]
async fn openai_models_path_resolves_virtual_alias() {
    let p = spawn_proxy().await;
    let target = "lmstudio-community/meta-llama-3.1-8b";
    mount_native_models(&p, target).await;
    create_copy_alias(&p, "my-alias", target).await;

    // Backend must receive the RESOLVED path segment, not the alias.
    Mock::given(method("GET"))
        .and(path(format!("/v1/models/{}", target)))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": target,
            "object": "model"
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .get(p.url("/v1/models/my-alias"))
        .send()
        .await
        .expect("GET /v1/models alias");

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.expect("json body");
    assert_eq!(body["id"], target);
}

// ── GET /v1/models lists proxy-created virtual aliases ────────────────────────
//
// A `/api/copy` alias lives only in the proxy's VirtualModelStore; LM Studio's
// listing cannot know it. `/api/tags` merges aliases in, and a `/v1` caller can
// already use one as a `model` or a `/v1/models/{alias}` path, so the listing a
// client browses must show it too.

#[tokio::test]
async fn openai_models_list_includes_virtual_alias() {
    let p = spawn_proxy().await;
    let target = "lmstudio-community/meta-llama-3.1-8b";
    let other = "publisher-two/second-model";
    mount_native_models(&p, target).await;
    create_copy_alias(&p, "my-alias", target).await;

    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-lmstudio-backend", "yes")
                .set_body_json(json!({
                    "object": "list",
                    "data": [
                        { "id": other, "object": "model", "owned_by": "publisher-two", "created": 111 },
                        { "id": target, "object": "model", "owned_by": "organization_owner", "created": 222 }
                    ]
                })),
        )
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .get(p.url("/v1/models"))
        .send()
        .await
        .expect("GET /v1/models");

    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers().get("x-lmstudio-backend").unwrap(),
        "yes",
        "the backend's response headers must survive the rebuilt envelope"
    );
    let body: serde_json::Value = resp.json().await.expect("json body");
    let data = body["data"].as_array().expect("data array");
    let ids: Vec<&str> = data.iter().filter_map(|e| e["id"].as_str()).collect();
    assert!(
        ids.contains(&target),
        "the backend's own entries must survive the merge: {body}"
    );
    assert!(
        ids.contains(&"my-alias"),
        "the `/api/copy` alias must be listed: {body}"
    );
    let alias_entry = data
        .iter()
        .find(|e| e["id"] == "my-alias")
        .expect("alias entry");
    assert_eq!(
        alias_entry["object"], "model",
        "the alias entry must be an OpenAI model object: {body}"
    );
    // The alias stands for `target`, so the fields the backend reports for that
    // model — not for whichever entry happens to be first — shape its row.
    assert_eq!(
        alias_entry["owned_by"], "organization_owner",
        "the alias row mirrors its target's owned_by: {body}"
    );
    assert_eq!(
        alias_entry["created"], 222,
        "the alias row mirrors its target's created: {body}"
    );
}

#[tokio::test]
async fn openai_models_path_never_merges_aliases() {
    let p = spawn_proxy().await;
    let target = "lmstudio-community/meta-llama-3.1-8b";
    mount_native_models(&p, target).await;
    create_copy_alias(&p, "my-alias", target).await;

    // A per-model body that happens to carry a `data` array: only the exact
    // listing path may merge, so this must round-trip byte-for-byte.
    let backend_body = json!({
        "id": target,
        "object": "model",
        "data": []
    });
    Mock::given(method("GET"))
        .and(path(format!("/v1/models/{target}")))
        .respond_with(ResponseTemplate::new(200).set_body_json(backend_body.clone()))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .get(p.url("/v1/models/my-alias"))
        .send()
        .await
        .expect("GET /v1/models/{alias}");

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.expect("json body");
    assert_eq!(
        body, backend_body,
        "the per-segment route must not merge aliases into its body: {body}"
    );
}
