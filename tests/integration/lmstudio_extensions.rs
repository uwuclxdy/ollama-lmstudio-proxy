// Integration tests for the /v1/* LM Studio extension passthroughs that
// resolve+remap the model field then forward the body verbatim.
//
// Mirrors `lmstudio_openai.rs::openai_responses_*` for /v1/responses (model
// remap pin) and adds /v1/images/generations (image-gen passthrough). Both ride
// the generic `/v1/{*path}` passthrough route (`src/proxy/routes.rs`), which
// resolves the Ollama-style model name to the LM Studio key before forwarding.

use serde_json::json;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, ResponseTemplate};

use crate::common::spawn_proxy;

/// GET /api/v1/models stub returning a single model whose key is `model_key`.
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

// ── POST /v1/responses — model remapped, body forwarded verbatim ────────────

#[tokio::test]
async fn responses_model_remapped_and_input_forwarded_verbatim() {
    let p = spawn_proxy().await;
    mount_native_models(&p, "openai/gpt-oss-20b").await;

    // The backend must receive the RESOLVED model id (full LM Studio key) and
    // the caller's `input` exactly as sent.
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

    // Client uses the short Ollama-style tail; proxy resolves and remaps it.
    let resp = p
        .client
        .post(p.url("/v1/responses"))
        .json(&json!({
            "model": "gpt-oss-20b",
            "input": "Provide a prime number less than 50"
        }))
        .send()
        .await
        .expect("POST /v1/responses");

    assert_eq!(resp.status(), 200);
    assert_eq!(resp.headers().get("x-response-id").unwrap(), "resp-001");
    let body: serde_json::Value = resp.json().await.expect("json body");
    assert_eq!(body["object"], "response");

    p.mock.verify().await;
}

// ── POST /v1/images/generations — body forwarded with model resolved ────────

#[tokio::test]
async fn images_generations_model_remapped_and_prompt_forwarded() {
    let p = spawn_proxy().await;
    mount_native_models(&p, "openai/dall-e-3").await;

    Mock::given(method("POST"))
        .and(path("/v1/images/generations"))
        .and(body_partial_json(json!({
            "model": "openai/dall-e-3",
            "prompt": "a cat in a tiny astronaut helmet",
            "n": 1,
            "size": "1024x1024"
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-lmstudio-images", "yes")
                .set_body_json(json!({
                    "created": 1_700_000_000u64,
                    "data": [{ "url": "https://images.test/cat-astronaut.png" }]
                })),
        )
        .expect(1)
        .mount(&p.mock)
        .await;

    // Client uses a short name; the proxy must resolve it to the full LM Studio
    // key before forwarding the body.
    let resp = p
        .client
        .post(p.url("/v1/images/generations"))
        .json(&json!({
            "model": "dall-e-3",
            "prompt": "a cat in a tiny astronaut helmet",
            "n": 1,
            "size": "1024x1024"
        }))
        .send()
        .await
        .expect("POST /v1/images/generations");

    assert_eq!(resp.status(), 200);
    assert_eq!(resp.headers().get("x-lmstudio-images").unwrap(), "yes");
    let body: serde_json::Value = resp.json().await.expect("json body");
    assert!(
        body["data"].is_array(),
        "image-gen response must pass through unchanged: {body}"
    );
    assert_eq!(
        body["data"][0]["url"],
        "https://images.test/cat-astronaut.png"
    );

    p.mock.verify().await;
}

// ── POST /v1/images/generations — backend error propagates verbatim ─────────

#[tokio::test]
async fn images_generations_backend_error_propagates() {
    let p = spawn_proxy().await;
    mount_native_models(&p, "openai/dall-e-3").await;

    Mock::given(method("POST"))
        .and(path("/v1/images/generations"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "error": { "message": "prompt rejected", "type": "invalid_request_error" }
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/v1/images/generations"))
        .json(&json!({ "model": "dall-e-3", "prompt": "x" }))
        .send()
        .await
        .expect("POST /v1/images/generations 400");

    assert_eq!(resp.status(), 400);
    let body: serde_json::Value = resp.json().await.expect("json body");
    assert_eq!(body["error"]["type"], "invalid_request_error");
}

// ── stream:true + backend non-2xx — real status, never a fabricated 200 SSE ──

#[tokio::test]
async fn passthrough_stream_backend_4xx_keeps_status_not_fake_sse() {
    let p = spawn_proxy().await;
    mount_native_models(&p, "meta/llama-3.2").await;

    // The backend rejects a bad stream:true request with a plain JSON 400 (no
    // SSE framing) — the exact shape a non-SSE upstream error takes.
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "type": "error",
            "error": { "type": "invalid_request_error", "message": "max_tokens must be positive" }
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/v1/messages"))
        .json(&json!({
            "model": "llama-3.2",
            "stream": true,
            "max_tokens": -5,
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .send()
        .await
        .expect("POST /v1/messages stream:true");

    // Must surface the real 400, never a fabricated 200 text/event-stream.
    assert_eq!(
        resp.status(),
        400,
        "a non-2xx backend response to a stream:true passthrough must keep its status"
    );
    let ctype = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        !ctype.contains("text/event-stream"),
        "a backend error must not be wrapped in an SSE stream; got content-type '{ctype}'"
    );
    let body: serde_json::Value = resp.json().await.expect("json error body");
    assert_eq!(body["error"]["type"], "invalid_request_error");

    p.mock.verify().await;
}

// ── response_format json_object rewritten to a permissive json_schema ────────

#[tokio::test]
async fn chat_completions_json_object_rewritten_to_json_schema() {
    let p = spawn_proxy().await;
    mount_native_models(&p, "meta/llama-3.2").await;

    // LM Studio rejects a bare json_object; the backend mock only matches when
    // the proxy has rewritten it into a json_schema envelope. An unrewritten
    // json_object would miss the mock and surface a 404.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({
            "model": "meta/llama-3.2",
            "response_format": {
                "type": "json_schema",
                "json_schema": { "schema": { "type": "object" } }
            }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "{}" },
                "finish_reason": "stop"
            }]
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/v1/chat/completions"))
        .json(&json!({
            "model": "llama-3.2",
            "messages": [{ "role": "user", "content": "give me json" }],
            "response_format": { "type": "json_object" }
        }))
        .send()
        .await
        .expect("POST /v1/chat/completions json_object");

    assert_eq!(
        resp.status(),
        200,
        "json_object must be rewritten to json_schema and accepted; got {}",
        resp.status()
    );

    p.mock.verify().await;
}

// ── /v1/messages proxy errors wear the Anthropic envelope ───────────────────

#[tokio::test]
async fn messages_unknown_model_returns_anthropic_error_envelope() {
    let p = spawn_proxy().await;
    mount_native_models(&p, "meta/llama-3.2").await;

    let resp = p
        .client
        .post(p.url("/v1/messages"))
        .json(&json!({
            "model": "does-not-exist",
            "max_tokens": 8,
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .send()
        .await
        .expect("POST /v1/messages unknown model");

    assert_eq!(resp.status(), 404);
    let body: serde_json::Value = resp.json().await.expect("json body");
    assert_eq!(body["type"], "error", "Anthropic envelope: {body}");
    assert_eq!(body["error"]["type"], "not_found_error");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("not found")),
        "message must carry the resolver detail: {body}"
    );
}

// ── /v1/models/{model}: path segment resolved like a body model ─────────────

#[tokio::test]
async fn models_path_segment_resolved_from_alias() {
    let p = spawn_proxy().await;
    mount_native_models(&p, "meta/llama-3.2").await;

    // The backend must be hit with the RESOLVED key in the path; an
    // unresolved alias would miss this mock and 404.
    Mock::given(method("GET"))
        .and(path("/v1/models/meta/llama-3.2"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "meta/llama-3.2",
            "object": "model",
            "owned_by": "organization_owner"
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .get(p.url("/v1/models/llama-3.2:latest"))
        .send()
        .await
        .expect("GET /v1/models/{alias}");

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.expect("json body");
    assert_eq!(body["id"], "meta/llama-3.2");

    p.mock.verify().await;
}

// ── /v1/embeddings encoding_format:"base64" transcoded by the proxy ─────────

#[tokio::test]
async fn embeddings_base64_transcoded_from_backend_floats() {
    use base64::Engine as _;

    let p = spawn_proxy().await;
    mount_native_models(&p, "nomic/embed-text").await;

    // LM Studio ignores encoding_format and always returns float arrays; the
    // proxy strips the field outbound and self-encodes the response.
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .and(body_partial_json(json!({ "model": "nomic/embed-text" })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [{ "object": "embedding", "index": 0, "embedding": [1.0, -0.5] }],
            "model": "nomic/embed-text",
            "usage": { "prompt_tokens": 2, "total_tokens": 2 }
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/v1/embeddings"))
        .json(&json!({
            "model": "nomic/embed-text",
            "input": "hi",
            "encoding_format": "base64"
        }))
        .send()
        .await
        .expect("POST /v1/embeddings base64");

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.expect("json body");
    let encoded = body["data"][0]["embedding"]
        .as_str()
        .expect("embedding must be a base64 string, not a float array");

    // OpenAI convention: base64 over the f32 little-endian byte concatenation.
    let mut expected_bytes = Vec::new();
    expected_bytes.extend_from_slice(&1.0f32.to_le_bytes());
    expected_bytes.extend_from_slice(&(-0.5f32).to_le_bytes());
    assert_eq!(
        encoded,
        base64::engine::general_purpose::STANDARD.encode(&expected_bytes)
    );
    assert_eq!(
        body["usage"]["prompt_tokens"], 2,
        "other fields pass through"
    );

    p.mock.verify().await;
}
