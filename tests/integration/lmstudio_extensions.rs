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
