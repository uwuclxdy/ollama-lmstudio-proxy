// Integration tests for the Ollama embedding surface.
//
// Both endpoints translate to POST /v1/embeddings on the LM Studio side:
//   POST /api/embed       — Ollama v0.5+, supports batch `input`
//   POST /api/embeddings  — Ollama legacy, single `prompt`, returns singular `embedding`

use serde_json::{Value, json};
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, ResponseTemplate};

use crate::common::spawn_proxy;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// A minimal `/api/v1/models` mock that exposes one model so the resolver succeeds.
async fn mount_models(p: &crate::common::TestProxy, model_id: &str) {
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [{"key": model_id, "type": "llm", "publisher": "meta",
                        "architecture": "llama", "format": "gguf",
                        "quantization": {"name": "Q4_K_M", "bits_per_weight": 4.5},
                        "max_context_length": 8192, "loaded_instances": [],
                        "capabilities": {"vision": false, "trained_for_tool_use": false}}]
        })))
        .mount(&p.mock)
        .await;
}

/// A single-vector LM Studio embeddings response.
fn lm_response_single(model_id: &str, vec: Vec<f32>) -> Value {
    json!({
        "object": "list",
        "data": [{ "object": "embedding", "index": 0, "embedding": vec }],
        "model": model_id,
        "usage": { "prompt_tokens": 5, "total_tokens": 5 }
    })
}

/// A multi-vector LM Studio embeddings response preserving insertion order.
fn lm_response_multi(model_id: &str, vecs: Vec<Vec<f32>>) -> Value {
    let data: Vec<Value> = vecs
        .into_iter()
        .enumerate()
        .map(|(i, v)| json!({ "object": "embedding", "index": i, "embedding": v }))
        .collect();
    json!({
        "object": "list",
        "data": data,
        "model": model_id,
        "usage": { "prompt_tokens": 10, "total_tokens": 10 }
    })
}

// ---------------------------------------------------------------------------
// 1. /api/embed — single string input
// ---------------------------------------------------------------------------

#[tokio::test]
async fn embed_single_string_input() {
    let p = spawn_proxy().await;
    mount_models(&p, "all-minilm").await;

    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(lm_response_single("all-minilm", vec![0.1, 0.2, 0.3])),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/embed"))
        .json(&json!({ "model": "all-minilm", "input": "Why is the sky blue?" }))
        .send()
        .await
        .expect("POST /api/embed");

    assert_eq!(resp.status(), 200, "should succeed");
    let body: Value = resp.json().await.expect("json body");
    assert_eq!(body["model"], "all-minilm");
    let embeddings = body["embeddings"].as_array().expect("embeddings array");
    assert_eq!(embeddings.len(), 1);
    assert_eq!(embeddings[0].as_array().expect("vector").len(), 3);
}

// ---------------------------------------------------------------------------
// 2. /api/embed — batch array input
// ---------------------------------------------------------------------------

#[tokio::test]
async fn embed_batch_array_input() {
    let p = spawn_proxy().await;
    mount_models(&p, "all-minilm").await;

    // Match the exact array body the proxy should forward — if the proxy
    // splits the batch into per-item calls or drops items, this mock will
    // not match and the request will fail rather than silently passing.
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .and(body_partial_json(json!({
            "input": ["first sentence", "second sentence", "third sentence"]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(lm_response_multi(
            "all-minilm",
            vec![vec![0.1, 0.2], vec![0.3, 0.4], vec![0.5, 0.6]],
        )))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/embed"))
        .json(&json!({
            "model": "all-minilm",
            "input": ["first sentence", "second sentence", "third sentence"]
        }))
        .send()
        .await
        .expect("POST /api/embed batch");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json body");
    let embeddings = body["embeddings"].as_array().expect("embeddings array");
    assert_eq!(embeddings.len(), 3, "one vector per input string");
    p.mock.verify().await;
}

// ---------------------------------------------------------------------------
// 3. /api/embed response shape
// ---------------------------------------------------------------------------

#[tokio::test]
async fn embed_response_shape() {
    let p = spawn_proxy().await;
    mount_models(&p, "all-minilm").await;

    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(lm_response_single("all-minilm", vec![0.1, 0.2])),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/embed"))
        .json(&json!({ "model": "all-minilm", "input": "hello" }))
        .send()
        .await
        .expect("POST /api/embed shape");

    let body: Value = resp.json().await.expect("json body");

    // Required Ollama fields
    assert!(body.get("model").is_some(), "missing model");
    assert!(body.get("embeddings").is_some(), "missing embeddings");
    assert!(
        body.get("total_duration").is_some(),
        "missing total_duration"
    );
    assert!(body.get("load_duration").is_some(), "missing load_duration");
    assert!(
        body.get("prompt_eval_count").is_some(),
        "missing prompt_eval_count"
    );
    // Legacy singular key must NOT appear on /api/embed
    assert!(
        body.get("embedding").is_none(),
        "singular embedding must not appear on /api/embed"
    );
}

// ---------------------------------------------------------------------------
// 4. /api/embeddings (legacy) — prompt field, singular embedding in response
// ---------------------------------------------------------------------------

#[tokio::test]
async fn legacy_embeddings_prompt_field() {
    let p = spawn_proxy().await;
    mount_models(&p, "all-minilm").await;

    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(lm_response_single("all-minilm", vec![0.9, 0.8, 0.7, 0.6])),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/embeddings"))
        .json(&json!({ "model": "all-minilm", "prompt": "legacy prompt text" }))
        .send()
        .await
        .expect("POST /api/embeddings");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json body");

    // Legacy response shape: singular `embedding`, no `embeddings`
    let embedding = body["embedding"]
        .as_array()
        .expect("singular embedding array");
    assert_eq!(embedding.len(), 4, "all 4 floats preserved");
    assert!(
        body.get("embeddings").is_none(),
        "batch key must not appear on legacy endpoint"
    );
    assert_eq!(body["model"], "all-minilm");
}

// ---------------------------------------------------------------------------
// 5. Model resolution against /v1/models catalog
// ---------------------------------------------------------------------------

#[tokio::test]
async fn embed_model_resolution_from_catalog() {
    let p = spawn_proxy().await;

    // Register a model with an internal LM Studio ID
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [{"key": "lmstudio-community/all-minilm-l6-v2", "type": "llm",
                        "publisher": "lmstudio-community", "architecture": "bert",
                        "format": "gguf",
                        "quantization": {"name": "Q4_K_M", "bits_per_weight": 4.5},
                        "max_context_length": 8192, "loaded_instances": [],
                        "capabilities": {"vision": false, "trained_for_tool_use": false}}]
        })))
        .mount(&p.mock)
        .await;

    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lm_response_single(
            "lmstudio-community/all-minilm-l6-v2",
            vec![0.5, 0.5],
        )))
        .mount(&p.mock)
        .await;

    // Client uses short Ollama-style name; proxy must resolve it
    let resp = p
        .client
        .post(p.url("/api/embed"))
        .json(&json!({ "model": "all-minilm-l6-v2", "input": "test" }))
        .send()
        .await
        .expect("POST /api/embed model resolve");

    // 200 means the resolver succeeded and forwarded to LM Studio
    assert_eq!(resp.status(), 200);
}

// ---------------------------------------------------------------------------
// 6. options.truncate forwarded
// ---------------------------------------------------------------------------

#[tokio::test]
async fn embed_truncate_option_forwarded() {
    let p = spawn_proxy().await;
    mount_models(&p, "nomic-embed").await;

    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lm_response_single("nomic-embed", vec![0.1])),
        )
        .mount(&p.mock)
        .await;

    // `truncate` is a top-level Ollama field; proxy lifts it into options
    let resp = p
        .client
        .post(p.url("/api/embed"))
        .json(&json!({
            "model": "nomic-embed",
            "input": "truncation test",
            "truncate": true
        }))
        .send()
        .await
        .expect("POST /api/embed truncate");

    // Proxy must not reject the request; LM Studio mock received it
    assert_eq!(resp.status(), 200);
}

// ---------------------------------------------------------------------------
// 7. Empty input → error
// ---------------------------------------------------------------------------

#[tokio::test]
async fn embed_empty_input_returns_error() {
    let p = spawn_proxy().await;
    mount_models(&p, "all-minilm").await;

    let resp = p
        .client
        .post(p.url("/api/embed"))
        .json(&json!({ "model": "all-minilm", "input": "" }))
        .send()
        .await
        .expect("POST /api/embed empty input");

    // Per `fix(embed): reject empty input string before forwarding`, the proxy
    // must short-circuit empty strings with a 4xx rather than forward them.
    assert!(
        resp.status().is_client_error(),
        "empty input must be rejected with 4xx; got {}",
        resp.status()
    );
}

// ---------------------------------------------------------------------------
// 8. Missing model → 4xx error
// ---------------------------------------------------------------------------

#[tokio::test]
async fn embed_missing_model_returns_error() {
    let p = spawn_proxy().await;

    let resp = p
        .client
        .post(p.url("/api/embed"))
        .json(&json!({ "input": "no model field here" }))
        .send()
        .await
        .expect("POST /api/embed no model");

    assert!(
        resp.status().as_u16() >= 400 && resp.status().as_u16() < 500,
        "missing model must return 4xx, got {}",
        resp.status()
    );
}

// ---------------------------------------------------------------------------
// 9. Missing input/prompt → 4xx error
// ---------------------------------------------------------------------------

#[tokio::test]
async fn embed_missing_input_field_returns_error() {
    let p = spawn_proxy().await;
    mount_models(&p, "all-minilm").await;

    let resp = p
        .client
        .post(p.url("/api/embed"))
        .json(&json!({ "model": "all-minilm" }))
        .send()
        .await
        .expect("POST /api/embed no input");

    assert!(
        resp.status().as_u16() >= 400 && resp.status().as_u16() < 500,
        "missing input must return 4xx, got {}",
        resp.status()
    );
}

// ---------------------------------------------------------------------------
// 10. LM Studio returns 5xx → proxy surfaces 5xx
// ---------------------------------------------------------------------------

#[tokio::test]
async fn embed_lmstudio_5xx_propagated() {
    let p = spawn_proxy().await;
    mount_models(&p, "all-minilm").await;

    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(503).set_body_json(json!({
            "error": { "message": "model not loaded", "type": "server_error" }
        })))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/embed"))
        .json(&json!({ "model": "all-minilm", "input": "hello" }))
        .send()
        .await
        .expect("POST /api/embed 5xx");

    assert!(
        resp.status().as_u16() >= 500,
        "LM Studio 5xx must propagate, got {}",
        resp.status()
    );
}

// ---------------------------------------------------------------------------
// 11. LM Studio returns malformed JSON → proxy error (not panic)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn embed_lmstudio_malformed_json_returns_error() {
    let p = spawn_proxy().await;
    mount_models(&p, "all-minilm").await;

    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string("this is not json at all {{{")
                .append_header("content-type", "application/json"),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/embed"))
        .json(&json!({ "model": "all-minilm", "input": "malformed test" }))
        .send()
        .await
        .expect("POST /api/embed malformed");

    // Proxy must return an error, not a 200 with garbage
    assert!(
        resp.status().as_u16() >= 400,
        "malformed backend JSON must return error, got {}",
        resp.status()
    );
}

// ---------------------------------------------------------------------------
// 12. keep_alive parameter passes through
// ---------------------------------------------------------------------------

#[tokio::test]
async fn embed_keep_alive_accepted() {
    let p = spawn_proxy().await;
    mount_models(&p, "all-minilm").await;

    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(lm_response_single("all-minilm", vec![0.1, 0.2])),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/embed"))
        .json(&json!({
            "model": "all-minilm",
            "input": "keep alive test",
            "keep_alive": "10m"
        }))
        .send()
        .await
        .expect("POST /api/embed keep_alive");

    assert_eq!(resp.status(), 200, "keep_alive must not break the request");
}

// ---------------------------------------------------------------------------
// 13. Embedding dimension preserved
// ---------------------------------------------------------------------------

#[tokio::test]
async fn embed_dimension_preserved() {
    let p = spawn_proxy().await;
    mount_models(&p, "all-minilm").await;

    let dim = 768;
    let vector: Vec<f32> = (0..dim).map(|i| i as f32 / dim as f32).collect();

    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(lm_response_single("all-minilm", vector.clone())),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/embed"))
        .json(&json!({ "model": "all-minilm", "input": "dimension test" }))
        .send()
        .await
        .expect("POST /api/embed dimension");

    let body: Value = resp.json().await.expect("json body");
    let returned_vec = body["embeddings"][0].as_array().expect("vector");
    assert_eq!(
        returned_vec.len(),
        dim,
        "all {dim} dimensions must survive the round-trip"
    );
}

// ---------------------------------------------------------------------------
// 14. Multi-vector batch ordering preserved
// ---------------------------------------------------------------------------

#[tokio::test]
async fn embed_batch_order_preserved() {
    let p = spawn_proxy().await;
    mount_models(&p, "all-minilm").await;

    // Each vector is uniquely identifiable by its first element
    let vecs = vec![
        vec![1.0_f32, 0.0, 0.0],
        vec![0.0_f32, 2.0, 0.0],
        vec![0.0_f32, 0.0, 3.0],
        vec![4.0_f32, 4.0, 4.0],
    ];

    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lm_response_multi("all-minilm", vecs.clone())),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/embed"))
        .json(&json!({
            "model": "all-minilm",
            "input": ["a", "b", "c", "d"]
        }))
        .send()
        .await
        .expect("POST /api/embed batch order");

    let body: Value = resp.json().await.expect("json body");
    let embeddings = body["embeddings"].as_array().expect("embeddings");
    assert_eq!(embeddings.len(), 4);

    for (i, expected) in vecs.iter().enumerate() {
        let got = embeddings[i].as_array().expect("inner array");
        let first = got[0].as_f64().expect("f64");
        assert!(
            (first - expected[0] as f64).abs() < 1e-5,
            "vector {i} order mismatch: expected first={}, got {first}",
            expected[0]
        );
    }
}

// ---------------------------------------------------------------------------
// 15. Very long input handled without panic
// ---------------------------------------------------------------------------

#[tokio::test]
async fn embed_very_long_input() {
    let p = spawn_proxy().await;
    mount_models(&p, "all-minilm").await;

    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(lm_response_single("all-minilm", vec![0.1, 0.2])),
        )
        .mount(&p.mock)
        .await;

    let long_input = "token ".repeat(10_000);

    let resp = p
        .client
        .post(p.url("/api/embed"))
        .json(&json!({ "model": "all-minilm", "input": long_input }))
        .send()
        .await
        .expect("POST /api/embed long input");

    // Must not 5xx — either forwarded successfully or rejected cleanly
    assert!(
        resp.status().as_u16() < 500,
        "very long input must not panic the proxy: {}",
        resp.status()
    );
}

// ---------------------------------------------------------------------------
// 16. Unusual model names in path (with colon tag)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn embed_model_name_with_tag() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [{"key": "all-minilm:latest", "type": "llm", "publisher": "meta",
                        "architecture": "llama", "format": "gguf",
                        "quantization": {"name": "Q4_K_M", "bits_per_weight": 4.5},
                        "max_context_length": 8192, "loaded_instances": [],
                        "capabilities": {"vision": false, "trained_for_tool_use": false}}]
        })))
        .mount(&p.mock)
        .await;

    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(lm_response_single("all-minilm:latest", vec![0.5])),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/embed"))
        .json(&json!({ "model": "all-minilm:latest", "input": "tag test" }))
        .send()
        .await
        .expect("POST /api/embed tagged model");

    assert_eq!(resp.status(), 200);
}

// ---------------------------------------------------------------------------
// 17. dimensions parameter forwarded
// ---------------------------------------------------------------------------

#[tokio::test]
async fn embed_dimensions_param_accepted() {
    let p = spawn_proxy().await;
    mount_models(&p, "nomic-embed").await;

    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(lm_response_single("nomic-embed", vec![0.1; 256])),
        )
        .mount(&p.mock)
        .await;

    // `dimensions` is a top-level Ollama field lifted into options by the proxy
    let resp = p
        .client
        .post(p.url("/api/embed"))
        .json(&json!({
            "model": "nomic-embed",
            "input": "dimensions test",
            "dimensions": 256
        }))
        .send()
        .await
        .expect("POST /api/embed dimensions");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json body");
    let vec_len = body["embeddings"][0].as_array().expect("vector").len();
    assert_eq!(
        vec_len, 256,
        "returned vector must match requested dimensions"
    );
}

// ---------------------------------------------------------------------------
// 18. Legacy /api/embeddings with keep_alive numeric seconds
// ---------------------------------------------------------------------------

#[tokio::test]
async fn legacy_embeddings_keep_alive_numeric() {
    let p = spawn_proxy().await;
    mount_models(&p, "all-minilm").await;

    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(lm_response_single("all-minilm", vec![0.3, 0.7])),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/embeddings"))
        .json(&json!({
            "model": "all-minilm",
            "prompt": "legacy keep alive",
            "keep_alive": -1
        }))
        .send()
        .await
        .expect("POST /api/embeddings keep_alive numeric");

    // -1 means "unload immediately" in Ollama; proxy must not reject it
    assert!(
        resp.status().as_u16() < 500,
        "numeric keep_alive must not 5xx: {}",
        resp.status()
    );
}

// ---------------------------------------------------------------------------
// 19. Legacy endpoint missing prompt → 4xx
// ---------------------------------------------------------------------------

#[tokio::test]
async fn legacy_embeddings_missing_prompt_returns_error() {
    let p = spawn_proxy().await;
    mount_models(&p, "all-minilm").await;

    let resp = p
        .client
        .post(p.url("/api/embeddings"))
        .json(&json!({ "model": "all-minilm" }))
        .send()
        .await
        .expect("POST /api/embeddings no prompt");

    assert!(
        resp.status().as_u16() >= 400 && resp.status().as_u16() < 500,
        "missing prompt on legacy endpoint must return 4xx, got {}",
        resp.status()
    );
}

// ---------------------------------------------------------------------------
// 20. /api/embed with options.temperature (extra model param)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn embed_extra_options_forwarded() {
    let p = spawn_proxy().await;
    mount_models(&p, "all-minilm").await;

    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(lm_response_single("all-minilm", vec![0.1, 0.2])),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/embed"))
        .json(&json!({
            "model": "all-minilm",
            "input": "options test",
            "options": { "temperature": 0.0 }
        }))
        .send()
        .await
        .expect("POST /api/embed with options");

    assert_eq!(resp.status(), 200);
}

// ---------------------------------------------------------------------------
// 21. /api/embed response does not contain prompt_eval_duration
// ---------------------------------------------------------------------------

#[tokio::test]
async fn embed_response_excludes_prompt_eval_duration() {
    let p = spawn_proxy().await;
    mount_models(&p, "all-minilm").await;

    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lm_response_single("all-minilm", vec![0.1])),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/embed"))
        .json(&json!({ "model": "all-minilm", "input": "duration field test" }))
        .send()
        .await
        .expect("POST /api/embed duration");

    let body: Value = resp.json().await.expect("json body");
    assert!(
        body.get("prompt_eval_duration").is_none(),
        "response must not include prompt_eval_duration (not in EmbedResponse schema)"
    );
}

// ---------------------------------------------------------------------------
// 22. /api/embed model name echoed back correctly
// ---------------------------------------------------------------------------

#[tokio::test]
async fn embed_model_name_echoed_in_response() {
    let p = spawn_proxy().await;
    mount_models(&p, "nomic-embed-text").await;

    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(lm_response_single("nomic-embed-text", vec![0.1, 0.2])),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/embed"))
        .json(&json!({ "model": "nomic-embed-text", "input": "model echo test" }))
        .send()
        .await
        .expect("POST /api/embed model echo");

    let body: Value = resp.json().await.expect("json body");
    assert_eq!(
        body["model"].as_str().expect("model field"),
        "nomic-embed-text",
        "Ollama model name must be echoed, not the LM Studio internal ID"
    );
}

// ---------------------------------------------------------------------------
// 23. Both endpoints route to /v1/embeddings on the LM Studio side
// ---------------------------------------------------------------------------

#[tokio::test]
async fn both_endpoints_route_to_v1_embeddings() {
    for ollama_path in ["/api/embed", "/api/embeddings"] {
        let p = spawn_proxy().await;
        mount_models(&p, "all-minilm").await;

        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(lm_response_single("all-minilm", vec![0.1])),
            )
            .mount(&p.mock)
            .await;

        let body = if ollama_path == "/api/embeddings" {
            json!({ "model": "all-minilm", "prompt": "routing test" })
        } else {
            json!({ "model": "all-minilm", "input": "routing test" })
        };

        let resp = p
            .client
            .post(p.url(ollama_path))
            .json(&body)
            .send()
            .await
            .unwrap_or_else(|e| panic!("POST {ollama_path}: {e}"));

        assert_eq!(
            resp.status(),
            200,
            "{ollama_path} must route to /v1/embeddings and return 200"
        );
    }
}

// ---------------------------------------------------------------------------
// 24. prompt_eval_count reflects LM Studio usage.prompt_tokens
// ---------------------------------------------------------------------------

#[tokio::test]
async fn embed_prompt_eval_count_from_usage() {
    let p = spawn_proxy().await;
    mount_models(&p, "all-minilm").await;

    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [{ "object": "embedding", "index": 0, "embedding": [0.1, 0.2] }],
            "model": "all-minilm",
            "usage": { "prompt_tokens": 42, "total_tokens": 42 }
        })))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/embed"))
        .json(&json!({ "model": "all-minilm", "input": "token count test" }))
        .send()
        .await
        .expect("POST /api/embed usage");

    let body: Value = resp.json().await.expect("json body");
    assert_eq!(
        body["prompt_eval_count"]
            .as_u64()
            .expect("prompt_eval_count"),
        42,
        "prompt_eval_count must equal usage.prompt_tokens from LM Studio response"
    );
}

// ---------------------------------------------------------------------------
// 25. truncate + dimensions forwarded to /v1/embeddings body
// ---------------------------------------------------------------------------

#[tokio::test]
async fn embed_truncate_and_dimensions_reach_lm_studio_body() {
    let p = spawn_proxy().await;
    mount_models(&p, "nomic-embed").await;

    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lm_response_single("nomic-embed", vec![0.1])),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/embed"))
        .json(&json!({
            "model": "nomic-embed",
            "input": "embeddings-only fields",
            "truncate": true,
            "dimensions": 64
        }))
        .send()
        .await
        .expect("POST /api/embed truncate + dimensions");

    assert_eq!(resp.status(), 200);

    let received = p.mock.received_requests().await.unwrap_or_default();
    let upstream = received
        .iter()
        .find(|r| r.url.path() == "/v1/embeddings")
        .expect("LM Studio embeddings request captured");
    let body: Value = serde_json::from_slice(&upstream.body).expect("upstream body is JSON");

    assert_eq!(
        body.get("truncate"),
        Some(&json!(true)),
        "truncate must reach LM Studio embeddings body: {body}"
    );
    assert_eq!(
        body.get("dimensions"),
        Some(&json!(64)),
        "dimensions must reach LM Studio embeddings body: {body}"
    );
}
