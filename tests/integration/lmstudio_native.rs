// Integration tests for /api/v0/* LM Studio native REST passthrough.
//
// The proxy forwards every /api/v0/* (and any /api/vN/*) request verbatim to
// the LM Studio backend. These tests assert transparent forwarding: method,
// path, body, and response status/headers all round-trip unchanged.
//
// Note: the native route guard requires the path segment after /api/ to start
// with 'v', so /api/v0, /api/v1, etc. are forwarded; /api/tags and other
// Ollama paths are handled by dedicated Ollama routes.

use serde_json::json;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, ResponseTemplate};

use crate::common::spawn_proxy;

// ── GET /api/v0/models (list all) ─────────────────────────────────────────────

#[tokio::test]
async fn native_models_list_forwarded() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v0/models"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-lmstudio-native", "true")
                .set_body_json(json!([
                    {
                        "id": "lmstudio-community/meta-llama-3.1-8b-instruct",
                        "object": "model",
                        "type": "llm"
                    }
                ])),
        )
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .get(p.url("/api/v0/models"))
        .send()
        .await
        .expect("GET /api/v0/models");

    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers().get("x-lmstudio-native").unwrap(),
        "true",
        "response header must pass through"
    );
    let body: serde_json::Value = resp.json().await.expect("json body");
    assert!(body.is_array());
}

// ── GET /api/v0/models/:model (single model info) ─────────────────────────────

#[tokio::test]
async fn native_model_info_forwarded() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v0/models/lmstudio-community/meta-llama-3.1-8b-instruct"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "lmstudio-community/meta-llama-3.1-8b-instruct",
            "object": "model",
            "type": "llm",
            "state": "not-loaded"
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .get(p.url("/api/v0/models/lmstudio-community/meta-llama-3.1-8b-instruct"))
        .send()
        .await
        .expect("GET /api/v0/models/:model");

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.expect("json body");
    assert_eq!(body["type"], "llm");
}

#[tokio::test]
async fn native_model_info_not_found_propagated() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v0/models/no-such-model"))
        .respond_with(ResponseTemplate::new(404).set_body_json(json!({
            "error": "model not found"
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .get(p.url("/api/v0/models/no-such-model"))
        .send()
        .await
        .expect("GET /api/v0/models/no-such-model 404");

    assert_eq!(resp.status(), 404);
}

// ── POST /api/v0/chat/completions ─────────────────────────────────────────────

#[tokio::test]
async fn native_chat_completions_forwarded() {
    let p = spawn_proxy().await;

    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .and(body_partial_json(json!({
            "model": "lmstudio-community/meta-llama-3.1-8b-instruct",
            "messages": [{ "role": "user", "content": "Hi" }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model_instance_id": "lmstudio-community/meta-llama-3.1-8b-instruct",
            "output": [{ "type": "message", "content": "Hello!" }],
            "stats": { "input_tokens": 5, "total_output_tokens": 3 }
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/v0/chat/completions"))
        .json(&json!({
            "model": "lmstudio-community/meta-llama-3.1-8b-instruct",
            "messages": [{ "role": "user", "content": "Hi" }]
        }))
        .send()
        .await
        .expect("POST /api/v0/chat/completions");

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.expect("json body");
    assert!(body["output"].is_array());
}

#[tokio::test]
async fn native_chat_completions_streaming_bytes_roundtrip() {
    let p = spawn_proxy().await;

    // LM Studio native streaming uses NDJSON lines prefixed with "data: "
    let sse_payload = concat!(
        "data: {\"type\":\"chat.start\"}\n\n",
        "data: {\"type\":\"message.delta\",\"delta\":\"Hello\"}\n\n",
        "data: {\"type\":\"chat.end\"}\n\n"
    );

    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .and(body_partial_json(json!({ "stream": true })))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(sse_payload.as_bytes(), "text/event-stream"),
        )
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/v0/chat/completions"))
        .json(&json!({
            "model": "lmstudio-community/meta-llama-3.1-8b-instruct",
            "messages": [{ "role": "user", "content": "Hi" }],
            "stream": true
        }))
        .send()
        .await
        .expect("POST /api/v0/chat/completions streaming");

    assert_eq!(resp.status(), 200);
    let bytes = resp.bytes().await.expect("bytes");
    assert_eq!(bytes.as_ref(), sse_payload.as_bytes());
}

// ── POST /api/v0/completions ──────────────────────────────────────────────────

#[tokio::test]
async fn native_completions_forwarded() {
    let p = spawn_proxy().await;

    Mock::given(method("POST"))
        .and(path("/api/v0/completions"))
        .and(body_partial_json(json!({
            "model": "lmstudio-community/meta-llama-3.1-8b-instruct",
            "prompt": "The quick brown fox"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model_instance_id": "lmstudio-community/meta-llama-3.1-8b-instruct",
            "completion": " jumped over the lazy dog."
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/v0/completions"))
        .json(&json!({
            "model": "lmstudio-community/meta-llama-3.1-8b-instruct",
            "prompt": "The quick brown fox"
        }))
        .send()
        .await
        .expect("POST /api/v0/completions");

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.expect("json body");
    assert!(body["completion"].is_string());
}

// ── POST /api/v0/embeddings ───────────────────────────────────────────────────

#[tokio::test]
async fn native_embeddings_forwarded() {
    let p = spawn_proxy().await;

    Mock::given(method("POST"))
        .and(path("/api/v0/embeddings"))
        .and(body_partial_json(json!({
            "model": "text-embedding-nomic-embed-text-v1.5",
            "input": "hello"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "text-embedding-nomic-embed-text-v1.5",
            "data": [{ "embedding": [0.1, 0.2] }]
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/v0/embeddings"))
        .json(&json!({
            "model": "text-embedding-nomic-embed-text-v1.5",
            "input": "hello"
        }))
        .send()
        .await
        .expect("POST /api/v0/embeddings");

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.expect("json body");
    assert!(body["data"].is_array());
}

// ── POST /api/v0/models/download ──────────────────────────────────────────────

#[tokio::test]
async fn native_model_download_forwarded() {
    let p = spawn_proxy().await;

    Mock::given(method("POST"))
        .and(path("/api/v0/models/download"))
        .and(body_partial_json(json!({
            "model": "lmstudio-community/meta-llama-3.1-8b-instruct"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "job_id": "job_493c7c9ded",
            "status": "downloading"
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/v0/models/download"))
        .json(&json!({
            "model": "lmstudio-community/meta-llama-3.1-8b-instruct"
        }))
        .send()
        .await
        .expect("POST /api/v0/models/download");

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.expect("json body");
    assert_eq!(body["job_id"], "job_493c7c9ded");
}

// ── GET /api/v0/downloads (list active downloads / download status) ───────────

#[tokio::test]
async fn native_downloads_list_forwarded() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v0/downloads"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!([
            {
                "job_id": "job_493c7c9ded",
                "status": "downloading",
                "total_size_bytes": 2279145003u64,
                "downloaded_bytes": 500000000u64
            }
        ])))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .get(p.url("/api/v0/downloads"))
        .send()
        .await
        .expect("GET /api/v0/downloads");

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.expect("json body");
    assert!(body.is_array());
    assert_eq!(body[0]["job_id"], "job_493c7c9ded");
}

// ── POST /api/v0/downloads/:id/cancel ────────────────────────────────────────

#[tokio::test]
async fn native_download_cancel_forwarded() {
    let p = spawn_proxy().await;

    Mock::given(method("POST"))
        .and(path("/api/v0/downloads/job_493c7c9ded/cancel"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "job_id": "job_493c7c9ded",
            "status": "cancelled"
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/v0/downloads/job_493c7c9ded/cancel"))
        .send()
        .await
        .expect("POST /api/v0/downloads/:id/cancel");

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.expect("json body");
    assert_eq!(body["status"], "cancelled");
}

// ── POST /api/v0/models/load ──────────────────────────────────────────────────

#[tokio::test]
async fn native_model_load_forwarded() {
    let p = spawn_proxy().await;

    Mock::given(method("POST"))
        .and(path("/api/v0/models/load"))
        .and(body_partial_json(json!({
            "model": "lmstudio-community/meta-llama-3.1-8b-instruct"
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-lmstudio-load-time", "1.23")
                .set_body_json(json!({
                    "model_instance_id": "lmstudio-community/meta-llama-3.1-8b-instruct",
                    "status": "loaded"
                })),
        )
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/v0/models/load"))
        .json(&json!({
            "model": "lmstudio-community/meta-llama-3.1-8b-instruct"
        }))
        .send()
        .await
        .expect("POST /api/v0/models/load");

    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers().get("x-lmstudio-load-time").unwrap(),
        "1.23"
    );
    let body: serde_json::Value = resp.json().await.expect("json body");
    assert_eq!(body["status"], "loaded");
}

// ── POST /api/v0/models/unload ────────────────────────────────────────────────

#[tokio::test]
async fn native_model_unload_forwarded() {
    let p = spawn_proxy().await;

    Mock::given(method("POST"))
        .and(path("/api/v0/models/unload"))
        .and(body_partial_json(json!({
            "model_instance_id": "lmstudio-community/meta-llama-3.1-8b-instruct"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "status": "unloaded"
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/v0/models/unload"))
        .json(&json!({
            "model_instance_id": "lmstudio-community/meta-llama-3.1-8b-instruct"
        }))
        .send()
        .await
        .expect("POST /api/v0/models/unload");

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.expect("json body");
    assert_eq!(body["status"], "unloaded");
}

#[tokio::test]
async fn native_model_unload_backend_error_propagated() {
    let p = spawn_proxy().await;

    Mock::given(method("POST"))
        .and(path("/api/v0/models/unload"))
        .respond_with(ResponseTemplate::new(404).set_body_json(json!({
            "error": "model instance not found"
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/v0/models/unload"))
        .json(&json!({ "model_instance_id": "ghost-model" }))
        .send()
        .await
        .expect("POST /api/v0/models/unload 404");

    assert_eq!(resp.status(), 404);
}
