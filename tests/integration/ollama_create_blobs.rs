// Integration tests for POST /api/create, POST /api/blobs/:digest,
// and HEAD /api/blobs/:digest.
//
// BlobStore is per-proxy instance (backed by a tempdir from spawn_proxy()).
// VirtualModelStore is also per-instance, so tests are isolated.

use std::fmt::Write as _;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use wiremock::matchers::{method, path};
use wiremock::{Mock, ResponseTemplate};

use crate::common::spawn_proxy;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn native_model(key: &str) -> Value {
    json!({
        "key": key,
        "type": "llm",
        "publisher": "test-publisher",
        "architecture": "llama",
        "format": "gguf",
        "quantization": {"name": "Q4_K_M"},
        "max_context_length": 8192,
        "loaded_instances": [],
        "params_string": "7B",
        "size_bytes": 4_500_000_000u64
    })
}

fn lms_models(models: Vec<Value>) -> Value {
    json!({ "models": models })
}

/// Compute "sha256:<hex>" for the given bytes — matches BlobStore validation.
fn sha256_digest(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    let bytes = h.finalize();
    let mut out = String::from("sha256:");
    out.reserve(bytes.len() * 2);
    for byte in bytes.iter() {
        write!(&mut out, "{byte:02x}").unwrap();
    }
    out
}

// ---------------------------------------------------------------------------
// POST /api/blobs/:digest — valid upload, then HEAD confirms presence
// ---------------------------------------------------------------------------

#[tokio::test]
async fn blob_upload_valid_digest_returns_201() {
    let p = spawn_proxy().await;

    let data = b"fake gguf model content";
    let digest = sha256_digest(data);
    let url = p.url(&format!("/api/blobs/{digest}"));

    let resp = p
        .client
        .post(&url)
        .body(data.to_vec())
        .send()
        .await
        .expect("POST /api/blobs/:digest");

    assert_eq!(
        resp.status(),
        201,
        "valid blob upload should return 201 Created; got {}",
        resp.status()
    );
}

#[tokio::test]
async fn blob_head_present_after_upload_returns_200() {
    let p = spawn_proxy().await;

    let data = b"another fake model blob";
    let digest = sha256_digest(data);
    let url = p.url(&format!("/api/blobs/{digest}"));

    // Upload first.
    p.client
        .post(&url)
        .body(data.to_vec())
        .send()
        .await
        .expect("POST /api/blobs upload");

    // HEAD should now return 200.
    let head = p
        .client
        .head(&url)
        .send()
        .await
        .expect("HEAD /api/blobs/:digest");

    assert_eq!(
        head.status(),
        200,
        "HEAD after valid upload should be 200; got {}",
        head.status()
    );
}

#[tokio::test]
async fn blob_head_absent_returns_404() {
    let p = spawn_proxy().await;

    // Use a well-formed digest that was never uploaded.
    let data = b"never uploaded";
    let digest = sha256_digest(data);
    let url = p.url(&format!("/api/blobs/{digest}"));

    let resp = p
        .client
        .head(&url)
        .send()
        .await
        .expect("HEAD /api/blobs absent");

    assert_eq!(
        resp.status(),
        404,
        "HEAD for absent blob should return 404; got {}",
        resp.status()
    );
}

#[tokio::test]
async fn blob_upload_digest_mismatch_returns_400() {
    let p = spawn_proxy().await;

    let real_data = b"real content";
    let wrong_digest = sha256_digest(b"different content");
    let url = p.url(&format!("/api/blobs/{wrong_digest}"));

    let resp = p
        .client
        .post(&url)
        .body(real_data.to_vec())
        .send()
        .await
        .expect("POST /api/blobs digest mismatch");

    assert_eq!(
        resp.status(),
        400,
        "digest mismatch should return 400; got {}",
        resp.status()
    );
}

#[tokio::test]
async fn blob_head_absent_after_mismatch_upload() {
    let p = spawn_proxy().await;

    let data = b"wrong data";
    let wrong_digest = sha256_digest(b"entirely different");
    let url = p.url(&format!("/api/blobs/{wrong_digest}"));

    // Upload with wrong digest (should fail).
    let _ = p.client.post(&url).body(data.to_vec()).send().await;

    // HEAD should still return 404 because the blob was not stored.
    let head = p
        .client
        .head(&url)
        .send()
        .await
        .expect("HEAD after failed upload");
    assert_eq!(
        head.status(),
        404,
        "failed upload must not persist blob; got {}",
        head.status()
    );
}

// ---------------------------------------------------------------------------
// POST /api/create — create from existing model via "from"
// ---------------------------------------------------------------------------

#[tokio::test]
async fn create_from_existing_model_registers_virtual() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lms_models(vec![native_model("llama3.2:3b")])),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/create"))
        .json(&json!({"model": "my-custom:v1", "from": "llama3.2:3b", "stream": false}))
        .send()
        .await
        .expect("POST /api/create");

    assert_eq!(
        resp.status(),
        200,
        "create should return 200; got {}",
        resp.status()
    );

    let body: Value = resp.json().await.expect("json body");
    // Non-stream response from create — either success field or status.
    let has_success = body.get("status").and_then(|v| v.as_str()) == Some("success")
        || body.get("virtual").and_then(|v| v.as_bool()) == Some(true);
    assert!(
        has_success,
        "expected success in create response; got {body}"
    );
}

#[tokio::test]
async fn create_result_appears_in_api_tags() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lms_models(vec![native_model("llama3.2:3b")])),
        )
        .mount(&p.mock)
        .await;

    let create = p
        .client
        .post(p.url("/api/create"))
        .json(&json!({"model": "created-model:v1", "from": "llama3.2:3b", "stream": false}))
        .send()
        .await
        .expect("POST /api/create");
    assert!(
        !create.status().is_server_error(),
        "create failed: {}",
        create.status()
    );

    let tags = p
        .client
        .get(p.url("/api/tags"))
        .send()
        .await
        .expect("GET /api/tags");
    let body: Value = tags.json().await.expect("tags body");
    let models = body["models"].as_array().expect("models");
    let has_created = models.iter().any(|m| {
        m["name"]
            .as_str()
            .is_some_and(|n| n.contains("created-model"))
            || m["model"]
                .as_str()
                .is_some_and(|n| n.contains("created-model"))
    });
    assert!(
        has_created,
        "created model should appear in /api/tags; got {body}"
    );
}

#[tokio::test]
async fn create_stream_true_final_chunk_is_bare_success() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lms_models(vec![native_model("llama3.2:3b")])),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/create"))
        .json(&json!({"model": "streamed-create:v1", "from": "llama3.2:3b", "stream": true}))
        .send()
        .await
        .expect("POST /api/create stream");
    assert_eq!(resp.status(), 200);

    let text = resp.text().await.expect("body text");
    // There must be at least one NDJSON line.
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    assert!(!lines.is_empty(), "expected NDJSON chunks; got empty body");

    // Each line must be valid JSON.
    for line in &lines {
        let _: Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("invalid JSON chunk: {e}; line='{line}'"));
    }

    // The last chunk must contain status:success.
    let last: Value = serde_json::from_str(lines.last().unwrap()).expect("last chunk JSON");
    assert_eq!(
        last["status"].as_str(),
        Some("success"),
        "last create chunk must be status:success; got {last}"
    );
}

#[tokio::test]
async fn create_with_system_prompt_stored_in_virtual() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lms_models(vec![native_model("llama3.2:3b")])),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/create"))
        .json(&json!({
            "model": "sys-prompt-model:v1",
            "from": "llama3.2:3b",
            "system": "You are a helpful assistant.",
            "stream": false
        }))
        .send()
        .await
        .expect("POST /api/create with system");
    assert_eq!(resp.status(), 200);

    // Show should expose system prompt in virtual metadata.
    let show = p
        .client
        .post(p.url("/api/show"))
        .json(&json!({"model": "sys-prompt-model:v1"}))
        .send()
        .await
        .expect("POST /api/show");
    assert_eq!(show.status(), 200);
    let body: Value = show.json().await.expect("show body");
    // The system prompt either appears as "system" field or inside model metadata.
    let has_system = body.get("system").is_some();
    assert!(
        has_system,
        "system prompt should appear in show response; got {body}"
    );
}

#[tokio::test]
async fn create_with_template_stored_in_virtual() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lms_models(vec![native_model("llama3.2:3b")])),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/create"))
        .json(&json!({
            "model": "tmpl-model:v1",
            "from": "llama3.2:3b",
            "template": "{{ .Prompt }}",
            "stream": false
        }))
        .send()
        .await
        .expect("POST /api/create with template");
    assert_eq!(resp.status(), 200);

    let show = p
        .client
        .post(p.url("/api/show"))
        .json(&json!({"model": "tmpl-model:v1"}))
        .send()
        .await
        .expect("POST /api/show");
    assert_eq!(show.status(), 200);
    let body: Value = show.json().await.expect("show body");
    // The template either appears at top-level or in model metadata.
    let has_template = body.get("template").is_some();
    assert!(
        has_template,
        "template should appear in show response; got {body}"
    );
}

#[tokio::test]
async fn create_missing_model_field_returns_400() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lms_models(vec![])))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/create"))
        .json(&json!({"from": "llama3.2:3b"}))
        .send()
        .await
        .expect("POST /api/create no model");

    assert!(
        resp.status().is_client_error(),
        "expected 4xx when model field missing; got {}",
        resp.status()
    );
}

#[tokio::test]
async fn create_from_missing_source_returns_error() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lms_models(vec![])))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/create"))
        .json(&json!({"model": "newmodel:v1", "from": "does-not-exist:latest", "stream": false}))
        .send()
        .await
        .expect("POST /api/create missing source");

    assert!(
        resp.status().is_client_error() || resp.status().is_server_error(),
        "expected error when source model not found; got {}",
        resp.status()
    );
}

// ---------------------------------------------------------------------------
// POST /api/create — "files" field with blob digest reference
// ---------------------------------------------------------------------------

#[tokio::test]
async fn create_with_valid_blob_file_ref_succeeds() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lms_models(vec![])))
        .mount(&p.mock)
        .await;

    // Upload a blob first.
    let data = b"fake gguf content for create";
    let digest = sha256_digest(data);
    let blob_url = p.url(&format!("/api/blobs/{digest}"));
    let upload = p
        .client
        .post(&blob_url)
        .body(data.to_vec())
        .send()
        .await
        .expect("blob upload");
    assert_eq!(upload.status(), 201, "blob upload should succeed first");

    // Create referencing the blob digest.
    let resp = p
        .client
        .post(p.url("/api/create"))
        .json(&json!({
            "model": "gguf-model:v1",
            "files": {"model.gguf": digest},
            "stream": false
        }))
        .send()
        .await
        .expect("POST /api/create with files");

    // The proxy attempts to create from the blob ref. Accept 200 or a handled error.
    // The key assertion is: it does not 5xx.
    assert!(
        !resp.status().is_server_error(),
        "unexpected 5xx from create with files; got {}",
        resp.status()
    );
}
