// Integration tests for POST /api/pull, DELETE /api/delete, POST /api/copy.
//
// Pull translates to POST /api/v1/models/download on LM Studio with status
// polling on GET /api/v1/models/download/status.
//
// Delete and copy operate on the in-process VirtualModelStore.

use serde_json::{Value, json};
use wiremock::matchers::{method, path, path_regex};
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

/// Returns a completed LM Studio download status response.
fn lms_download_completed(job_id: &str) -> Value {
    json!({
        "job_id": job_id,
        "status": "completed",
        "total_size_bytes": 4_000_000_000u64,
        "downloaded_bytes": 4_000_000_000u64,
        "completed_at": "2026-01-01T00:00:00Z"
    })
}

/// Returns an in-progress LM Studio download status response.
fn lms_download_downloading(job_id: &str, done: u64, total: u64) -> Value {
    json!({
        "job_id": job_id,
        "status": "downloading",
        "total_size_bytes": total,
        "downloaded_bytes": done,
        "bytes_per_second": 1_024_000.0,
        "estimated_completion": "2026-01-01T00:05:00Z",
        "started_at": "2026-01-01T00:00:00Z"
    })
}

fn lms_download_paused(job_id: &str, done: u64, total: u64) -> Value {
    json!({
        "job_id": job_id,
        "status": "paused",
        "total_size_bytes": total,
        "downloaded_bytes": done,
        "started_at": "2026-01-01T00:00:00Z"
    })
}

fn lms_download_already(job_id: &str) -> Value {
    json!({
        "job_id": job_id,
        "status": "already_downloaded"
    })
}

// ---------------------------------------------------------------------------
// POST /api/pull — stream:false, success
// ---------------------------------------------------------------------------

#[tokio::test]
async fn pull_stream_false_returns_status_success() {
    let p = spawn_proxy().await;

    // Initiate download → immediate "completed".
    Mock::given(method("POST"))
        .and(path("/api/v1/models/download"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lms_download_completed("job1")))
        .mount(&p.mock)
        .await;

    // Model lookup for name resolution.
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lms_models(vec![native_model("llama3.2:3b")])),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/pull"))
        .json(&json!({"model": "llama3.2:3b", "stream": false}))
        .send()
        .await
        .expect("POST /api/pull");
    assert_eq!(resp.status(), 200);

    let body: Value = resp.json().await.expect("json body");
    assert_eq!(
        body["status"].as_str(),
        Some("success"),
        "expected status:success for non-stream pull; got {body}"
    );
}

#[tokio::test]
async fn pull_already_downloaded_returns_status_success() {
    let p = spawn_proxy().await;

    Mock::given(method("POST"))
        .and(path("/api/v1/models/download"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lms_download_already("job2")))
        .mount(&p.mock)
        .await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lms_models(vec![native_model("llama3.2:3b")])),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/pull"))
        .json(&json!({"model": "llama3.2:3b", "stream": false}))
        .send()
        .await
        .expect("POST /api/pull already_downloaded");
    assert_eq!(resp.status(), 200);

    let body: Value = resp.json().await.expect("json body");
    assert_eq!(
        body["status"].as_str(),
        Some("success"),
        "already_downloaded should map to success; got {body}"
    );
}

// ---------------------------------------------------------------------------
// POST /api/pull — stream:true, terminal chunk is bare {"status":"success"}
// ---------------------------------------------------------------------------

#[tokio::test]
async fn pull_stream_true_terminal_chunk_is_bare_success() {
    let p = spawn_proxy().await;

    // Initiate: in-progress.
    Mock::given(method("POST"))
        .and(path("/api/v1/models/download"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lms_download_downloading(
                "job3",
                0,
                4_000_000_000,
            )),
        )
        .mount(&p.mock)
        .await;

    // Status poll: completed on first poll.
    // The proxy appends the job_id: GET /api/v1/models/download/status/{job_id}.
    Mock::given(method("GET"))
        .and(path_regex(r"^/api/v1/models/download/status/.*"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lms_download_completed("job3")))
        .mount(&p.mock)
        .await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lms_models(vec![native_model("llama3.2:3b")])),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/pull"))
        .json(&json!({"model": "llama3.2:3b", "stream": true}))
        .send()
        .await
        .expect("POST /api/pull stream");
    assert_eq!(resp.status(), 200);

    let body_bytes = resp.bytes().await.expect("body bytes");
    let text = String::from_utf8_lossy(&body_bytes);

    // NDJSON: find the last non-empty line.
    let last_line = text
        .lines()
        .rfind(|l| !l.trim().is_empty())
        .expect("at least one NDJSON line");

    let last_chunk: Value = serde_json::from_str(last_line)
        .unwrap_or_else(|e| panic!("last NDJSON line is not valid JSON: {e}; line='{last_line}'"));

    // The terminal chunk must be exactly {"status":"success"} — no extra keys.
    let obj = last_chunk.as_object().expect("chunk must be object");
    assert_eq!(
        obj.get("status").and_then(|v| v.as_str()),
        Some("success"),
        "terminal chunk status must be 'success'; got {last_chunk}"
    );
    assert_eq!(
        obj.len(),
        1,
        "terminal chunk must have ONLY 'status' key (got extra keys); full chunk: {last_chunk}"
    );
}

#[tokio::test]
async fn pull_stream_true_in_progress_chunk_matches_status_event_schema() {
    let p = spawn_proxy().await;

    Mock::given(method("POST"))
        .and(path("/api/v1/models/download"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lms_download_downloading(
                "job-prog",
                1_000_000,
                4_000_000_000,
            )),
        )
        .mount(&p.mock)
        .await;

    Mock::given(method("GET"))
        .and(path_regex(r"^/api/v1/models/download/status/.*"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lms_download_completed("job-prog")))
        .mount(&p.mock)
        .await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lms_models(vec![native_model("llama3.2:3b")])),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/pull"))
        .json(&json!({"model": "llama3.2:3b", "stream": true}))
        .send()
        .await
        .expect("POST /api/pull stream");
    assert_eq!(resp.status(), 200);

    let text = resp.text().await.expect("body text");
    let mut chunks: Vec<Value> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            serde_json::from_str(l).unwrap_or_else(|e| panic!("invalid ndjson line '{l}': {e}"))
        })
        .collect();

    let terminal = chunks.pop().expect("at least one chunk");
    assert_eq!(
        terminal,
        json!({"status": "success"}),
        "terminal chunk must be bare success; got {terminal}"
    );

    // At least one in-progress chunk preceded it.
    let in_progress = chunks
        .iter()
        .find(|c| c["status"].as_str() != Some("success"))
        .expect("expected at least one in-progress chunk before terminal success");

    assert_eq!(
        in_progress["total"].as_u64(),
        Some(4_000_000_000),
        "in-progress chunk must carry `total` from LM Studio; got {in_progress}"
    );
    assert_eq!(
        in_progress["completed"].as_u64(),
        Some(1_000_000),
        "in-progress chunk must carry `completed` (downloaded_bytes); got {in_progress}"
    );

    let obj = in_progress
        .as_object()
        .expect("in-progress chunk must be object");
    let internal_keys = [
        "model",
        "detail",
        "job_id",
        "bytes_per_second",
        "estimated_completion",
        "started_at",
        "completed_at",
        "error",
    ];
    for key in internal_keys {
        assert!(
            !obj.contains_key(key),
            "in-progress chunk must not emit internal key `{key}`; got {in_progress}"
        );
    }

    assert!(
        obj.keys()
            .all(|key| matches!(key.as_str(), "status" | "digest" | "total" | "completed")),
        "in-progress chunk must match Ollama StatusEvent schema; got {in_progress}"
    );
}

#[tokio::test]
async fn pull_stream_false_paused_polls_until_completed() {
    let p = spawn_proxy().await;

    Mock::given(method("POST"))
        .and(path("/api/v1/models/download"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lms_download_paused(
                "job-paused-nonstream",
                1_000_000,
                4_000_000_000,
            )),
        )
        .mount(&p.mock)
        .await;

    Mock::given(method("GET"))
        .and(path_regex(r"^/api/v1/models/download/status/.*"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(lms_download_completed("job-paused-nonstream")),
        )
        .expect(1)
        .mount(&p.mock)
        .await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lms_models(vec![native_model("llama3.2:3b")])),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/pull"))
        .json(&json!({"model": "llama3.2:3b", "stream": false}))
        .send()
        .await
        .expect("POST /api/pull stream:false paused");
    assert_eq!(resp.status(), 200);

    let body: Value = resp.json().await.expect("json body");
    assert_eq!(body, json!({"status": "success"}));
}

#[tokio::test]
async fn pull_stream_true_paused_polls_until_completed() {
    let p = spawn_proxy().await;

    Mock::given(method("POST"))
        .and(path("/api/v1/models/download"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lms_download_paused(
                "job-paused-stream",
                1_000_000,
                4_000_000_000,
            )),
        )
        .mount(&p.mock)
        .await;

    Mock::given(method("GET"))
        .and(path_regex(r"^/api/v1/models/download/status/.*"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lms_download_completed("job-paused-stream")),
        )
        .expect(1)
        .mount(&p.mock)
        .await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lms_models(vec![native_model("llama3.2:3b")])),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/pull"))
        .json(&json!({"model": "llama3.2:3b", "stream": true}))
        .send()
        .await
        .expect("POST /api/pull stream:true paused");
    assert_eq!(resp.status(), 200);

    let text = resp.text().await.expect("body text");
    let chunks: Vec<Value> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("invalid ndjson line '{line}': {e}"))
        })
        .collect();

    assert!(
        chunks
            .iter()
            .any(|chunk| chunk["status"].as_str() == Some("paused")),
        "stream must emit the paused progress chunk before completion; got {chunks:?}"
    );
    assert_eq!(
        chunks.last(),
        Some(&json!({"status": "success"})),
        "stream must finish after paused status; got {chunks:?}"
    );
}

#[tokio::test]
async fn pull_stream_true_already_downloaded_bare_success() {
    let p = spawn_proxy().await;

    Mock::given(method("POST"))
        .and(path("/api/v1/models/download"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lms_download_already("job4")))
        .mount(&p.mock)
        .await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lms_models(vec![native_model("llama3.2:3b")])),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/pull"))
        .json(&json!({"model": "llama3.2:3b", "stream": true}))
        .send()
        .await
        .expect("POST /api/pull stream already_downloaded");
    assert_eq!(resp.status(), 200);

    let text = resp.text().await.expect("body text");
    let last_line = text
        .lines()
        .rfind(|l| !l.trim().is_empty())
        .expect("last line");
    let last: Value = serde_json::from_str(last_line).expect("valid JSON");
    assert_eq!(
        last,
        json!({"status": "success"}),
        "terminal chunk must be bare; got {last}"
    );
}

// ---------------------------------------------------------------------------
// POST /api/pull — model name forwarded to LM Studio
// ---------------------------------------------------------------------------

#[tokio::test]
async fn pull_model_name_forwarded_to_lmstudio() {
    let p = spawn_proxy().await;

    // The proxy should POST to /api/v1/models/download with the model name.
    Mock::given(method("POST"))
        .and(path("/api/v1/models/download"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lms_download_completed("job5")))
        .expect(1)
        .mount(&p.mock)
        .await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lms_models(vec![native_model("phi3:mini")])),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/pull"))
        .json(&json!({"model": "phi3:mini", "stream": false}))
        .send()
        .await
        .expect("POST /api/pull phi3:mini");
    assert_eq!(resp.status(), 200);
    // wiremock asserts the expectation on drop.
}

// ---------------------------------------------------------------------------
// POST /api/pull — insecure flag is accepted (warning logged, not rejected)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn pull_insecure_flag_is_accepted_with_success() {
    // `insecure: true` has no LM Studio equivalent. The proxy logs a warning
    // and proceeds normally — it must not reject the request.
    let p = spawn_proxy().await;

    Mock::given(method("POST"))
        .and(path("/api/v1/models/download"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lms_download_completed("job-ins")))
        .mount(&p.mock)
        .await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lms_models(vec![native_model("llama3.2:3b")])),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/pull"))
        .json(&json!({"model": "llama3.2:3b", "stream": false, "insecure": true}))
        .send()
        .await
        .expect("POST /api/pull insecure");
    assert_eq!(
        resp.status(),
        200,
        "insecure flag must not cause rejection; got {}",
        resp.status()
    );

    let body: Value = resp.json().await.expect("json body");
    assert_eq!(
        body["status"].as_str(),
        Some("success"),
        "pull with insecure flag should still return success; got {body}"
    );
}

// ---------------------------------------------------------------------------
// POST /api/pull — missing model field → 400
// ---------------------------------------------------------------------------

#[tokio::test]
async fn pull_missing_model_field_returns_400() {
    let p = spawn_proxy().await;

    let resp = p
        .client
        .post(p.url("/api/pull"))
        .json(&json!({}))
        .send()
        .await
        .expect("POST /api/pull empty body");

    assert!(
        resp.status().is_client_error(),
        "expected 4xx for missing model; got {}",
        resp.status()
    );
}

// ---------------------------------------------------------------------------
// DELETE /api/delete
// ---------------------------------------------------------------------------

#[tokio::test]
async fn delete_virtual_model_succeeds() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lms_models(vec![native_model("llama3.2:3b")])),
        )
        .mount(&p.mock)
        .await;

    // Create a virtual model first.
    let copy = p
        .client
        .post(p.url("/api/copy"))
        .json(&json!({"source": "llama3.2:3b", "destination": "to-delete:v1"}))
        .send()
        .await
        .expect("POST /api/copy");
    assert!(
        !copy.status().is_server_error(),
        "copy failed: {}",
        copy.status()
    );

    // Delete it.
    let del = p
        .client
        .delete(p.url("/api/delete"))
        .json(&json!({"model": "to-delete:v1"}))
        .send()
        .await
        .expect("DELETE /api/delete");
    assert_eq!(del.status(), 200, "delete should return 200");
    let bytes = del.bytes().await.expect("delete response bytes");
    assert!(
        bytes.is_empty(),
        "delete 200 body must be empty per spec; got {bytes:?}"
    );
}

#[tokio::test]
async fn delete_unknown_model_returns_404() {
    // Non-virtual (native LM Studio) models are not deletable through this proxy;
    // any unrecognised name returns 404 — this is an architectural limit.
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lms_models(vec![])))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .delete(p.url("/api/delete"))
        .json(&json!({"model": "ghost:latest"}))
        .send()
        .await
        .expect("DELETE /api/delete unknown");

    assert_eq!(
        resp.status(),
        404,
        "expected 404 for non-existent model; got {}",
        resp.status()
    );
}

#[tokio::test]
async fn delete_virtual_model_removed_from_tags() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lms_models(vec![native_model("llama3.2:3b")])),
        )
        .mount(&p.mock)
        .await;

    let copy = p
        .client
        .post(p.url("/api/copy"))
        .json(&json!({"source": "llama3.2:3b", "destination": "ephemeral:v2"}))
        .send()
        .await
        .expect("POST /api/copy");
    assert!(
        !copy.status().is_server_error(),
        "copy failed: {}",
        copy.status()
    );

    // Delete.
    let del = p
        .client
        .delete(p.url("/api/delete"))
        .json(&json!({"model": "ephemeral:v2"}))
        .send()
        .await
        .expect("DELETE /api/delete");
    assert_eq!(del.status(), 200);

    // The model should no longer appear in /api/tags virtual overlay.
    let tags = p
        .client
        .get(p.url("/api/tags"))
        .send()
        .await
        .expect("GET /api/tags");
    let body: Value = tags.json().await.expect("tags body");
    let models = body["models"].as_array().expect("models");
    let still_present = models.iter().any(|m| {
        m["name"].as_str().is_some_and(|n| n.contains("ephemeral"))
            || m["model"].as_str().is_some_and(|n| n.contains("ephemeral"))
    });
    assert!(
        !still_present,
        "deleted model should not appear in /api/tags; got {body}"
    );
}

#[tokio::test]
async fn delete_missing_model_field_returns_400() {
    let p = spawn_proxy().await;

    let resp = p
        .client
        .delete(p.url("/api/delete"))
        .json(&json!({}))
        .send()
        .await
        .expect("DELETE /api/delete empty");

    assert!(
        resp.status().is_client_error(),
        "expected 4xx for missing model field; got {}",
        resp.status()
    );
}

#[tokio::test]
async fn delete_success_returns_empty_body() {
    // Spec: 200 response declares no content block — body must be empty.
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lms_models(vec![native_model("llama3.2:3b")])),
        )
        .mount(&p.mock)
        .await;

    let copy = p
        .client
        .post(p.url("/api/copy"))
        .json(&json!({"source": "llama3.2:3b", "destination": "empty-body-check:v1"}))
        .send()
        .await
        .expect("POST /api/copy");
    assert!(
        !copy.status().is_server_error(),
        "copy failed: {}",
        copy.status()
    );

    let del = p
        .client
        .delete(p.url("/api/delete"))
        .json(&json!({"model": "empty-body-check:v1"}))
        .send()
        .await
        .expect("DELETE /api/delete");

    assert_eq!(del.status(), 200, "delete should return 200");
    let bytes = del.bytes().await.expect("delete response bytes");
    assert!(
        bytes.is_empty(),
        "delete 200 body must be empty; got {bytes:?}"
    );
}

// ---------------------------------------------------------------------------
// POST /api/copy
// ---------------------------------------------------------------------------

#[tokio::test]
async fn copy_returns_bare_status_success() {
    // The Ollama spec defines no body for the 200 response. The proxy emits
    // {"status":"success"} and must not include proxy-internal fields such as
    // `model`, `virtual`, `source_model`, `target_model_id`, `created_at`,
    // or `updated_at`.
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
        .post(p.url("/api/copy"))
        .json(&json!({"source": "llama3.2:3b", "destination": "bare-copy:v1"}))
        .send()
        .await
        .expect("POST /api/copy");
    assert_eq!(
        resp.status(),
        200,
        "copy should return 200; got {}",
        resp.status()
    );

    let body: Value = resp.json().await.expect("json body");
    assert_eq!(
        body,
        json!({"status": "success"}),
        "copy response must be bare {{\"status\":\"success\"}}; got {body}"
    );

    // Proxy-internal fields must be absent.
    for key in &[
        "model",
        "virtual",
        "source_model",
        "target_model_id",
        "created_at",
        "updated_at",
    ] {
        assert!(
            body.get(key).is_none(),
            "copy response must not contain proxy-internal field '{key}'; got {body}"
        );
    }
}

#[tokio::test]
async fn copy_creates_virtual_alias_in_tags() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lms_models(vec![native_model("llama3.2:3b")])),
        )
        .mount(&p.mock)
        .await;

    let copy = p
        .client
        .post(p.url("/api/copy"))
        .json(&json!({"source": "llama3.2:3b", "destination": "copy-target:v1"}))
        .send()
        .await
        .expect("POST /api/copy");
    assert_eq!(
        copy.status(),
        200,
        "copy should return 200; got {}",
        copy.status()
    );

    let tags = p
        .client
        .get(p.url("/api/tags"))
        .send()
        .await
        .expect("GET /api/tags");
    let body: Value = tags.json().await.expect("tags body");
    let models = body["models"].as_array().expect("models");
    let has_copy = models.iter().any(|m| {
        m["name"]
            .as_str()
            .is_some_and(|n| n.contains("copy-target"))
            || m["model"]
                .as_str()
                .is_some_and(|n| n.contains("copy-target"))
    });
    assert!(
        has_copy,
        "copy destination should appear in /api/tags; got {body}"
    );
}

#[tokio::test]
async fn copy_missing_source_returns_error() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lms_models(vec![])))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/copy"))
        .json(&json!({"source": "nonexistent:latest", "destination": "dest:v1"}))
        .send()
        .await
        .expect("POST /api/copy missing source");

    assert!(
        resp.status().is_client_error() || resp.status().is_server_error(),
        "expected error for missing source; got {}",
        resp.status()
    );
}

#[tokio::test]
async fn copy_missing_fields_returns_400() {
    // The Ollama spec declares no 400 response, but the proxy returns 400 for
    // missing required fields as a pragmatic guard against silent failures.
    let p = spawn_proxy().await;

    let resp = p
        .client
        .post(p.url("/api/copy"))
        .json(&json!({"source": "llama3.2:3b"}))
        .send()
        .await
        .expect("POST /api/copy no destination");

    assert_eq!(
        resp.status(),
        400,
        "expected 400 when destination missing; got {}",
        resp.status()
    );
}

#[tokio::test]
async fn copy_creates_virtual_model_showable_via_show() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lms_models(vec![native_model("mistral:7b")])),
        )
        .mount(&p.mock)
        .await;

    let copy = p
        .client
        .post(p.url("/api/copy"))
        .json(&json!({"source": "mistral:7b", "destination": "showable-copy:v1"}))
        .send()
        .await
        .expect("POST /api/copy");
    assert!(
        !copy.status().is_server_error(),
        "copy failed: {}",
        copy.status()
    );

    let show = p
        .client
        .post(p.url("/api/show"))
        .json(&json!({"model": "showable-copy:v1"}))
        .send()
        .await
        .expect("POST /api/show copied model");
    assert_eq!(
        show.status(),
        200,
        "show on copied alias should succeed; got {}",
        show.status()
    );
}
