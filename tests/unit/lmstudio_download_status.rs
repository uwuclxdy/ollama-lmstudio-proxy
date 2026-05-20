use super::*;
use serde_json::json;

fn lm_status(status: &str) -> LmStudioDownloadStatus {
    serde_json::from_value(json!({
        "job_id": "job123",
        "status": status,
        "total_size_bytes": 2_142_590_208u64,
        "downloaded_bytes": 2_142_590_208u64,
        "completed_at": "2026-05-18T10:00:00Z"
    }))
    .unwrap()
}

#[test]
fn terminal_chunk_is_bare_status_success() {
    let s = lm_status("completed");
    let chunk = s.to_chunk("llama3.2");
    let obj = chunk.as_object().expect("chunk must be an object");
    assert_eq!(obj.get("status"), Some(&json!("success")));
    // Strict-equality clients want EXACTLY {"status":"success"} on terminal success.
    let extras: Vec<_> = obj.keys().filter(|k| *k != "status").collect();
    assert!(
        extras.is_empty(),
        "terminal success chunk must contain only 'status', got extras: {extras:?}"
    );
}

#[test]
fn already_downloaded_terminal_is_also_bare() {
    let s = lm_status("already_downloaded");
    let chunk = s.to_chunk("llama3.2");
    let obj = chunk.as_object().unwrap();
    assert_eq!(obj.get("status"), Some(&json!("success")));
    let extras: Vec<_> = obj.keys().filter(|k| *k != "status").collect();
    assert!(extras.is_empty(), "got extras: {extras:?}");
}

#[test]
fn non_stream_success_returns_bare_status() {
    let s = lm_status("completed");
    let value = s
        .into_final_response("llama3.2")
        .expect("completed should map to success");
    let obj = value.as_object().expect("final response must be an object");
    assert_eq!(obj.get("status"), Some(&json!("success")));
    let extras: Vec<_> = obj.keys().filter(|k| *k != "status").collect();
    assert!(
        extras.is_empty(),
        "non-stream success must be {{\"status\":\"success\"}} only, got extras: {extras:?}"
    );
}

#[test]
fn in_progress_chunk_retains_progress_fields() {
    let s: LmStudioDownloadStatus = serde_json::from_value(json!({
        "job_id": "job123",
        "status": "downloading",
        "total_size_bytes": 2_142_590_208u64,
        "downloaded_bytes": 241_970u64
    }))
    .unwrap();
    let chunk = s.to_chunk("llama3.2");
    let obj = chunk.as_object().unwrap();
    // status is forwarded from LM Studio — not "success", not empty.
    assert!(obj.contains_key("status"));
    assert_ne!(obj.get("status"), Some(&json!("success")));
    // Progress numbers must be forwarded.
    assert_eq!(obj.get("total"), Some(&json!(2_142_590_208u64)));
    assert_eq!(obj.get("completed"), Some(&json!(241_970u64)));
}

// GAP B: StatusEvent.digest is optional in the Ollama OpenAPI schema (no
// `required` constraint). LM Studio's download status does not return a
// content digest, so `to_chunk` deliberately omits the field. This test
// confirms that conformant omission.
#[test]
fn in_progress_chunk_omits_digest() {
    let s: LmStudioDownloadStatus = serde_json::from_value(json!({
        "job_id": "job456",
        "status": "downloading",
        "total_size_bytes": 1_000_000u64,
        "downloaded_bytes": 500_000u64
    }))
    .unwrap();
    let chunk = s.to_chunk("llama3.2");
    let obj = chunk.as_object().expect("chunk must be an object");
    assert!(
        !obj.contains_key("digest"),
        "in-progress chunk must not emit `digest` \
         (LM Studio returns no content digest; omission is doc-conformant); got {chunk}"
    );
}

fn lm_status_value(extra: serde_json::Value) -> LmStudioDownloadStatus {
    let mut base = json!({
        "job_id": "job_abc",
        "status": "downloading"
    });
    if let Some(obj) = extra.as_object() {
        for (k, v) in obj {
            base[k] = v.clone();
        }
    }
    serde_json::from_value(base).expect("test fixture must deserialize")
}

#[test]
fn download_status_is_terminal_matches_known_states() {
    assert!(lm_status_value(json!({"status": "completed"})).is_terminal());
    assert!(lm_status_value(json!({"status": "already_downloaded"})).is_terminal());
    assert!(lm_status_value(json!({"status": "failed"})).is_terminal());
    assert!(!lm_status_value(json!({"status": "downloading"})).is_terminal());
    assert!(!lm_status_value(json!({"status": "queued"})).is_terminal());
}

#[test]
fn download_status_failed_into_final_response_propagates_error_message() {
    let s = lm_status_value(json!({
        "status": "failed",
        "error": "disk full"
    }));
    let result = s.into_final_response("llama3.2");
    let err = result.expect_err("failed status must produce Err");
    assert!(
        err.message.contains("disk full"),
        "expected LM Studio error preserved, got: {}",
        err.message
    );
}

#[test]
fn download_status_failed_without_error_field_uses_fallback_message() {
    let s = lm_status_value(json!({ "status": "failed" }));
    let err = s
        .into_final_response("llama3.2")
        .expect_err("failed must be Err");
    assert!(!err.message.is_empty());
    assert_ne!(err.message, "success");
}

#[test]
fn download_status_unknown_status_into_final_response_is_err() {
    let s = lm_status_value(json!({ "status": "paused" }));
    let err = s
        .into_final_response("llama3.2")
        .expect_err("unknown status must be Err");
    assert!(
        err.message.contains("paused") || err.message.to_lowercase().contains("unexpected"),
        "error should mention the unexpected status, got: {}",
        err.message
    );
}

#[test]
fn download_status_failed_chunk_includes_error_and_is_not_success_sentinel() {
    let s = lm_status_value(json!({
        "status": "failed",
        "error": "checksum mismatch"
    }));
    let chunk = s.to_chunk("llama3.2");
    let obj = chunk.as_object().expect("chunk must be an object");
    assert_eq!(
        obj.get("error"),
        Some(&json!("checksum mismatch")),
        "failed chunk must include the upstream error string"
    );
    assert_ne!(
        obj.get("status"),
        Some(&json!("success")),
        "failed terminal chunk must NOT be shaped like the bare success sentinel"
    );
    assert!(
        obj.len() > 1,
        "failed chunk must carry more than just `status`, got: {chunk}"
    );
}
