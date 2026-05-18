//! /api/pull NDJSON status stream — terminal chunk and non-stream response shape.
//!
//! Reference (api_docs/ollama.md lines 1582-1626):
//!   - In-progress chunk: {"status":"pulling <digest>", "digest":..., "total":..., "completed":...}
//!   - Final chunk:       {"status":"success"}   (no extras)
//!   - Non-stream success: same — single {"status":"success"} object
//!
//! Many Ollama clients (Open WebUI, the ollama CLI) match the literal
//! {"status":"success"} sentinel to know when to stop reading. Extra keys break
//! strict equality matching and cause clients to wait forever.

#[path = "../src/constants.rs"]
#[allow(dead_code)]
mod constants;

#[path = "../src/error.rs"]
#[allow(dead_code)]
mod error;

#[path = "../src/handlers/ollama/download_status.rs"]
#[allow(dead_code)]
mod download_status;

use download_status::LmStudioDownloadStatus;
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
    // status need not be exactly "downloading" — Ollama uses "pulling <digest>",
    // but it must be present and non-success.
    assert!(obj.contains_key("status"));
    assert_ne!(obj.get("status"), Some(&json!("success")));
    // Progress numbers must be forwarded.
    assert_eq!(obj.get("total"), Some(&json!(2_142_590_208u64)));
    assert_eq!(obj.get("completed"), Some(&json!(241_970u64)));
}
