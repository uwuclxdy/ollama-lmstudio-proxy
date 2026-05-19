use serde_json::json;
use tokio::sync::mpsc;

use super::*;

// ════════════════════════════════════════════════════════════════════════════
// is_streaming_request
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn stream_true_is_streaming() {
    assert!(is_streaming_request(&json!({"stream": true})));
}

#[test]
fn stream_false_is_not_streaming() {
    assert!(!is_streaming_request(&json!({"stream": false})));
}

#[test]
fn stream_field_absent_defaults_to_false() {
    assert!(!is_streaming_request(&json!({"model": "llama3"})));
}

#[test]
fn stream_null_is_not_streaming() {
    assert!(!is_streaming_request(&json!({"stream": null})));
}

#[test]
fn stream_string_true_is_not_streaming() {
    // Only bool `true` counts; "true" as string must not be streaming
    assert!(!is_streaming_request(&json!({"stream": "true"})));
}

#[test]
fn stream_number_one_is_not_streaming() {
    assert!(!is_streaming_request(&json!({"stream": 1})));
}

#[test]
fn empty_object_is_not_streaming() {
    assert!(!is_streaming_request(&json!({})));
}

// ════════════════════════════════════════════════════════════════════════════
// create_streaming_response — response construction
// ════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn ndjson_streaming_response_returns_ok() {
    let (tx, rx) = mpsc::unbounded_channel::<Result<bytes::Bytes, std::io::Error>>();
    drop(tx); // close immediately — we only care about construction
    let result = create_streaming_response(rx, StreamContentType::Ndjson);
    assert!(result.is_ok(), "NDJSON response construction must succeed");
}

#[tokio::test]
async fn sse_streaming_response_returns_ok() {
    let (tx, rx) = mpsc::unbounded_channel::<Result<bytes::Bytes, std::io::Error>>();
    drop(tx);
    let result = create_streaming_response(rx, StreamContentType::Sse);
    assert!(result.is_ok(), "SSE response construction must succeed");
}

#[tokio::test]
async fn ndjson_response_has_correct_content_type() {
    let (tx, rx) = mpsc::unbounded_channel::<Result<bytes::Bytes, std::io::Error>>();
    drop(tx);
    let response = create_streaming_response(rx, StreamContentType::Ndjson).unwrap();
    let ct = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        ct.contains("ndjson"),
        "content-type must contain ndjson; got {ct:?}"
    );
}

#[tokio::test]
async fn sse_response_has_event_stream_content_type() {
    let (tx, rx) = mpsc::unbounded_channel::<Result<bytes::Bytes, std::io::Error>>();
    drop(tx);
    let response = create_streaming_response(rx, StreamContentType::Sse).unwrap();
    let ct = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        ct.contains("event-stream"),
        "content-type must be text/event-stream; got {ct:?}"
    );
}

#[tokio::test]
async fn streaming_response_has_cors_header() {
    let (tx, rx) = mpsc::unbounded_channel::<Result<bytes::Bytes, std::io::Error>>();
    drop(tx);
    let response = create_streaming_response(rx, StreamContentType::Ndjson).unwrap();
    let origin = response
        .headers()
        .get("access-control-allow-origin")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert_eq!(origin, "*", "CORS origin header must be *");
}

#[tokio::test]
async fn streaming_response_has_cache_control_no_cache() {
    let (tx, rx) = mpsc::unbounded_channel::<Result<bytes::Bytes, std::io::Error>>();
    drop(tx);
    let response = create_streaming_response(rx, StreamContentType::Ndjson).unwrap();
    let cc = response
        .headers()
        .get("cache-control")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert_eq!(cc, "no-cache", "cache-control must be no-cache");
}

#[tokio::test]
async fn streaming_response_has_keep_alive_connection() {
    let (tx, rx) = mpsc::unbounded_channel::<Result<bytes::Bytes, std::io::Error>>();
    drop(tx);
    let response = create_streaming_response(rx, StreamContentType::Ndjson).unwrap();
    let conn = response
        .headers()
        .get("connection")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert_eq!(conn, "keep-alive", "connection header must be keep-alive");
}

#[tokio::test]
async fn streaming_response_status_200() {
    let (tx, rx) = mpsc::unbounded_channel::<Result<bytes::Bytes, std::io::Error>>();
    drop(tx);
    let response = create_streaming_response(rx, StreamContentType::Ndjson).unwrap();
    assert_eq!(response.status().as_u16(), 200);
}

// ════════════════════════════════════════════════════════════════════════════
// create_ndjson_stream_response
// ════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn ndjson_stream_response_convenience_returns_ok() {
    let (tx, rx) = mpsc::unbounded_channel::<Result<bytes::Bytes, std::io::Error>>();
    drop(tx);
    let result = create_ndjson_stream_response(rx, "test error message");
    assert!(result.is_ok());
}

#[tokio::test]
async fn ndjson_stream_response_content_type_matches_direct_call() {
    let (tx, rx) = mpsc::unbounded_channel::<Result<bytes::Bytes, std::io::Error>>();
    drop(tx);
    let (tx2, rx2) = mpsc::unbounded_channel::<Result<bytes::Bytes, std::io::Error>>();
    drop(tx2);

    let via_convenience = create_ndjson_stream_response(rx, "err").unwrap();
    let via_direct = create_streaming_response(rx2, StreamContentType::Ndjson).unwrap();

    let ct1 = via_convenience
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let ct2 = via_direct
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert_eq!(
        ct1, ct2,
        "convenience wrapper must produce same content-type"
    );
}

// ════════════════════════════════════════════════════════════════════════════
// StreamContentType enum
// ════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn both_content_type_variants_produce_different_content_types() {
    let (tx1, rx1) = mpsc::unbounded_channel::<Result<bytes::Bytes, std::io::Error>>();
    let (tx2, rx2) = mpsc::unbounded_channel::<Result<bytes::Bytes, std::io::Error>>();
    drop(tx1);
    drop(tx2);

    let ndjson = create_streaming_response(rx1, StreamContentType::Ndjson).unwrap();
    let sse = create_streaming_response(rx2, StreamContentType::Sse).unwrap();

    let ct_ndjson = ndjson
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let ct_sse = sse
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    assert_ne!(
        ct_ndjson, ct_sse,
        "Ndjson and Sse variants must produce different content-type headers"
    );
}
