use super::*;
use tokio_util::sync::CancellationToken;

// ── CancellableRequest construction ─────────────────────────────────────────

#[tokio::test]
async fn token_accessor_returns_same_token() {
    let client = reqwest::Client::new();
    let token = CancellationToken::new();
    let req = CancellableRequest::new(&client, token.clone());
    // The returned token must be the same logical token (cancel one, other sees it).
    req.token().cancel();
    assert!(token.is_cancelled(), "token() must expose the shared token");
}

#[tokio::test]
async fn new_with_fresh_token_is_not_cancelled() {
    let client = reqwest::Client::new();
    let token = CancellationToken::new();
    let req = CancellableRequest::new(&client, token);
    assert!(!req.token().is_cancelled());
}

// ── handle_json_response error-path coverage (no live network) ───────────────
//
// These tests exercise the response-parsing logic by constructing synthetic
// reqwest::Response objects.  The live request-execution paths
// (make_request, make_raw_request) require a running HTTP server and are
// therefore covered by integration tests only.

fn synthetic_response(status: u16, body: &str) -> reqwest::Response {
    reqwest::Response::from(
        http::Response::builder()
            .status(status)
            .body(body.to_string())
            .expect("build synthetic response"),
    )
}

#[tokio::test]
async fn non_json_error_body_preserves_upstream_status() {
    // A 429 with a non-JSON body must keep its status, not collapse to 500
    // (matches the streaming path's reject_pre_stream_error).
    let resp = synthetic_response(429, "Too Many Requests — plain text, not JSON");
    let err = handle_json_response(resp, CancellationToken::new())
        .await
        .expect_err("a 429 must surface as an error");
    assert_eq!(
        err.status_code, 429,
        "a non-JSON error body must preserve the upstream status"
    );
}

#[tokio::test]
async fn non_json_success_body_stays_500() {
    // Garbage on a 2xx is a genuine backend fault → still 500.
    let resp = synthetic_response(200, "definitely not json");
    let err = handle_json_response(resp, CancellationToken::new())
        .await
        .expect_err("garbage on a 200 is a real 500");
    assert_eq!(err.status_code, 500);
}

#[tokio::test]
async fn json_error_body_keeps_status_and_message() {
    let resp = synthetic_response(400, r#"{"error":{"message":"bad request from backend"}}"#);
    let err = handle_json_response(resp, CancellationToken::new())
        .await
        .expect_err("400 error");
    assert_eq!(err.status_code, 400);
    assert!(
        err.message.contains("bad request from backend"),
        "message must carry the backend detail; got {}",
        err.message
    );
}
