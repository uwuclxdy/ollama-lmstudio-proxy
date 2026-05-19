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
