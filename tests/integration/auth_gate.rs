// Integration tests for the inbound `--api-key` / `OLLAMA_API_KEY` Bearer gate
// (`src/proxy/auth.rs::api_key_gate`).
//
// When configured, every inbound request must carry `Authorization: Bearer
// <key>` or receive a 401 with a JSON `{"error":"unauthorized"}` body. When
// unset (the default), the proxy is fully open and inspects no auth header.
// `/api/version` is served locally with no LM Studio mock, so these tests are
// fully deterministic.

use serde_json::Value;

use crate::common::{spawn_proxy, spawn_proxy_with_api_key};

const KEY: &str = "s3cret-test-key";

// ── gate open by default ────────────────────────────────────────────────────

#[tokio::test]
async fn no_key_configured_is_open_without_header() {
    let p = spawn_proxy().await;
    let resp = p
        .client
        .get(p.url("/api/version"))
        .send()
        .await
        .expect("GET /api/version");
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("JSON body");
    assert!(
        body.get("version")
            .map(serde_json::Value::is_string)
            .unwrap_or(false),
        "open proxy must serve /api/version: {body}"
    );
}

// ── gate closed: missing / wrong / correct bearer ───────────────────────────

#[tokio::test]
async fn key_set_rejects_request_with_no_authorization() {
    let p = spawn_proxy_with_api_key(KEY).await;

    let resp = p
        .client
        .get(p.url("/api/version"))
        .send()
        .await
        .expect("GET /api/version");
    assert_eq!(resp.status(), 401);

    let body: Value = resp.json().await.expect("JSON error body");
    assert!(
        body.get("error")
            .and_then(|v| v.as_str())
            .map(|s| s.contains("unauthorized"))
            .unwrap_or(false),
        "401 body must mention 'unauthorized': {body}"
    );
}

#[tokio::test]
async fn key_set_rejects_wrong_bearer() {
    let p = spawn_proxy_with_api_key(KEY).await;

    let resp = p
        .client
        .get(p.url("/api/version"))
        .header("authorization", "Bearer wrong-key")
        .send()
        .await
        .expect("GET /api/version wrong bearer");

    assert_eq!(resp.status(), 401);
    let body: Value = resp.json().await.expect("JSON error body");
    assert!(
        body.get("error").and_then(|v| v.as_str()).is_some(),
        "401 must carry an error field: {body}"
    );
}

#[tokio::test]
async fn key_set_accepts_correct_bearer() {
    let p = spawn_proxy_with_api_key(KEY).await;

    let resp = p
        .client
        .get(p.url("/api/version"))
        .header("authorization", format!("Bearer {KEY}"))
        .send()
        .await
        .expect("GET /api/version correct bearer");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("JSON body");
    assert!(
        body.get("version")
            .map(serde_json::Value::is_string)
            .unwrap_or(false),
        "authorized request must reach /api/version: {body}"
    );
}

// ── scheme is case-insensitive (bearer vs Bearer) ───────────────────────────

#[tokio::test]
async fn key_set_accepts_lowercase_bearer_scheme() {
    let p = spawn_proxy_with_api_key(KEY).await;

    let resp = p
        .client
        .get(p.url("/api/version"))
        .header("authorization", format!("bearer {KEY}"))
        .send()
        .await
        .expect("GET /api/version lowercase bearer");

    assert_eq!(resp.status(), 200);
}

// ── CORS preflight (OPTIONS) bypasses the gate ──────────────────────────────

#[tokio::test]
async fn key_set_allows_options_preflight_without_bearer() {
    let p = spawn_proxy_with_api_key(KEY).await;

    let resp = p
        .client
        .request(reqwest::Method::OPTIONS, p.url("/api/chat"))
        .header("origin", "http://localhost")
        .header("access-control-request-method", "POST")
        .send()
        .await
        .expect("OPTIONS /api/chat preflight");

    // CORS preflight must NOT be auth-blocked (401). It either serves the CORS
    // handshake or falls through, but never 401.
    assert_ne!(
        resp.status().as_u16(),
        401,
        "OPTIONS preflight must bypass the api-key gate"
    );
}
