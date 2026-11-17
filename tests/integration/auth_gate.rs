// Integration tests for the inbound `--api-key` / `OLLAMA_API_KEY` Bearer gate
// (`src/proxy/auth.rs::api_key_gate`).
//
// When configured, every inbound request must carry `Authorization: Bearer
// <key>` (or the Anthropic `x-api-key` header) or receive a 401 with a JSON
// `{"error":"unauthorized"}` body. When unset (the default), the proxy is fully
// open and inspects no auth header. `/api/version` is exempt from the gate
// (served unauthenticated, like real Ollama), so gate-behavior tests probe `/`
// (root) instead — both are served locally with no LM Studio mock, keeping
// these tests fully deterministic.

use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, ResponseTemplate};

use crate::common::{spawn_proxy, spawn_proxy_with_api_key, spawn_proxy_with_api_key_and_token};

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

    let resp = p.client.get(p.url("/")).send().await.expect("GET /");
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
        .get(p.url("/"))
        .header("authorization", "Bearer wrong-key")
        .send()
        .await
        .expect("GET / wrong bearer");

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
        .get(p.url("/"))
        .header("authorization", format!("Bearer {KEY}"))
        .send()
        .await
        .expect("GET / correct bearer");

    assert_eq!(resp.status(), 200);
    let body = resp.text().await.expect("body text");
    assert!(
        body.contains("Ollama is running"),
        "authorized request must reach the root handler: {body}"
    );
}

// ── scheme is case-insensitive (bearer vs Bearer) ───────────────────────────

#[tokio::test]
async fn key_set_accepts_lowercase_bearer_scheme() {
    let p = spawn_proxy_with_api_key(KEY).await;

    let resp = p
        .client
        .get(p.url("/"))
        .header("authorization", format!("bearer {KEY}"))
        .send()
        .await
        .expect("GET / lowercase bearer");

    assert_eq!(resp.status(), 200);
}

// ── /api/version is exempt from the gate (unauthenticated, like real Ollama) ──

#[tokio::test]
async fn version_exempt_from_gate_without_header() {
    let p = spawn_proxy_with_api_key(KEY).await;

    let resp = p
        .client
        .get(p.url("/api/version"))
        .send()
        .await
        .expect("GET /api/version no header");

    assert_eq!(
        resp.status(),
        200,
        "/api/version must be served unauthenticated even when the api-key gate is on"
    );
    let body: Value = resp.json().await.expect("JSON body");
    assert!(
        body.get("version")
            .map(serde_json::Value::is_string)
            .unwrap_or(false),
        "exempt /api/version must still return the version payload: {body}"
    );
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

// ── x-api-key (Anthropic SDK default header) is accepted alongside Bearer ────

#[tokio::test]
async fn key_set_accepts_correct_x_api_key() {
    let p = spawn_proxy_with_api_key(KEY).await;

    let resp = p
        .client
        .get(p.url("/"))
        .header("x-api-key", KEY)
        .send()
        .await
        .expect("GET / x-api-key");

    assert_eq!(
        resp.status(),
        200,
        "a correct x-api-key must pass the gate (Anthropic SDK default header)"
    );
}

#[tokio::test]
async fn key_set_rejects_wrong_x_api_key() {
    let p = spawn_proxy_with_api_key(KEY).await;

    let resp = p
        .client
        .get(p.url("/"))
        .header("x-api-key", "wrong-key")
        .send()
        .await
        .expect("GET / wrong x-api-key");

    assert_eq!(resp.status(), 401);
}

#[tokio::test]
async fn correct_x_api_key_is_accepted_beside_a_stale_bearer() {
    // A client switching to x-api-key can leave a stale Authorization header
    // behind; the gate must judge each credential on its own rather than let
    // the first non-empty one short-circuit the other.
    let p = spawn_proxy_with_api_key(KEY).await;

    let resp = p
        .client
        .get(p.url("/"))
        .header("authorization", "Bearer wrong-key")
        .header("x-api-key", KEY)
        .send()
        .await
        .expect("GET / stale bearer + correct x-api-key");

    assert_eq!(
        resp.status(),
        200,
        "a correct x-api-key must authenticate even when Authorization carries a stale Bearer token"
    );
}

#[tokio::test]
async fn wrong_values_in_both_credential_headers_are_rejected() {
    let p = spawn_proxy_with_api_key(KEY).await;

    let resp = p
        .client
        .get(p.url("/"))
        .header("authorization", "Bearer wrong-key")
        .header("x-api-key", "wrong-key")
        .send()
        .await
        .expect("GET / wrong bearer + wrong x-api-key");

    assert_eq!(resp.status(), 401);
}

// ── /v1/messages 401 wears the Anthropic error envelope ─────────────────────

#[tokio::test]
async fn key_set_rejects_v1_messages_with_anthropic_envelope() {
    let p = spawn_proxy_with_api_key(KEY).await;

    let resp = p
        .client
        .post(p.url("/v1/messages"))
        .json(&serde_json::json!({
            "model": "m",
            "max_tokens": 5,
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .send()
        .await
        .expect("POST /v1/messages no auth");

    assert_eq!(resp.status(), 401);
    let body: Value = resp.json().await.expect("JSON error body");
    assert_eq!(body["type"], "error", "Anthropic envelope: {body}");
    assert_eq!(body["error"]["type"], "authentication_error");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("unauthorized")),
        "message must mention unauthorized: {body}"
    );
}

// ── the api-key gate owns Authorization: the proxy key never rides upstream ──
//
// With `--api-key` set, the inbound `Authorization` / `x-api-key` header is the
// PROXY credential. Forwarding it overwrote the configured `--lmstudio-token`
// (reqwest's per-request header wins over the client default), so a backend
// with its own auth answered 401.

/// Mount a plain 200 for `GET /v1/models` and return the request the backend saw.
async fn forwarded_v1_models_request(p: &crate::common::TestProxy) -> wiremock::Request {
    let recorded = p
        .mock
        .received_requests()
        .await
        .expect("request recording enabled");
    recorded
        .into_iter()
        .find(|r| r.url.path() == "/v1/models")
        .expect("the backend must see GET /v1/models")
}

async fn mount_v1_models(p: &crate::common::TestProxy) {
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "data": [] })))
        .mount(&p.mock)
        .await;
}

#[tokio::test]
async fn matched_api_key_is_stripped_so_the_lmstudio_token_is_forwarded() {
    let p = spawn_proxy_with_api_key_and_token(KEY, "lms-outbound-token").await;
    mount_v1_models(&p).await;

    let resp = p
        .client
        .get(p.url("/v1/models"))
        .header("authorization", format!("Bearer {KEY}"))
        .send()
        .await
        .expect("GET /v1/models with the proxy key");

    assert_eq!(resp.status(), 200);
    let forwarded = forwarded_v1_models_request(&p).await;
    assert_eq!(
        forwarded
            .headers
            .get("authorization")
            .and_then(|v| v.to_str().ok()),
        Some("Bearer lms-outbound-token"),
        "the backend must receive the configured LM Studio token, never the proxy key"
    );
}

#[tokio::test]
async fn matched_x_api_key_is_stripped_from_the_forwarded_request() {
    let p = spawn_proxy_with_api_key_and_token(KEY, "lms-outbound-token").await;
    mount_v1_models(&p).await;

    let resp = p
        .client
        .get(p.url("/v1/models"))
        .header("x-api-key", KEY)
        .send()
        .await
        .expect("GET /v1/models with x-api-key");

    assert_eq!(resp.status(), 200);
    let forwarded = forwarded_v1_models_request(&p).await;
    assert!(
        forwarded.headers.get("x-api-key").is_none(),
        "the proxy credential must not ride upstream as x-api-key"
    );
    assert_eq!(
        forwarded
            .headers
            .get("authorization")
            .and_then(|v| v.to_str().ok()),
        Some("Bearer lms-outbound-token"),
        "the configured LM Studio token must be the one forwarded"
    );
}

#[tokio::test]
async fn matched_api_key_is_not_forwarded_without_an_lmstudio_token() {
    let p = spawn_proxy_with_api_key(KEY).await;
    mount_v1_models(&p).await;

    let resp = p
        .client
        .get(p.url("/v1/models"))
        .header("authorization", format!("Bearer {KEY}"))
        .send()
        .await
        .expect("GET /v1/models with the proxy key");

    assert_eq!(resp.status(), 200);
    let forwarded = forwarded_v1_models_request(&p).await;
    assert!(
        forwarded.headers.get("authorization").is_none(),
        "with no configured backend credential, nothing must be forwarded as Authorization"
    );
}

#[tokio::test]
async fn matched_authorization_strips_a_caller_x_api_key_too() {
    let p = spawn_proxy_with_api_key_and_token(KEY, "lms-outbound-token").await;
    mount_v1_models(&p).await;

    let resp = p
        .client
        .get(p.url("/v1/models"))
        .header("authorization", format!("Bearer {KEY}"))
        .header("x-api-key", "caller-lm-token")
        .send()
        .await
        .expect("GET /v1/models with both credential headers");

    assert_eq!(resp.status(), 200);
    let forwarded = forwarded_v1_models_request(&p).await;
    assert!(
        forwarded.headers.get("x-api-key").is_none(),
        "the second credential channel must not override the backend token"
    );
    assert_eq!(
        forwarded
            .headers
            .get("authorization")
            .and_then(|v| v.to_str().ok()),
        Some("Bearer lms-outbound-token"),
        "the configured LM Studio token must stand"
    );
}

#[tokio::test]
async fn matched_x_api_key_is_not_forwarded_without_an_lmstudio_token() {
    let p = spawn_proxy_with_api_key(KEY).await;
    mount_v1_models(&p).await;

    let resp = p
        .client
        .get(p.url("/v1/models"))
        .header("x-api-key", KEY)
        .send()
        .await
        .expect("GET /v1/models with x-api-key and no backend token");

    assert_eq!(resp.status(), 200);
    let forwarded = forwarded_v1_models_request(&p).await;
    assert!(
        forwarded.headers.get("x-api-key").is_none(),
        "the matched x-api-key must never ride upstream"
    );
    assert!(
        forwarded.headers.get("authorization").is_none(),
        "no configured backend credential means no Authorization either"
    );
}
