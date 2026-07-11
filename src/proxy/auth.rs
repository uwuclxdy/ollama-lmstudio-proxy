use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use subtle::ConstantTimeEq;

const UNAUTHORIZED_BODY: &str = r#"{"error":"unauthorized"}"#;

/// Inbound Bearer API-key gate. When the configured key is `None` the gate is a
/// pure no-op (fully open). When set, inbound requests must carry
/// `Authorization: Bearer <key>` or receive a 401.
pub async fn api_key_gate(
    State(expected): State<Arc<Option<String>>>,
    req: Request,
    next: Next,
) -> Response {
    // No key configured -> fully open.
    let Some(expected) = expected.as_deref() else {
        return next.run(req).await;
    };

    // CORS owns preflight; never auth-block OPTIONS.
    if req.method() == axum::http::Method::OPTIONS {
        return next.run(req).await;
    }

    // `/api/version` is unauthenticated in real Ollama (its OpenAPI declares
    // `security: []`) and the version string is not sensitive — exempt it from
    // the gate. Every other endpoint stays gated (deliberate hardening).
    if req.uri().path() == "/api/version" {
        return next.run(req).await;
    }

    // Accept the OpenAI-style `Authorization: Bearer <key>` or the Anthropic
    // SDK's default `x-api-key: <key>` header. Constant-time compare so the
    // check can't be timed into a byte-at-a-time oracle; ct_eq returns unequal
    // (0) without panicking on differing token lengths. `matches!` drops the
    // header borrow before `req` moves into `next.run`.
    let authorized = matches!(
        extract_bearer(req.headers().get(header::AUTHORIZATION))
            .or_else(|| extract_api_key(req.headers().get("x-api-key"))),
        Some(token) if expected.as_bytes().ct_eq(token.as_bytes()).into()
    );

    if authorized {
        next.run(req).await
    } else {
        log::warn!(
            "auth: rejected {} {} (missing or invalid credentials)",
            req.method(),
            req.uri().path()
        );
        unauthorized(req.uri().path())
    }
}

/// Parse `Authorization: Bearer <token>`; scheme match is case-insensitive.
fn extract_bearer(value: Option<&HeaderValue>) -> Option<&str> {
    let header = value?.to_str().ok()?;
    let (scheme, rest) = header.split_once(' ')?;
    if !scheme.eq_ignore_ascii_case("bearer") {
        return None;
    }
    let token = rest.trim_start();
    if token.is_empty() {
        return None;
    }
    Some(token)
}

/// Parse the Anthropic-style `x-api-key: <token>` header (no scheme prefix).
fn extract_api_key(value: Option<&HeaderValue>) -> Option<&str> {
    let key = value?.to_str().ok()?.trim();
    if key.is_empty() {
        return None;
    }
    Some(key)
}

fn unauthorized(path: &str) -> Response {
    // Anthropic SDKs on /v1/messages* parse {"type":"error","error":{...}};
    // the Ollama {"error":msg} shape reads as a malformed response to them.
    if path.starts_with("/v1/messages") {
        let body = crate::error::anthropic_error_body(401, "unauthorized").to_string();
        return (
            StatusCode::UNAUTHORIZED,
            [(header::CONTENT_TYPE, "application/json")],
            body,
        )
            .into_response();
    }
    (
        StatusCode::UNAUTHORIZED,
        [(header::CONTENT_TYPE, "application/json")],
        UNAUTHORIZED_BODY,
    )
        .into_response()
}
