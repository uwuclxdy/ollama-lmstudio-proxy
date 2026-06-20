use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

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

    match extract_bearer(req.headers().get(header::AUTHORIZATION)) {
        Some(token) if token == expected => next.run(req).await,
        _ => unauthorized(),
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

fn unauthorized() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        [(header::CONTENT_TYPE, "application/json")],
        UNAUTHORIZED_BODY,
    )
        .into_response()
}
