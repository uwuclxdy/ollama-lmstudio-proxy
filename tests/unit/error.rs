use crate::error::{ProxyError, is_anthropic_surface};

// ── is_anthropic_surface: which paths wear the anthropic envelope ───────────

#[test]
fn surface_predicate_bounds_anthropic_to_the_messages_endpoint() {
    // The predicate answers "is this the anthropic surface?", which means
    // exactly `/v1/messages` and its subpaths — not any same-prefix stranger.
    assert!(is_anthropic_surface("/v1/messages"));
    assert!(is_anthropic_surface("/v1/messages/count_tokens"));
    assert!(is_anthropic_surface("/v1/messages/abc/def"));
    assert!(!is_anthropic_surface("/v1/messagesXYZ"));
    assert!(!is_anthropic_surface("/v1/messages-thing"));
    assert!(!is_anthropic_surface("/v1/chat/completions"));
    assert!(!is_anthropic_surface("/api/chat"));
    assert!(!is_anthropic_surface("/"));
}

// ── the envelope choice rides the error itself, not response extensions ────

#[test]
fn anthropic_error_type_maps_documented_statuses() {
    use crate::error::anthropic_error_type;
    // 405 rides the invalid_request arm: an unsupported method is a request
    // error in Anthropic's taxonomy, not a generic api_error.
    assert_eq!(anthropic_error_type(400), "invalid_request_error");
    assert_eq!(anthropic_error_type(405), "invalid_request_error");
    assert_eq!(anthropic_error_type(401), "authentication_error");
    assert_eq!(anthropic_error_type(404), "not_found_error");
    assert_eq!(anthropic_error_type(413), "request_too_large");
    assert_eq!(anthropic_error_type(429), "rate_limit_error");
    assert_eq!(anthropic_error_type(500), "api_error");
}

#[tokio::test]
async fn flagged_error_renders_the_anthropic_envelope() {
    use axum::http::StatusCode;
    use axum::response::IntoResponse;

    // The flag is set at the rejection site, and IntoResponse renders it —
    // with no envelope layer and no response extensions involved.
    let response = ProxyError::not_found("endpoint not found")
        .on_anthropic_surface_if("/v1/messages")
        .into_response();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    let value: serde_json::Value = serde_json::from_slice(&body).expect("json body");
    assert_eq!(
        value,
        serde_json::json!({
            "type": "error",
            "error": { "type": "not_found_error", "message": "endpoint not found" }
        })
    );
}

#[tokio::test]
async fn unflagged_error_renders_the_ollama_envelope() {
    use axum::http::StatusCode;
    use axum::response::IntoResponse;

    let response = ProxyError::not_found("endpoint not found")
        .on_anthropic_surface_if("/api/chat")
        .into_response();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    let value: serde_json::Value = serde_json::from_slice(&body).expect("json body");
    assert_eq!(value, serde_json::json!({ "error": "endpoint not found" }));
}
