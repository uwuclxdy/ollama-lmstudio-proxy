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

// ── a framework rejection's client bytes answer repr'd, not raw ────────────

#[tokio::test]
async fn bad_numeric_path_param_answers_its_value_quoted() {
    // Drives the production PathParams wrapper (and so extraction_rejected)
    // with a Path<u64> route, the shape a future typed param route would be:
    // the client value must answer quoted, never echoing raw as proxy wording.
    use std::sync::Arc;

    use axum::Router;
    use axum::routing::get;
    use serde_json::Value;

    use crate::proxy::routes::PathParams;

    #[axum::debug_handler]
    async fn num(_v: PathParams<u64>) -> &'static str {
        "ok"
    }

    let app = Router::new()
        .route("/n/{v}", get(num))
        .with_state(Arc::new(()));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve");
    });

    let resp = reqwest::Client::new()
        .get(format!("http://{}/n/12ab", addr))
        .send()
        .await
        .expect("GET /n/12ab");
    assert_eq!(resp.status(), 400);
    let value: Value = resp.json().await.expect("json body");
    let msg = value["error"].as_str().expect("`error` carries a string");
    // The whole rejection text answers repr'd: wrapped in double quotes, so
    // the client bytes inside read as a quoted echo of what was sent, never
    // as proxy wording.
    assert!(
        msg.starts_with('"') && msg.ends_with('"'),
        "rejection text must answer quoted, got: {msg}"
    );
    assert!(
        msg.contains("12ab"),
        "the value must survive the repr: {msg}"
    );

    handle.abort();
}
