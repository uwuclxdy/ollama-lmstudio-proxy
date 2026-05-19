use super::*;
use bytes::Bytes;

// handlers/lmstudio.rs exposes one public function (handle_lmstudio_passthrough)
// and one pub struct (LmStudioPassthroughRequest).
// The only pure helper inside the module is determine_passthrough_endpoint_url,
// which is private (not re-exported).  All other behaviour involves live HTTP
// or streaming I/O.
//
// What we can test without a network is the LmStudioPassthroughRequest
// struct (construction / field access) and the fact that the module compiles
// with the expected public surface.

#[test]
fn passthrough_request_fields_accessible() {
    let req = LmStudioPassthroughRequest {
        method: http::Method::GET,
        endpoint: "/v1/models".to_string(),
        body: Bytes::new(),
        headers: http::HeaderMap::new(),
        query: None,
    };

    assert_eq!(req.method, http::Method::GET);
    assert_eq!(req.endpoint, "/v1/models");
    assert!(req.body.is_empty());
    assert!(req.query.is_none());
}

#[test]
fn passthrough_request_with_query() {
    let req = LmStudioPassthroughRequest {
        method: http::Method::POST,
        endpoint: "/v1/chat/completions".to_string(),
        body: Bytes::from_static(b"{}"),
        headers: http::HeaderMap::new(),
        query: Some("stream=true".to_string()),
    };

    assert_eq!(req.query.as_deref(), Some("stream=true"));
    assert!(!req.body.is_empty());
}
