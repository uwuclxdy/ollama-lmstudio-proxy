use super::*;
use http::HeaderMap;
use http::header::{HeaderName, HeaderValue};
use serde_json::json;

fn make_warp_headers(pairs: &[(&str, &str)]) -> HeaderMap {
    let mut h = HeaderMap::new();
    for (k, v) in pairs {
        h.insert(
            HeaderName::from_bytes(k.as_bytes()).unwrap(),
            HeaderValue::from_str(v).unwrap(),
        );
    }
    h
}

// ── json_response ────────────────────────────────────────────────────────────

#[test]
fn json_response_status_is_200() {
    let resp = json_response(&json!({"ok": true}));
    assert_eq!(resp.status(), http::StatusCode::OK);
}

#[test]
fn json_response_content_type_header_present() {
    let resp = json_response(&json!({"x": 1}));
    let ct = resp
        .headers()
        .get("content-type")
        .expect("content-type header must be present");
    assert!(
        ct.to_str().unwrap().contains("application/json"),
        "content-type must contain application/json, got {:?}",
        ct
    );
}

#[tokio::test]
async fn json_response_body_round_trips() {
    use http_body_util::BodyExt;
    let original = json!({"model": "llama3", "done": true});
    let resp = json_response(&original);
    let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let parsed: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(parsed, original);
}

#[test]
fn json_response_cors_headers_present() {
    let resp = json_response(&json!({}));
    let headers = resp.headers();
    assert!(
        headers.contains_key("access-control-allow-origin"),
        "CORS origin header must be present"
    );
    assert!(
        headers.contains_key("access-control-allow-methods"),
        "CORS methods header must be present"
    );
    assert!(
        headers.contains_key("access-control-allow-headers"),
        "CORS headers header must be present"
    );
}

#[test]
fn json_response_content_length_matches_body() {
    let value = json!({"hello": "world"});
    let json_str = serde_json::to_string(&value).unwrap();
    let expected_len = json_str.len();
    let resp = json_response(&value);
    let cl = resp
        .headers()
        .get("content-length")
        .expect("content-length must be present");
    let reported: usize = cl.to_str().unwrap().parse().unwrap();
    assert_eq!(reported, expected_len);
}

#[test]
fn json_response_cache_control_is_no_cache() {
    let resp = json_response(&json!(null));
    let cc = resp
        .headers()
        .get("cache-control")
        .expect("cache-control must be present");
    assert_eq!(cc.to_str().unwrap(), "no-cache");
}

// ── build_forward_headers ────────────────────────────────────────────────────

#[test]
fn hop_by_hop_headers_are_removed() {
    let h = make_warp_headers(&[
        ("host", "localhost:11434"),
        ("content-length", "42"),
        ("transfer-encoding", "chunked"),
        ("authorization", "Bearer tok"),
    ]);
    let out = build_forward_headers(&h, false);
    assert!(
        !out.contains_key("host"),
        "host must be stripped, got {:?}",
        out
    );
    assert!(
        !out.contains_key("content-length"),
        "content-length must be stripped"
    );
    assert!(
        !out.contains_key("transfer-encoding"),
        "transfer-encoding must be stripped"
    );
    assert!(out.contains_key("authorization"), "authorization must pass");
}

#[test]
fn force_json_removes_original_content_type_and_sets_json() {
    let h = make_warp_headers(&[("content-type", "text/plain")]);
    let out = build_forward_headers(&h, true);
    let ct = out
        .get("content-type")
        .expect("content-type must be present after force_json");
    assert!(
        ct.to_str().unwrap().contains("application/json"),
        "content-type must be application/json when force_json=true, got {:?}",
        ct
    );
}

#[test]
fn force_json_false_preserves_original_content_type() {
    let h = make_warp_headers(&[("content-type", "text/plain")]);
    let out = build_forward_headers(&h, false);
    let ct = out
        .get("content-type")
        .expect("content-type must be preserved");
    assert_eq!(ct.to_str().unwrap(), "text/plain");
}

#[test]
fn empty_headers_with_force_json_produces_only_content_type() {
    let h = HeaderMap::new();
    let out = build_forward_headers(&h, true);
    assert!(out.contains_key("content-type"), "must insert content-type");
    assert_eq!(out.len(), 1);
}

#[test]
fn empty_headers_without_force_json_produces_empty_map() {
    let h = HeaderMap::new();
    let out = build_forward_headers(&h, false);
    assert!(out.is_empty());
}

#[test]
fn custom_headers_pass_through() {
    let h = make_warp_headers(&[
        ("x-request-id", "abc-123"),
        ("authorization", "Bearer secret"),
    ]);
    let out = build_forward_headers(&h, false);
    assert!(out.contains_key("x-request-id"));
    assert!(out.contains_key("authorization"));
}

#[test]
fn header_names_are_case_insensitively_stripped() {
    // warp HeaderMap normalizes to lowercase, but test the edge explicitly
    let h = make_warp_headers(&[("host", "example.com"), ("content-length", "0")]);
    let out = build_forward_headers(&h, false);
    assert!(!out.contains_key("host"));
    assert!(!out.contains_key("content-length"));
}

// ── caller authorization header preserved on passthrough ─────────────────────

#[test]
fn caller_authorization_is_forwarded_on_passthrough() {
    // When the caller provides an Authorization header, build_forward_headers
    // must carry it through so that it overrides any client-level default header
    // (i.e., the proxy token is not forced onto passthrough requests that
    // already have their own auth).
    let h = make_warp_headers(&[("authorization", "Bearer caller-token")]);
    let out = build_forward_headers(&h, false);
    let auth = out
        .get("authorization")
        .expect("authorization must be forwarded");
    assert_eq!(auth.to_str().unwrap(), "Bearer caller-token");
}

#[test]
fn no_authorization_from_caller_produces_no_auth_header() {
    // When the caller sends no Authorization, build_forward_headers must not
    // invent one — the client default header (if set) fills the gap at send time.
    let h = make_warp_headers(&[("content-type", "application/json")]);
    let out = build_forward_headers(&h, false);
    assert!(
        !out.contains_key("authorization"),
        "must not inject an authorization header when the caller provided none"
    );
}
