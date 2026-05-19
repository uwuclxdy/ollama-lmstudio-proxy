use super::*;
use http::HeaderMap;
use http::header::{HeaderName, HeaderValue};

fn headers_with_content_type(ct: &'static str) -> HeaderMap {
    let mut h = HeaderMap::new();
    h.insert(
        HeaderName::from_static("content-type"),
        HeaderValue::from_static(ct),
    );
    h
}

// ── contains_json_content_type ───────────────────────────────────────────────

#[test]
fn application_json_recognized() {
    let h = headers_with_content_type("application/json");
    assert!(contains_json_content_type(&h));
}

#[test]
fn application_json_with_charset_recognized() {
    let h = headers_with_content_type("application/json; charset=utf-8");
    assert!(contains_json_content_type(&h));
}

#[test]
fn application_json_uppercase_recognized() {
    let h = headers_with_content_type("Application/JSON");
    assert!(contains_json_content_type(&h));
}

#[test]
fn application_plus_json_recognized() {
    let h = headers_with_content_type("application/*+json");
    assert!(contains_json_content_type(&h));
}

#[test]
fn text_plain_not_json() {
    let h = headers_with_content_type("text/plain");
    assert!(!contains_json_content_type(&h));
}

#[test]
fn text_html_not_json() {
    let h = headers_with_content_type("text/html");
    assert!(!contains_json_content_type(&h));
}

#[test]
fn empty_headers_not_json() {
    let h = HeaderMap::new();
    assert!(!contains_json_content_type(&h));
}

// ── body_looks_like_json ─────────────────────────────────────────────────────

#[test]
fn object_body_detected() {
    assert!(body_looks_like_json(b"{\"key\": 1}"));
}

#[test]
fn array_body_detected() {
    assert!(body_looks_like_json(b"[1, 2, 3]"));
}

#[test]
fn object_with_leading_whitespace_detected() {
    assert!(body_looks_like_json(b"   { }"));
}

#[test]
fn array_with_leading_whitespace_detected() {
    assert!(body_looks_like_json(b"\n\t[1]"));
}

#[test]
fn plain_text_not_detected_as_json() {
    assert!(!body_looks_like_json(b"hello world"));
}

#[test]
fn empty_body_not_detected_as_json() {
    assert!(!body_looks_like_json(b""));
}

#[test]
fn non_utf8_bytes_not_detected_as_json() {
    assert!(!body_looks_like_json(&[0xff, 0xfe, 0x00]));
}

#[test]
fn quoted_string_not_detected_as_json() {
    assert!(!body_looks_like_json(b"\"just a string\""));
}

// ── should_parse_as_json ─────────────────────────────────────────────────────

#[test]
fn empty_body_returns_false() {
    let h = headers_with_content_type("application/json");
    assert!(!should_parse_as_json(&h, b"").unwrap());
}

#[test]
fn json_content_type_with_body_returns_true() {
    let h = headers_with_content_type("application/json");
    assert!(should_parse_as_json(&h, b"{\"x\":1}").unwrap());
}

#[test]
fn no_content_type_json_looking_body_returns_true() {
    let h = HeaderMap::new();
    assert!(should_parse_as_json(&h, b"{\"x\":1}").unwrap());
}

#[test]
fn no_content_type_non_json_body_returns_false() {
    let h = HeaderMap::new();
    assert!(!should_parse_as_json(&h, b"raw bytes here").unwrap());
}

#[test]
fn text_plain_content_type_with_array_body_returns_true_via_heuristic() {
    let h = headers_with_content_type("text/plain");
    // body looks like JSON regardless of content-type
    assert!(should_parse_as_json(&h, b"[1,2]").unwrap());
}

// ── parse_json_body_template ─────────────────────────────────────────────────

#[test]
fn valid_json_body_with_json_content_type_returns_some() {
    let h = headers_with_content_type("application/json");
    let result = parse_json_body_template(&h, b"{\"model\":\"llama3\"}").unwrap();
    assert!(result.is_some());
    assert_eq!(result.unwrap()["model"], "llama3");
}

#[test]
fn valid_json_body_without_content_type_returns_some_via_heuristic() {
    let h = HeaderMap::new();
    let result = parse_json_body_template(&h, b"{\"a\":1}").unwrap();
    assert!(result.is_some());
}

#[test]
fn empty_body_returns_none() {
    let h = headers_with_content_type("application/json");
    let result = parse_json_body_template(&h, b"").unwrap();
    assert!(result.is_none());
}

#[test]
fn invalid_json_with_json_content_type_returns_error() {
    let h = headers_with_content_type("application/json");
    let result = parse_json_body_template(&h, b"{not valid json}");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.status_code, 400, "malformed JSON must yield 400");
}

#[test]
fn non_json_content_type_non_json_body_returns_none() {
    let h = headers_with_content_type("multipart/form-data");
    let result = parse_json_body_template(&h, b"--boundary\r\ncontent").unwrap();
    assert!(result.is_none());
}

#[test]
fn array_json_body_parsed_correctly() {
    let h = headers_with_content_type("application/json");
    let result = parse_json_body_template(&h, b"[1,2,3]").unwrap();
    assert!(result.is_some());
    assert_eq!(result.unwrap(), serde_json::json!([1, 2, 3]));
}
