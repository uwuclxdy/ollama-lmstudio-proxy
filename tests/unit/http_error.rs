// map_reqwest_error constructs ProxyError from a reqwest::Error.
// reqwest::Error cannot be constructed in pure unit tests (it requires an
// actual failed request), so the three branches (connect, timeout, other) are
// covered by integration tests instead.
//
// What we *can* test here are the ProxyError helpers that map_reqwest_error
// delegates to, and the is_model_loading_error predicate that lives alongside
// ProxyError.  We bring them in via crate paths because they live one level up
// from this module.

use crate::error::{ProxyError, is_model_loading_error};

// ── ProxyError constructors ──────────────────────────────────────────────────

#[test]
fn new_stores_message_and_status() {
    let e = ProxyError::new("oops".to_string(), 418);
    assert_eq!(e.message, "oops");
    assert_eq!(e.status_code, 418);
}

#[test]
fn internal_server_error_is_500() {
    let e = ProxyError::internal_server_error("boom");
    assert_eq!(e.status_code, 500);
    assert_eq!(e.message, "boom");
}

#[test]
fn bad_request_is_400() {
    let e = ProxyError::bad_request("nope");
    assert_eq!(e.status_code, 400);
    assert_eq!(e.message, "nope");
}

#[test]
fn not_found_is_404() {
    let e = ProxyError::not_found("gone");
    assert_eq!(e.status_code, 404);
    assert_eq!(e.message, "gone");
}

#[test]
fn not_implemented_is_501() {
    let e = ProxyError::not_implemented("todo");
    assert_eq!(e.status_code, 501);
    assert_eq!(e.message, "todo");
}

#[test]
fn request_cancelled_is_499() {
    let e = ProxyError::request_cancelled();
    assert_eq!(e.status_code, 499);
    assert!(e.is_cancelled());
}

#[test]
fn lm_studio_unavailable_is_503() {
    let e = ProxyError::lm_studio_unavailable("down");
    assert_eq!(e.status_code, 503);
    assert!(e.is_lm_studio_unavailable());
}

#[test]
fn is_cancelled_false_for_non_499() {
    let e = ProxyError::internal_server_error("x");
    assert!(!e.is_cancelled());
}

#[test]
fn is_lm_studio_unavailable_false_for_non_503() {
    let e = ProxyError::request_cancelled();
    assert!(!e.is_lm_studio_unavailable());
}

#[test]
fn display_includes_code_and_message() {
    let e = ProxyError::new("some error".to_string(), 422);
    let s = format!("{e}");
    assert!(s.contains("422"), "display must include status code: {s}");
    assert!(s.contains("some error"), "display must include message: {s}");
}

// ── is_model_loading_error ───────────────────────────────────────────────────

#[test]
fn loading_model_phrase_is_detected() {
    assert!(is_model_loading_error("loading model"));
    assert!(is_model_loading_error("model loading"));
    assert!(is_model_loading_error("model is loading"));
}

#[test]
fn not_loaded_phrase_is_detected() {
    assert!(is_model_loading_error("model not loaded"));
    assert!(is_model_loading_error("not loaded"));
}

#[test]
fn model_not_found_is_detected() {
    assert!(is_model_loading_error("model not found"));
}

#[test]
fn failed_to_load_is_detected() {
    assert!(is_model_loading_error("failed to load"));
    assert!(is_model_loading_error("loading failed"));
}

#[test]
fn is_not_embedding_is_detected() {
    assert!(is_model_loading_error("is not embedding model"));
}

#[test]
fn service_unavailable_phrase_is_detected() {
    assert!(is_model_loading_error("service unavailable"));
}

#[test]
fn timeout_is_detected() {
    assert!(is_model_loading_error("timeout"));
}

#[test]
fn plain_503_in_message_is_detected() {
    assert!(is_model_loading_error("HTTP 503 error returned"));
}

#[test]
fn negative_plus_model_ref_is_detected() {
    assert!(is_model_loading_error("no model available"));
    assert!(is_model_loading_error("invalid model selected"));
    assert!(is_model_loading_error("unknown model name"));
}

#[test]
fn unrelated_error_is_not_detected() {
    assert!(!is_model_loading_error("disk full"));
    assert!(!is_model_loading_error("permission denied"));
}

#[test]
fn empty_string_is_not_detected() {
    assert!(!is_model_loading_error(""));
}

#[test]
fn detection_is_case_insensitive() {
    assert!(is_model_loading_error("MODEL NOT READY"));
    assert!(is_model_loading_error("Loading Model"));
    assert!(is_model_loading_error("FAILED TO LOAD the model"));
}
