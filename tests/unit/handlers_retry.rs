// with_retry_and_cancellation / with_simple_retry / execute_request_with_retry
// all require a live RequestContext (network) and async runtime with tokio::select!.
// The only pure-classification logic lives in crate::error::is_model_loading_error,
// which is imported here for exhaustive coverage.
use crate::error::is_model_loading_error;

// --- is_model_loading_error: known loading indicator phrases ---

#[test]
fn loading_indicator_exact_phrases() {
    let phrases = [
        "loading model",
        "model loading",
        "model is loading",
        "loading the model",
        "model not loaded",
        "not loaded",
        "model unavailable",
        "model not available",
        "model not found",
        "no model",
        "invalid model",
        "unknown model",
        "failed to load",
        "loading failed",
        "model error",
        "is not embedding",
        "model initialization",
        "initializing model",
        "warming up model",
        "model startup",
        "preparing model",
        "model not ready",
    ];
    for phrase in phrases {
        assert!(
            is_model_loading_error(phrase),
            "expected loading error for: {phrase}"
        );
    }
}

#[test]
fn loading_indicator_phrases_are_case_insensitive() {
    assert!(is_model_loading_error("MODEL NOT LOADED"));
    assert!(is_model_loading_error("Loading Model"));
    assert!(is_model_loading_error("UNKNOWN MODEL"));
}

// --- lm-studio-style patterns ---

#[test]
fn lm_studio_loading_patterns_trigger() {
    assert!(is_model_loading_error("503 service unavailable"));
    assert!(is_model_loading_error("500 internal error"));
    assert!(is_model_loading_error("server error occurred"));
    assert!(is_model_loading_error("connection refused"));
    assert!(is_model_loading_error("request timeout"));
}

// --- messages that should NOT trigger retry ---

#[test]
fn unrelated_errors_do_not_trigger() {
    assert!(!is_model_loading_error("invalid json body"));
    assert!(!is_model_loading_error("missing required field"));
    assert!(!is_model_loading_error("unauthorized"));
    assert!(!is_model_loading_error("rate limit exceeded"));
}

#[test]
fn empty_string_does_not_trigger() {
    assert!(!is_model_loading_error(""));
}
