// with_retry_and_cancellation / with_simple_retry / execute_request_with_retry
// all require a live RequestContext (network) and async runtime with tokio::select!.
// The only pure-classification logic lives in crate::lmstudio::is_model_loading_error,
// which is imported here for exhaustive coverage.
use crate::api::retry::should_trigger_load;
use crate::lmstudio::is_model_loading_error;

// --- should_trigger_load: status + message gate the load-and-retry path ---

#[test]
fn no_models_loaded_400_triggers_load() {
    // LM Studio answers a not-yet-loaded model with a 400 "No models loaded";
    // that must fall through to the trigger-and-retry path, not return as-is.
    assert!(should_trigger_load(400, "No models loaded"));
}

#[test]
fn unrelated_400_does_not_trigger_load() {
    assert!(!should_trigger_load(400, "missing required field"));
}

#[test]
fn proxy_validation_400_does_not_trigger_load() {
    // The loose `is_model_loading_error` classifier matches this (it sees "no"
    // inside "cannot" + the word "model"), but it's a proxy-side validation
    // 400, not a load signal — the narrow 400 gate must reject it.
    assert!(!should_trigger_load(
        400,
        "`raw` cannot be combined with `images`: LM Studio cannot serve raw prompts to vision models"
    ));
}

#[test]
fn model_not_found_404_does_not_trigger_load() {
    // A 404 means the model genuinely doesn't exist; loading it is futile, so it
    // must return verbatim rather than detour through a load-and-retry.
    assert!(!should_trigger_load(
        404,
        "model 'does-not-exist' not found in LM Studio"
    ));
}

#[test]
fn loading_5xx_triggers_load() {
    assert!(should_trigger_load(503, "model is loading"));
    assert!(should_trigger_load(500, "model not ready"));
}

#[test]
fn passthrough_statuses_never_trigger_load() {
    // 429/502 are forwarded verbatim by the caller and must not detour through
    // a load even if their message would otherwise match the classifier.
    assert!(!should_trigger_load(429, "model not loaded"));
    assert!(!should_trigger_load(502, "model unreachable"));
}

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

// --- bare numbers and transport words in free text are not load state ---

#[test]
fn numbers_in_free_text_do_not_trigger() {
    // The decode error embeds the upstream url, so the ephemeral port can supply
    // "500"/"503" as a substring; classification must not read numbers as load state.
    assert!(!is_model_loading_error(
        "invalid JSON from LM Studio: error decoding response body for url (http://127.0.0.1:50003/v1/embeddings)"
    ));
    assert!(!is_model_loading_error("500 internal error"));
    assert!(!is_model_loading_error("request timeout"));
}

#[test]
fn transport_words_do_not_trigger() {
    assert!(!is_model_loading_error("server error occurred"));
    assert!(!is_model_loading_error("connection refused"));
}

#[test]
fn service_unavailable_still_triggers_via_model_ref() {
    // "unavailable" carries both a negative and an availability model-reference.
    assert!(is_model_loading_error("503 service unavailable"));
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
