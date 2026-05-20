//! Classifies LM Studio error messages that look like "model not loaded yet"
//! so the retry layer knows to trigger a load and try again.

pub fn is_model_loading_error(message: &str) -> bool {
    let lower = message.to_lowercase();

    let loading_indicators = [
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

    if loading_indicators
        .iter()
        .any(|&pattern| lower.contains(pattern))
    {
        return true;
    }

    let has_negative = lower.contains("no")
        || lower.contains("not")
        || lower.contains("missing")
        || lower.contains("invalid")
        || lower.contains("unknown")
        || lower.contains("failed")
        || lower.contains("unavailable")
        || lower.contains("unreachable");

    let has_model_ref = lower.contains("model")
        || lower.contains("load")
        || lower.contains("available")
        || lower.contains("ready")
        || lower.contains("initialize");

    let lm_studio_loading_patterns = [
        "service unavailable",
        "server error",
        "internal error",
        "timeout",
        "connection",
        "503",
        "500",
    ];

    let has_lm_studio_loading = lm_studio_loading_patterns
        .iter()
        .any(|&pattern| lower.contains(pattern));

    (has_negative && has_model_ref) || has_lm_studio_loading
}
