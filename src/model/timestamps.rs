//! Stable timestamps for /api/tags model entries.
//!
//! LM Studio's model list does not expose per-model last-modified time. Ollama's
//! schema still requires `modified_at`, so use a deterministic RFC3339 fallback
//! rather than a proxy runtime timestamp that changes every restart.

pub const MODEL_MODIFIED_AT_FALLBACK: &str = "1970-01-01T00:00:00Z";

pub fn model_modified_at_fallback() -> &'static str {
    MODEL_MODIFIED_AT_FALLBACK
}

#[cfg(test)]
#[path = "../../tests/unit/model_timestamps.rs"]
mod tests;
