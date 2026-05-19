//! Stable timestamps for /api/tags model entries.
//!
//! Ollama's `modified_at` field reflects when the model was last modified on
//! disk. LM Studio does not surface that timestamp, so the proxy used to
//! return `Utc::now()` on every call — breaking client caches that key on the
//! timestamp. Cache once at process start and return the same RFC3339 string
//! for the lifetime of the proxy.

use std::sync::OnceLock;

use chrono::{DateTime, Utc};

static PROCESS_START: OnceLock<DateTime<Utc>> = OnceLock::new();

fn process_start() -> &'static DateTime<Utc> {
    PROCESS_START.get_or_init(Utc::now)
}

pub fn process_start_modified_at() -> String {
    process_start().to_rfc3339()
}

#[cfg(test)]
#[path = "../../tests/unit/model_timestamps.rs"]
mod tests;
