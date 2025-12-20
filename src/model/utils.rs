use serde_json::Value;

use crate::constants::ERROR_MISSING_MODEL;
use crate::error::ProxyError;

pub fn extract_required_model_name(body: &Value) -> Result<&str, ProxyError> {
    body.get("model")
        .and_then(|m| m.as_str())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| ProxyError::bad_request(ERROR_MISSING_MODEL))
}

pub fn clean_model_name(name: &str) -> &str {
    if name.is_empty() {
        return name;
    }
    let after_latest = if let Some(pos) = name.rfind(":latest") {
        &name[..pos]
    } else {
        name
    };
    if let Some(colon_pos) = after_latest.rfind(':') {
        let suffix = &after_latest[colon_pos + 1..];
        if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) && colon_pos > 0 {
            return &after_latest[..colon_pos];
        }
    }
    after_latest
}
