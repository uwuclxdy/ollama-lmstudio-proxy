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
    if let Some(pos) = name.rfind(":latest") {
        &name[..pos]
    } else {
        name
    }
}

#[cfg(test)]
#[path = "../../tests/unit/model_naming.rs"]
mod tests;
