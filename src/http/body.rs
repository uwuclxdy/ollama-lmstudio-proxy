use http::HeaderMap;
use serde_json::Value;

use crate::error::ProxyError;

pub struct PreparedBody {
    pub bytes: Option<Vec<u8>>,
    pub is_json: bool,
}

pub fn prepare_request_body(
    json_body: Option<Value>,
    original_bytes: &[u8],
) -> Result<PreparedBody, ProxyError> {
    if let Some(value) = json_body {
        let serialized = serde_json::to_vec(&value).map_err(|e| {
            ProxyError::bad_request(&format!("failed to serialize request body: {}", e))
        })?;
        return Ok(PreparedBody {
            bytes: Some(serialized),
            is_json: true,
        });
    }
    if !original_bytes.is_empty() {
        return Ok(PreparedBody {
            bytes: Some(original_bytes.to_vec()),
            is_json: false,
        });
    }
    Ok(PreparedBody {
        bytes: None,
        is_json: false,
    })
}

pub fn parse_json_body_template(
    headers: &HeaderMap,
    body: &[u8],
) -> Result<Option<Value>, ProxyError> {
    if should_parse_as_json(headers, body)? {
        serde_json::from_slice(body)
            .map(Some)
            .map_err(|e| ProxyError::bad_request(&format!("invalid JSON body: {}", e)))
    } else {
        Ok(None)
    }
}

pub fn should_parse_as_json(headers: &HeaderMap, body: &[u8]) -> Result<bool, ProxyError> {
    if body.is_empty() {
        return Ok(false);
    }

    if contains_json_content_type(headers) {
        return Ok(true);
    }

    Ok(body_looks_like_json(body))
}

pub fn contains_json_content_type(headers: &HeaderMap) -> bool {
    headers
        .get("content-type")
        .and_then(|ct| ct.to_str().ok())
        .map(|ct| {
            let lower = ct.to_lowercase();
            lower.contains("application/json") || lower.contains("application/*+json")
        })
        .unwrap_or(false)
}

pub fn body_looks_like_json(body: &[u8]) -> bool {
    let Ok(text) = std::str::from_utf8(body) else {
        return false;
    };
    let trimmed = text.trim();
    trimmed.starts_with('{') || trimmed.starts_with('[')
}

#[cfg(test)]
#[path = "../../tests/unit/http_body.rs"]
mod tests;
