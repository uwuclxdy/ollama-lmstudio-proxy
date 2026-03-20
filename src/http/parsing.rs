use serde_json::Value;
use warp::http::HeaderMap;

use crate::error::ProxyError;

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
            ct.to_lowercase().contains("application/json")
                || ct.to_lowercase().contains("application/*+json")
        })
        .unwrap_or(false)
}

pub fn body_looks_like_json(body: &[u8]) -> bool {
    if let Ok(text) = std::str::from_utf8(body) {
        let trimmed = text.trim();
        return trimmed.starts_with('{') || trimmed.starts_with('[');
    }
    false
}
