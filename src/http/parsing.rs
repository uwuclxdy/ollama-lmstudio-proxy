use serde_json::Value;
use warp::http::HeaderMap;

/// Parses JSON body template from headers and body
pub fn parse_json_body_template(
    headers: &HeaderMap,
    body: &[u8],
) -> Result<Option<Value>, Box<dyn std::error::Error + Send + Sync>> {
    if should_parse_as_json(headers, body)? {
        serde_json::from_slice(body)
            .map(Some)
            .map_err(|e| format!("Failed to parse JSON body: {}", e).into())
    } else {
        Ok(None)
    }
}

/// Determines if the request body should be parsed as JSON
pub fn should_parse_as_json(
    headers: &HeaderMap,
    body: &[u8],
) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
    if body.is_empty() {
        return Ok(false);
    }

    if contains_json_content_type(headers) {
        return Ok(true);
    }

    Ok(body_looks_like_json(body))
}

/// Checks if the request contains a JSON content type header
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

/// Determines if the body content looks like JSON
pub fn body_looks_like_json(body: &[u8]) -> bool {
    if let Ok(text) = std::str::from_utf8(body) {
        let trimmed = text.trim();
        return trimmed.starts_with('{') || trimmed.starts_with('[');
    }
    false
}
