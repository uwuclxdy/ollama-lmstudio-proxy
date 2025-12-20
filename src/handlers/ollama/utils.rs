use serde_json::Value;

pub use super::download_utils::{determine_download_identifier, looks_like_remote_identifier};
pub use super::keep_alive::{keep_alive_requests_unload, parse_keep_alive_seconds};
pub use super::model_resolution::{resolve_model_target, resolve_model_with_context};
pub use super::status_streaming::{
    send_status_chunk, send_status_error_chunk, stream_status_messages,
};

/// Extracts system prompt from request body
pub fn extract_system_prompt(body: &Value) -> Option<String> {
    body.get("system")
        .and_then(|value| value.as_str())
        .map(|s| s.to_string())
        .or_else(|| {
            body.get("options")
                .and_then(|opts| opts.get("system"))
                .and_then(|value| value.as_str())
                .map(|s| s.to_string())
        })
}
