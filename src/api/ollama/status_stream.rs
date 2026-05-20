use bytes::Bytes;
use serde_json::json;
use tokio::sync::mpsc;

use crate::error::ProxyError;

pub fn stream_status_messages(
    chunks: Vec<serde_json::Value>,
    error_label: &str,
) -> Result<axum::response::Response, ProxyError> {
    let (tx, rx) = mpsc::unbounded_channel();
    for chunk in chunks {
        if !send_status_chunk(&tx, &chunk) {
            break;
        }
    }
    drop(tx);
    crate::streaming::create_ndjson_stream_response(rx, error_label)
}

pub fn send_status_chunk(
    tx: &mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
    chunk: &serde_json::Value,
) -> bool {
    match serde_json::to_string(chunk) {
        Ok(serialized) => tx
            .send(Ok(Bytes::from(format!("{}\n", serialized))))
            .is_ok(),
        Err(e) => {
            log::warn!("pull chunk: serialization failed: {}", e);
            false
        }
    }
}

pub fn send_status_error_chunk(
    tx: &mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
    message: &str,
) {
    // Ollama spec §"Errors that occur while streaming": mid-stream error chunks
    // must be bare {"error":"..."} with no other fields.
    let chunk = json!({ "error": message });
    let _ = send_status_chunk(tx, &chunk);
}
