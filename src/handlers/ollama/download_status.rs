//! /api/pull status payload translation.
//!
//! LM Studio's /api/v1/models/download[/status] endpoint reports rich progress
//! state per job. Ollama's /api/pull NDJSON stream is narrower:
//!   - in-progress: {"status":"pulling <digest>", "digest":..., "total":..., "completed":...}
//!   - terminal:    {"status":"success"}   (literal — clients match by equality)
//!
//! See api_docs/ollama.md lines 1582-1626.

use serde::Deserialize;
use serde_json::Value;

use crate::error::ProxyError;

#[derive(Debug, Clone, Deserialize)]
pub struct LmStudioDownloadStatus {
    #[serde(default)]
    pub(crate) job_id: Option<String>,
    pub(crate) status: String,
    #[serde(default)]
    pub(crate) total_size_bytes: Option<u64>,
    #[serde(default)]
    pub(crate) downloaded_bytes: Option<u64>,
    #[serde(default)]
    pub(crate) bytes_per_second: Option<f64>,
    #[serde(default)]
    pub(crate) estimated_completion: Option<String>,
    #[serde(default)]
    pub(crate) started_at: Option<String>,
    #[serde(default)]
    pub(crate) completed_at: Option<String>,
    #[serde(default)]
    pub(crate) error: Option<String>,
}

impl LmStudioDownloadStatus {
    fn translated_status(&self) -> String {
        match self.status.as_str() {
            "completed" | "already_downloaded" => "success".to_string(),
            other => other.to_string(),
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(
            self.status.as_str(),
            "completed" | "already_downloaded" | "failed"
        )
    }

    pub(crate) fn is_failure(&self) -> bool {
        matches!(self.status.as_str(), "failed")
    }

    pub(crate) fn job_id(&self) -> Result<&str, ProxyError> {
        self.job_id.as_deref().ok_or_else(|| {
            ProxyError::internal_server_error("LM Studio download response missing job identifier")
        })
    }

    pub fn to_chunk(&self, model: &str) -> Value {
        // Ollama clients match {"status":"success"} by equality to detect the end
        // of a pull stream. Terminal success chunks must therefore contain ONLY
        // the status field; in-progress chunks carry the spec progress numbers.
        let translated = self.translated_status();
        if translated == "success" {
            return serde_json::json!({ "status": "success" });
        }

        let mut chunk = serde_json::Map::new();
        chunk.insert("status".to_string(), Value::String(translated));
        chunk.insert("model".to_string(), Value::String(model.to_string()));
        chunk.insert("detail".to_string(), Value::String(self.status.clone()));
        if let Some(job_id) = &self.job_id {
            chunk.insert("job_id".to_string(), Value::String(job_id.clone()));
        }
        if let Some(total) = self.total_size_bytes {
            chunk.insert("total".to_string(), Value::from(total));
        }
        if let Some(done) = self.downloaded_bytes {
            chunk.insert("completed".to_string(), Value::from(done));
        }
        if let Some(rate) = self.bytes_per_second {
            chunk.insert("bytes_per_second".to_string(), Value::from(rate));
        }
        if let Some(eta) = &self.estimated_completion {
            chunk.insert(
                "estimated_completion".to_string(),
                Value::String(eta.clone()),
            );
        }
        if let Some(started) = &self.started_at {
            chunk.insert("started_at".to_string(), Value::String(started.clone()));
        }
        if let Some(done_at) = &self.completed_at {
            chunk.insert("completed_at".to_string(), Value::String(done_at.clone()));
        }
        if let Some(err) = &self.error {
            chunk.insert("error".to_string(), Value::String(err.clone()));
        }
        Value::Object(chunk)
    }

    pub fn into_final_response(self, _model: &str) -> Result<Value, ProxyError> {
        match self.status.as_str() {
            "completed" | "already_downloaded" => {
                // Non-stream success: Ollama returns the bare sentinel object.
                Ok(serde_json::json!({ "status": "success" }))
            }
            "failed" => Err(ProxyError::internal_server_error(
                &self
                    .error
                    .clone()
                    .unwrap_or_else(|| "LM Studio reported download failure".to_string()),
            )),
            other => Err(ProxyError::internal_server_error(&format!(
                "unexpected download status: {}",
                other
            ))),
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/handlers_ollama_download_status.rs"]
mod tests;
