//! LM Studio download orchestration.
//!
//! Translates Ollama's `/api/pull` request into LM Studio's
//! `/api/v1/models/download[/status]` endpoints and produces Ollama-shaped
//! NDJSON progress chunks.
//!
//! LM Studio's status payload is rich; Ollama's `/api/pull` stream is narrower
//! (see api_docs/ollama.md §"Pull a Model"):
//!   - in-progress: `{"status":"pulling <digest>", "digest":..., "total":..., "completed":...}`
//!   - terminal:    `{"status":"success"}` (literal — clients match by equality).

use std::time::Duration;

use bytes::Bytes;
use reqwest::Method;
use serde::Deserialize;
use serde_json::Value;
use tokio::sync::mpsc;
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;

use crate::api::RequestContext;
use crate::constants::{LM_STUDIO_NATIVE_DOWNLOAD, LM_STUDIO_NATIVE_DOWNLOAD_STATUS};
use crate::error::ProxyError;
use crate::http::client::{CancellableRequest, handle_json_response};
use crate::logging::log_request;
use crate::model::ModelResolver;
use crate::model::clean_model_name;
use crate::storage::VirtualModelEntry;

use crate::api::ollama::status_stream::send_status_chunk;

const DOWNLOAD_STATUS_POLL_INTERVAL_MS: u64 = 500;

// ---------------------------------------------------------------------------
// Download status payload (was download_status.rs)
// ---------------------------------------------------------------------------

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

    pub fn to_chunk(&self, _model: &str) -> Value {
        // Ollama clients match {"status":"success"} by equality to detect the end
        // of a pull stream. Terminal success chunks must therefore contain ONLY
        // the status field; in-progress chunks carry the spec progress numbers.
        //
        // `digest` is omitted: it is optional in the StatusEvent schema and LM
        // Studio returns no content digest for downloads.
        let translated = self.translated_status();
        if translated == "success" {
            return serde_json::json!({ "status": "success" });
        }

        let mut chunk = serde_json::Map::new();
        chunk.insert("status".to_string(), Value::String(translated));
        if let Some(total) = self.total_size_bytes {
            chunk.insert("total".to_string(), Value::from(total));
        }
        if let Some(done) = self.downloaded_bytes {
            chunk.insert("completed".to_string(), Value::from(done));
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

// ---------------------------------------------------------------------------
// Identifier resolution (was download_utils.rs)
// ---------------------------------------------------------------------------

pub fn looks_like_remote_identifier(identifier: &str) -> bool {
    let lowered = identifier.to_ascii_lowercase();
    lowered.starts_with("http://")
        || lowered.starts_with("https://")
        || lowered.starts_with("hf://")
        || lowered.starts_with("s3://")
        || lowered.starts_with("gs://")
}

pub fn extract_virtual_download_source(entry: &VirtualModelEntry) -> Option<String> {
    entry
        .metadata
        .parameters
        .as_ref()
        .and_then(|params| params.get("download_source"))
        .and_then(|value| value.as_str())
        .map(|value| value.to_string())
}

pub fn publisher_prefers_hf_link(publisher: &str) -> bool {
    matches!(
        publisher.to_ascii_lowercase().as_str(),
        "lmstudio-community" | "huggingface"
    )
}

pub fn build_hf_download_url(publisher: &str, model_id: &str) -> String {
    format!(
        "https://huggingface.co/{}/{}",
        publisher.trim().trim_end_matches('/'),
        model_id.trim_start_matches('/')
    )
}

pub fn build_catalog_identifier(publisher: &str, model_id: &str) -> Option<String> {
    let trimmed = publisher.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(format!(
        "{}/{}",
        trimmed.trim_end_matches('/'),
        model_id.trim_start_matches('/')
    ))
}

pub async fn determine_download_identifier(
    context: &RequestContext<'_>,
    model_resolver: &std::sync::Arc<ModelResolver>,
    requested_model: &str,
    source_override: Option<&str>,
    resolved_model: Option<(String, Option<VirtualModelEntry>)>,
    cancellation_token: CancellationToken,
) -> Result<String, ProxyError> {
    if let Some(source) = source_override {
        return Ok(source.to_string());
    }

    if looks_like_remote_identifier(requested_model) {
        return Ok(requested_model.to_string());
    }

    if let Some((resolved_model_id, virtual_entry)) = resolved_model {
        if let Some(source) = virtual_entry
            .as_ref()
            .and_then(extract_virtual_download_source)
        {
            return Ok(source);
        }

        if looks_like_remote_identifier(&resolved_model_id) {
            return Ok(resolved_model_id);
        }

        if resolved_model_id.contains('/') && !resolved_model_id.contains(' ') {
            return Ok(resolved_model_id);
        }

        if let Some(model_info) = crate::api::ollama::resolution::fetch_model_info_for_id(
            context,
            model_resolver,
            &resolved_model_id,
            cancellation_token,
        )
        .await?
        {
            let cleaned_id = clean_model_name(&model_info.id).to_string();
            if publisher_prefers_hf_link(&model_info.publisher) {
                return Ok(build_hf_download_url(&model_info.publisher, &cleaned_id));
            }

            if let Some(identifier) = build_catalog_identifier(&model_info.publisher, &cleaned_id) {
                return Ok(identifier);
            }
        }

        return Ok(resolved_model_id);
    }

    Ok(requested_model.to_string())
}

// ---------------------------------------------------------------------------
// Download orchestration
// ---------------------------------------------------------------------------

pub async fn initiate_lmstudio_download(
    client: &reqwest::Client,
    base_url: &str,
    model_identifier: &str,
    quantization: Option<&str>,
    cancellation_token: CancellationToken,
) -> Result<LmStudioDownloadStatus, ProxyError> {
    let mut payload = serde_json::Map::new();
    payload.insert(
        "model".to_string(),
        Value::String(model_identifier.to_string()),
    );
    if let Some(q) = quantization {
        payload.insert("quantization".to_string(), Value::String(q.to_string()));
    }

    let url = format!("{}{}", base_url, LM_STUDIO_NATIVE_DOWNLOAD);
    log_request("POST", &url, Some(model_identifier));

    let request = CancellableRequest::new(client, cancellation_token);
    let response = request
        .make_request(Method::POST, &url, Some(Value::Object(payload)))
        .await?;

    let response_value = handle_json_response(response, request.token().clone()).await?;

    serde_json::from_value(response_value).map_err(|e| {
        ProxyError::internal_server_error(&format!("invalid download response: {}", e))
    })
}

pub async fn wait_for_download_completion(
    client: &reqwest::Client,
    base_url: &str,
    mut status: LmStudioDownloadStatus,
    cancellation_token: CancellationToken,
) -> Result<LmStudioDownloadStatus, ProxyError> {
    while !status.is_terminal() {
        let job_id = status.job_id()?.to_string();
        tokio::select! {
            _ = cancellation_token.cancelled() => {
                return Err(ProxyError::request_cancelled());
            }
            _ = sleep(Duration::from_millis(DOWNLOAD_STATUS_POLL_INTERVAL_MS)) => {}
        }
        status =
            fetch_lmstudio_download_status(client, base_url, &job_id, cancellation_token.clone())
                .await?;
    }
    Ok(status)
}

pub async fn stream_download_status_updates(
    client: reqwest::Client,
    base_url: String,
    mut status: LmStudioDownloadStatus,
    model_name: String,
    cancellation_token: CancellationToken,
    tx: mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
) -> Result<(), ProxyError> {
    loop {
        if !send_status_chunk(&tx, &status.to_chunk(&model_name)) {
            return Ok(());
        }

        if status.is_failure() {
            return Err(ProxyError::internal_server_error(
                &status
                    .error
                    .clone()
                    .unwrap_or_else(|| "LM Studio download failed".to_string()),
            ));
        }

        if status.is_terminal() {
            return Ok(());
        }

        let job_id = status.job_id()?.to_string();
        tokio::select! {
            _ = cancellation_token.cancelled() => {
                return Err(ProxyError::request_cancelled());
            }
            _ = sleep(Duration::from_millis(DOWNLOAD_STATUS_POLL_INTERVAL_MS)) => {}
        }

        status =
            fetch_lmstudio_download_status(&client, &base_url, &job_id, cancellation_token.clone())
                .await?;
    }
}

async fn fetch_lmstudio_download_status(
    client: &reqwest::Client,
    base_url: &str,
    job_id: &str,
    cancellation_token: CancellationToken,
) -> Result<LmStudioDownloadStatus, ProxyError> {
    let url = format!(
        "{}{}/{}",
        base_url, LM_STUDIO_NATIVE_DOWNLOAD_STATUS, job_id
    );

    let request = CancellableRequest::new(client, cancellation_token);
    let response = request
        .make_request(Method::GET, &url, Option::<&Value>::None)
        .await?;

    let response_value = handle_json_response(response, request.token().clone()).await?;

    serde_json::from_value(response_value).map_err(|e| {
        ProxyError::internal_server_error(&format!("invalid download status payload: {}", e))
    })
}

#[cfg(test)]
#[path = "../../tests/unit/lmstudio_download_status.rs"]
mod tests_status;

#[cfg(test)]
#[path = "../../tests/unit/lmstudio_download_utils.rs"]
mod tests_utils;
