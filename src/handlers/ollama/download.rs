use bytes::Bytes;
use reqwest::Method;
use serde::Deserialize;
use serde_json::Value;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;

use crate::constants::{LM_STUDIO_NATIVE_DOWNLOAD, LM_STUDIO_NATIVE_DOWNLOAD_STATUS};
use crate::error::ProxyError;
use crate::http::client::handle_json_response;
use crate::logging::log_request;

use super::utils::send_status_chunk;

const DOWNLOAD_STATUS_POLL_INTERVAL_MS: u64 = 500;

#[derive(Debug, Clone, Deserialize)]
pub struct LmStudioDownloadStatus {
    #[serde(default)]
    job_id: Option<String>,
    status: String,
    #[serde(default)]
    total_size_bytes: Option<u64>,
    #[serde(default)]
    downloaded_bytes: Option<u64>,
    #[serde(default)]
    bytes_per_second: Option<f64>,
    #[serde(default)]
    estimated_completion: Option<String>,
    #[serde(default)]
    started_at: Option<String>,
    #[serde(default)]
    completed_at: Option<String>,
    #[serde(default)]
    error: Option<String>,
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

    fn is_failure(&self) -> bool {
        matches!(self.status.as_str(), "failed")
    }

    fn job_id(&self) -> Result<&str, ProxyError> {
        self.job_id.as_deref().ok_or_else(|| {
            ProxyError::internal_server_error("LM Studio download response missing job identifier")
        })
    }

    pub fn to_chunk(&self, model: &str) -> Value {
        let mut chunk = serde_json::Map::new();
        chunk.insert(
            "status".to_string(),
            Value::String(self.translated_status()),
        );
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

    pub fn into_final_response(self, model: &str) -> Result<Value, ProxyError> {
        match self.status.as_str() {
            "completed" | "already_downloaded" => {
                let mut map = serde_json::Map::new();
                map.insert("status".to_string(), Value::String("success".to_string()));
                map.insert("model".to_string(), Value::String(model.to_string()));
                map.insert("detail".to_string(), Value::String(self.status));
                if let Some(job_id) = self.job_id {
                    map.insert("job_id".to_string(), Value::String(job_id));
                }
                if let Some(total) = self.total_size_bytes {
                    map.insert("total".to_string(), Value::from(total));
                }
                if let Some(done) = self.downloaded_bytes {
                    map.insert("completed".to_string(), Value::from(done));
                }
                if let Some(done_at) = self.completed_at {
                    map.insert("completed_at".to_string(), Value::String(done_at));
                }
                Ok(Value::Object(map))
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
    let response_value = send_json_request(
        client,
        Method::POST,
        &url,
        Some(&Value::Object(payload)),
        cancellation_token,
    )
    .await?;

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
    let response_value =
        send_json_request(client, Method::GET, &url, None, cancellation_token).await?;

    serde_json::from_value(response_value).map_err(|e| {
        ProxyError::internal_server_error(&format!("invalid download status payload: {}", e))
    })
}

async fn send_json_request(
    client: &reqwest::Client,
    method: Method,
    url: &str,
    body: Option<&Value>,
    cancellation_token: CancellationToken,
) -> Result<Value, ProxyError> {
    let mut builder = client.request(method, url);
    if let Some(payload) = body {
        builder = builder.json(payload);
    }

    tokio::select! {
        response = builder.send() => {
            match response {
                Ok(resp) => handle_json_response(resp, cancellation_token).await,
                Err(err) => Err(crate::http::error::map_reqwest_error(err)),
            }
        }
        _ = cancellation_token.cancelled() => Err(ProxyError::request_cancelled()),
    }
}
