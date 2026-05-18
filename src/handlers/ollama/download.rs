use bytes::Bytes;
use reqwest::Method;
use serde_json::Value;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;

use crate::constants::{LM_STUDIO_NATIVE_DOWNLOAD, LM_STUDIO_NATIVE_DOWNLOAD_STATUS};
use crate::error::ProxyError;
use crate::http::client::{CancellableRequest, handle_json_response};
use crate::logging::log_request;

use super::download_status::LmStudioDownloadStatus;
use super::utils::send_status_chunk;

const DOWNLOAD_STATUS_POLL_INTERVAL_MS: u64 = 500;

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
        .make_request_with_response(Method::POST, &url, Some(Value::Object(payload)))
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
        .make_request_with_response(Method::GET, &url, Option::<&Value>::None)
        .await?;

    let response_value = handle_json_response(response, request.token().clone()).await?;

    serde_json::from_value(response_value).map_err(|e| {
        ProxyError::internal_server_error(&format!("invalid download status payload: {}", e))
    })
}
