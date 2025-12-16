use std::time::{Duration, Instant};

use bytes::Bytes;
use reqwest::Method;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;

use crate::constants::{
    LM_STUDIO_NATIVE_DOWNLOAD, LM_STUDIO_NATIVE_DOWNLOAD_STATUS, LOG_PREFIX_SUCCESS,
};
use crate::error::ProxyError;
use crate::handlers::RequestContext;
use crate::http::client::handle_json_response;
use crate::http::json_response;
use crate::logging::{LogConfig, log_request, log_timed};
use crate::server::ModelResolverType;
use crate::storage::VirtualModelMetadata;
use crate::streaming::create_ndjson_stream_response;

use super::utils::{
    build_virtual_model_response, create_virtual_model_alias, determine_download_identifier,
    extract_model_name, looks_like_remote_identifier, resolve_model_target, send_status_chunk,
    send_status_error_chunk, stream_status_messages,
};

const DOWNLOAD_STATUS_POLL_INTERVAL_MS: u64 = 500;

pub async fn handle_ollama_pull(
    context: RequestContext<'_>,
    model_resolver: ModelResolverType,
    body: Value,
    cancellation_token: CancellationToken,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();
    if LogConfig::get().debug_enabled {
        log::debug!(
            "pull request: {}",
            serde_json::to_string_pretty(&body).unwrap_or_default()
        );
    }
    let requested_model = extract_model_name(&body, "model")?;
    let stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(true);
    let quantization = body
        .get("quantization")
        .and_then(|q| q.as_str())
        .map(|s| s.to_string());
    let source_override = body.get("source").and_then(|s| s.as_str());

    log_request("POST", "/api/pull", Some(requested_model));

    let client = context.client.clone();
    let base_url = context.lmstudio_url.to_string();

    let resolved_model_context =
        if source_override.is_none() && !looks_like_remote_identifier(requested_model) {
            Some(
                resolve_model_target(
                    &context,
                    &model_resolver,
                    requested_model,
                    cancellation_token.clone(),
                )
                .await?,
            )
        } else {
            None
        };

    let download_identifier = determine_download_identifier(
        &context,
        &model_resolver,
        requested_model,
        source_override,
        resolved_model_context,
        cancellation_token.clone(),
    )
    .await?;

    let initial_status = initiate_lmstudio_download(
        &client,
        &base_url,
        &download_identifier,
        quantization.as_deref(),
        cancellation_token.clone(),
    )
    .await?;

    if !stream {
        let final_status = if initial_status.is_terminal() {
            initial_status
        } else {
            wait_for_download_completion(
                &client,
                &base_url,
                initial_status,
                cancellation_token.clone(),
            )
            .await?
        };

        let response_body = final_status.into_final_response(requested_model)?;
        log_timed(LOG_PREFIX_SUCCESS, "Ollama pull", start_time);
        if LogConfig::get().debug_enabled {
            log::debug!(
                "pull response: {}",
                serde_json::to_string_pretty(&response_body).unwrap_or_default()
            );
        }
        return Ok(json_response(&response_body));
    }

    let (tx, rx) = mpsc::unbounded_channel();
    let stream_client = client.clone();
    let stream_base_url = base_url.clone();
    let model_for_stream = requested_model.to_string();
    let token_for_stream = cancellation_token.clone();

    tokio::spawn(async move {
        let progress_sender = tx.clone();
        if let Err(e) = stream_download_status_updates(
            stream_client,
            stream_base_url,
            initial_status,
            model_for_stream.clone(),
            token_for_stream,
            progress_sender,
        )
        .await
        {
            log::error!("Ollama pull stream: {}", e.message);
            send_status_error_chunk(&tx, &model_for_stream, &e.message);
        }
    });

    let response = create_ndjson_stream_response(rx, "failed to create pull streaming response")?;
    log_timed(LOG_PREFIX_SUCCESS, "Ollama pull stream open", start_time);
    if LogConfig::get().debug_enabled {
        log::debug!("pull response: (streaming)");
    }
    Ok(response)
}

pub async fn handle_ollama_create(
    context: RequestContext<'_>,
    model_resolver: ModelResolverType,
    body: Value,
    cancellation_token: CancellationToken,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();
    if LogConfig::get().debug_enabled {
        log::debug!(
            "create request: {}",
            serde_json::to_string_pretty(&body).unwrap_or_default()
        );
    }
    let new_model_name = extract_model_name(&body, "model")?;
    let source_model_name = body
        .get("from")
        .and_then(|value| value.as_str())
        .unwrap_or(new_model_name);
    let stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(true);

    log_request("POST", "/api/create", Some(new_model_name));

    let entry = create_virtual_model_alias(
        &context,
        &model_resolver,
        new_model_name,
        source_model_name,
        &body,
        cancellation_token,
    )
    .await?;

    log_timed(LOG_PREFIX_SUCCESS, "Ollama create", start_time);

    if stream {
        let statuses = vec![
            json!({"status": "reading model metadata", "model": new_model_name}),
            json!({
                "status": "creating alias",
                "model": new_model_name,
                "source": source_model_name,
                "target_model_id": entry.target_model_id
            }),
            json!({"status": "writing manifest", "model": new_model_name}),
            json!({"status": "success", "model": new_model_name, "virtual": true}),
        ];
        if LogConfig::get().debug_enabled {
            log::debug!("create response: (streaming)");
        }
        return stream_status_messages(statuses, "failed to create model alias stream");
    }

    let response = build_virtual_model_response(&entry);
    if LogConfig::get().debug_enabled {
        log::debug!(
            "create response: {}",
            serde_json::to_string_pretty(&response).unwrap_or_default()
        );
    }
    Ok(json_response(&response))
}

pub async fn handle_ollama_copy(
    context: RequestContext<'_>,
    model_resolver: ModelResolverType,
    body: Value,
    cancellation_token: CancellationToken,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();
    if LogConfig::get().debug_enabled {
        log::debug!(
            "copy request: {}",
            serde_json::to_string_pretty(&body).unwrap_or_default()
        );
    }
    let source = body
        .get("source")
        .and_then(|value| value.as_str())
        .ok_or_else(|| ProxyError::bad_request("missing 'source' field"))?;
    let destination = body
        .get("destination")
        .and_then(|value| value.as_str())
        .ok_or_else(|| ProxyError::bad_request("missing 'destination' field"))?;

    log_request("POST", "/api/copy", Some(destination));

    let entry = if let Some(existing) = context.virtual_models.get(source).await {
        context
            .virtual_models
            .create_alias(
                destination,
                existing.source_model.clone(),
                existing.target_model_id.clone(),
                existing.metadata.clone(),
            )
            .await?
    } else {
        let (resolved_id, _) =
            resolve_model_target(&context, &model_resolver, source, cancellation_token).await?;

        context
            .virtual_models
            .create_alias(
                destination,
                source.to_string(),
                resolved_id,
                VirtualModelMetadata::default(),
            )
            .await?
    };

    log_timed(LOG_PREFIX_SUCCESS, "Ollama copy", start_time);
    let response = build_virtual_model_response(&entry);
    if LogConfig::get().debug_enabled {
        log::debug!(
            "copy response: {}",
            serde_json::to_string_pretty(&response).unwrap_or_default()
        );
    }
    Ok(json_response(&response))
}

pub async fn handle_ollama_delete(
    context: RequestContext<'_>,
    body: Value,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();
    if LogConfig::get().debug_enabled {
        log::debug!(
            "delete request: {}",
            serde_json::to_string_pretty(&body).unwrap_or_default()
        );
    }
    let model_name = extract_model_name(&body, "model")?;
    log_request("DELETE", "/api/delete", Some(model_name));

    if context.virtual_models.get(model_name).await.is_none() {
        return Err(ProxyError::not_found(&format!(
            "model '{}' not managed by this proxy",
            model_name
        )));
    }

    let removed = context.virtual_models.delete(model_name).await?;
    log_timed(LOG_PREFIX_SUCCESS, "Ollama delete", start_time);
    let response = json!({
        "status": "success",
        "model": removed.name,
        "virtual": true
    });
    if LogConfig::get().debug_enabled {
        log::debug!(
            "delete response: {}",
            serde_json::to_string_pretty(&response).unwrap_or_default()
        );
    }
    Ok(json_response(&response))
}

pub async fn handle_ollama_push(
    context: RequestContext<'_>,
    model_resolver: ModelResolverType,
    body: Value,
    cancellation_token: CancellationToken,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();
    if LogConfig::get().debug_enabled {
        log::debug!(
            "push request: {}",
            serde_json::to_string_pretty(&body).unwrap_or_default()
        );
    }
    let model_name = extract_model_name(&body, "model")?;
    let stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(true);

    log_request("POST", "/api/push", Some(model_name));

    let (resolved_model_id, _) =
        resolve_model_target(&context, &model_resolver, model_name, cancellation_token).await?;

    log_timed(LOG_PREFIX_SUCCESS, "Ollama push (noop)", start_time);

    if stream {
        let statuses = vec![
            json!({"status": "retrieving manifest", "model": model_name}),
            json!({
                "status": "starting upload",
                "model": model_name,
                "target_model_id": resolved_model_id
            }),
            json!({"status": "pushing manifest", "model": model_name}),
            json!({
                "status": "success",
                "model": model_name,
                "detail": "push is a no-op when targeting LM Studio"
            }),
        ];
        if LogConfig::get().debug_enabled {
            log::debug!("push response: (streaming)");
        }
        return stream_status_messages(statuses, "failed to stream push status");
    }

    let response = json!({
        "status": "success",
        "model": model_name,
        "detail": "push is a no-op when targeting LM Studio",
        "target_model_id": resolved_model_id
    });
    if LogConfig::get().debug_enabled {
        log::debug!(
            "push response: {}",
            serde_json::to_string_pretty(&response).unwrap_or_default()
        );
    }
    Ok(json_response(&response))
}

#[derive(Debug, Clone, Deserialize)]
struct LmStudioDownloadStatus {
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

    fn is_terminal(&self) -> bool {
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

    fn to_chunk(&self, model: &str) -> Value {
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

    fn into_final_response(self, model: &str) -> Result<Value, ProxyError> {
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

async fn initiate_lmstudio_download(
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

async fn wait_for_download_completion(
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
        status = fetch_lmstudio_download_status_with_client(
            client,
            base_url,
            &job_id,
            cancellation_token.clone(),
        )
        .await?;
    }
    Ok(status)
}

async fn stream_download_status_updates(
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

        status = fetch_lmstudio_download_status_with_client(
            &client,
            &base_url,
            &job_id,
            cancellation_token.clone(),
        )
        .await?;
    }
}

async fn fetch_lmstudio_download_status_with_client(
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
                Err(err) => {
                    if err.is_connect() {
                        Err(ProxyError::lm_studio_unavailable(crate::constants::ERROR_LM_STUDIO_UNAVAILABLE))
                    } else if err.is_timeout() {
                        Err(ProxyError::lm_studio_unavailable(crate::constants::ERROR_TIMEOUT))
                    } else {
                        Err(ProxyError::internal_server_error(&format!(
                            "LM Studio request failed: {}",
                            err
                        )))
                    }
                }
            }
        }
        _ = cancellation_token.cancelled() => Err(ProxyError::request_cancelled()),
    }
}
