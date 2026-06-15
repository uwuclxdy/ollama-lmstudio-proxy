use std::time::Instant;

use http::StatusCode;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::api::RequestContext;
use crate::constants::LOG_PREFIX_SUCCESS;
use crate::error::ProxyError;
use crate::http::json_response;
use crate::logging::{log_request, log_timed};
use crate::model::ModelResolver;
use crate::model::naming::extract_required_model_name;
use crate::storage::VirtualModelStore;
use crate::storage::virtual_models::VirtualModelMetadata;
use std::sync::Arc;

use super::resolution::resolve_model_target;
use super::status_stream::{send_status_error_chunk, stream_status_messages};
use crate::lmstudio::download::{determine_download_identifier, looks_like_remote_identifier};
use crate::lmstudio::download::{
    initiate_lmstudio_download, stream_download_status_updates, wait_for_download_completion,
};
use crate::logging::log_handler_io;
use crate::streaming::create_ndjson_stream_response;

pub async fn handle_ollama_pull(
    context: RequestContext<'_>,
    model_resolver: Arc<ModelResolver>,
    body: Value,
    cancellation_token: CancellationToken,
) -> Result<axum::response::Response, ProxyError> {
    let start_time = Instant::now();
    log_handler_io("pull", Some(&body), None);
    let requested_model = extract_required_model_name(&body)?;
    let stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(true);
    let quantization = body
        .get("quantization")
        .and_then(|q| q.as_str())
        .map(|s| s.to_string());
    let source_override = body.get("source").and_then(|s| s.as_str());

    // `insecure` requests a TLS-bypass that LM Studio's
    // /api/v1/models/download cannot map. Real Ollama never rejects the flag,
    // so accept and ignore it rather than 400 — the download proceeds normally.

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
        log_handler_io("pull", None, Some(&response_body));
        return Ok(json_response(&response_body));
    }

    let (tx, rx) = mpsc::unbounded_channel();
    let stream_client = client.clone();
    let stream_base_url = base_url.clone();
    let model_for_stream = requested_model.to_string();
    let token_for_stream = cancellation_token.clone();

    tokio::spawn(async move {
        if let Err(e) = stream_download_status_updates(
            stream_client,
            stream_base_url,
            initial_status,
            model_for_stream.clone(),
            token_for_stream,
            tx.clone(),
        )
        .await
        {
            log::error!("Ollama pull stream: {}", e.message);
            send_status_error_chunk(&tx, &e.message);
        }
    });

    let response = create_ndjson_stream_response(rx, "failed to create pull streaming response")?;
    log_timed(LOG_PREFIX_SUCCESS, "Ollama pull stream open", start_time);
    log_handler_io("pull", None, None);
    Ok(response)
}

pub async fn handle_ollama_create(
    context: RequestContext<'_>,
    model_resolver: Arc<ModelResolver>,
    body: Value,
    cancellation_token: CancellationToken,
) -> Result<axum::response::Response, ProxyError> {
    let start_time = Instant::now();
    log_handler_io("create", Some(&body), None);
    let new_model_name = extract_required_model_name(&body)?;
    let stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(true);

    log_request("POST", "/api/create", Some(new_model_name));

    // LM Studio has no API for creating real models; the proxy implements
    // virtual aliases only. Files and quantization require real model creation
    // which is not possible upstream.
    if let Some(files) = body.get("files") {
        let has_content = match files {
            Value::Object(map) => !map.is_empty(),
            Value::Array(arr) => !arr.is_empty(),
            Value::Null => false,
            _ => true,
        };
        if has_content {
            return Err(ProxyError::bad_request(
                "creating from raw files is unsupported by the LM Studio backend (no GGUF-blob import surface)",
            ));
        }
    }

    if body.get("quantize").is_some() {
        return Err(ProxyError::bad_request(
            "quantize is unsupported by the LM Studio backend (no quantization surface)",
        ));
    }

    // `from` is required unless a system prompt or template is the only
    // customization (both still need a base model to alias). Silently
    // defaulting to `model` produces a self-referential alias that resolves
    // to itself rather than a real LM Studio model — that is a footgun.
    let source_model_name = body
        .get("from")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ProxyError::bad_request("'from' is required"))?;

    let (resolved_id, source_virtual_entry) = resolve_model_target(
        &context,
        &model_resolver,
        source_model_name,
        cancellation_token,
    )
    .await?;

    let base_metadata = source_virtual_entry.map(|entry| entry.metadata);
    let metadata = VirtualModelStore::build_metadata_from_request(&body, base_metadata);

    context
        .virtual_models
        .upsert_alias(
            new_model_name,
            source_model_name.to_string(),
            resolved_id,
            metadata,
        )
        .await?;

    log_timed(LOG_PREFIX_SUCCESS, "Ollama create", start_time);

    if stream {
        let statuses = vec![
            json!({"status": "reading model metadata"}),
            json!({"status": "creating alias"}),
            json!({"status": "writing manifest"}),
            json!({"status": "success"}),
        ];
        log_handler_io("create", None, None);
        return stream_status_messages(statuses, "failed to create model alias stream");
    }

    let response = json!({"status": "success"});
    log_handler_io("create", None, Some(&response));
    Ok(json_response(&response))
}

pub async fn handle_ollama_copy(
    context: RequestContext<'_>,
    model_resolver: Arc<ModelResolver>,
    body: Value,
    cancellation_token: CancellationToken,
) -> Result<axum::response::Response, ProxyError> {
    let start_time = Instant::now();
    log_handler_io("copy", Some(&body), None);
    // The Ollama spec declares no 400 for this endpoint, but silently accepting
    // a request with missing required fields would produce a confusing failure
    // downstream. We keep the 400 as a pragmatic guard.
    let source = body
        .get("source")
        .and_then(|value| value.as_str())
        .ok_or_else(|| ProxyError::bad_request("missing 'source' field"))?;
    let destination = body
        .get("destination")
        .and_then(|value| value.as_str())
        .ok_or_else(|| ProxyError::bad_request("missing 'destination' field"))?;

    log_request("POST", "/api/copy", Some(destination));

    if let Some(existing) = context.virtual_models.get(source).await {
        context
            .virtual_models
            .upsert_alias(
                destination,
                existing.source_model.clone(),
                existing.target_model_id.clone(),
                existing.metadata.clone(),
            )
            .await?;
    } else {
        let (resolved_id, _) =
            resolve_model_target(&context, &model_resolver, source, cancellation_token).await?;

        context
            .virtual_models
            .upsert_alias(
                destination,
                source.to_string(),
                resolved_id,
                VirtualModelMetadata::default(),
            )
            .await?;
    }

    log_timed(LOG_PREFIX_SUCCESS, "Ollama copy", start_time);
    // Ollama returns 200 with an empty body (no content block declared); the
    // alias is upserted, so copying onto an existing destination overwrites.
    log_handler_io("copy", None, None);
    axum::response::Response::builder()
        .status(StatusCode::OK)
        .body(axum::body::Body::empty())
        .map_err(|_| ProxyError::internal_server_error("failed to build copy response"))
}

pub async fn handle_ollama_delete(
    context: RequestContext<'_>,
    body: Value,
) -> Result<axum::response::Response, ProxyError> {
    let start_time = Instant::now();
    log_handler_io("delete", Some(&body), None);
    let model_name = extract_required_model_name(&body)?;
    log_request("DELETE", "/api/delete", Some(model_name));

    // Only proxy-managed virtual aliases can be deleted. Native LM Studio models
    // are not writable through this proxy, so any unknown name returns 404.
    if context.virtual_models.get(model_name).await.is_none() {
        return Err(ProxyError::not_found(&format!(
            "model '{}' cannot be deleted: only proxy-managed virtual aliases are deletable; \
             LM Studio's REST API exposes no model-file delete",
            model_name
        )));
    }

    context.virtual_models.delete(model_name).await?;
    log_timed(LOG_PREFIX_SUCCESS, "Ollama delete", start_time);
    log_handler_io("delete", None, None);
    axum::response::Response::builder()
        .status(StatusCode::OK)
        .body(axum::body::Body::empty())
        .map_err(|_| ProxyError::internal_server_error("failed to build delete response"))
}

#[cfg(test)]
#[path = "../../../tests/unit/handlers_ollama_lifecycle.rs"]
mod tests;
