use std::time::Instant;

use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::constants::LOG_PREFIX_SUCCESS;
use crate::error::ProxyError;
use crate::handlers::RequestContext;
use crate::http::json_response;
use crate::logging::{log_request, log_timed};
use crate::model::utils::extract_required_model_name;
use crate::server::ModelResolverType;
use crate::storage::virtual_models::VirtualModelMetadata;

use super::download::{
    initiate_lmstudio_download, stream_download_status_updates, wait_for_download_completion,
};
use super::utils::{
    determine_download_identifier, looks_like_remote_identifier, resolve_model_target,
    send_status_error_chunk, stream_status_messages,
};
use crate::logging::log_handler_io;
use crate::streaming::create_ndjson_stream_response;

pub async fn handle_ollama_pull(
    context: RequestContext<'_>,
    model_resolver: ModelResolverType,
    body: Value,
    cancellation_token: CancellationToken,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();
    log_handler_io("pull", Some(&body), None, false);
    let requested_model = extract_required_model_name(&body)?;
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
        log_handler_io("pull", None, Some(&response_body), false);
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
            send_status_error_chunk(&tx, &model_for_stream, &e.message);
        }
    });

    let response = create_ndjson_stream_response(rx, "failed to create pull streaming response")?;
    log_timed(LOG_PREFIX_SUCCESS, "Ollama pull stream open", start_time);
    log_handler_io("pull", None, None, true);
    Ok(response)
}

pub async fn handle_ollama_create(
    context: RequestContext<'_>,
    model_resolver: ModelResolverType,
    body: Value,
    cancellation_token: CancellationToken,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();
    log_handler_io("create", Some(&body), None, false);
    let new_model_name = extract_required_model_name(&body)?;
    let source_model_name = body
        .get("from")
        .and_then(|value| value.as_str())
        .unwrap_or(new_model_name);
    let stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(true);

    log_request("POST", "/api/create", Some(new_model_name));

    let entry = context
        .virtual_models
        .create_from_request(
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
        log_handler_io("create", None, None, true);
        return stream_status_messages(statuses, "failed to create model alias stream");
    }

    let response = entry.to_response();
    log_handler_io("create", None, Some(&response), false);
    Ok(json_response(&response))
}

pub async fn handle_ollama_copy(
    context: RequestContext<'_>,
    model_resolver: ModelResolverType,
    body: Value,
    cancellation_token: CancellationToken,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();
    log_handler_io("copy", Some(&body), None, false);
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
    let response = entry.to_response();
    log_handler_io("copy", None, Some(&response), false);
    Ok(json_response(&response))
}

pub async fn handle_ollama_delete(
    context: RequestContext<'_>,
    body: Value,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();
    log_handler_io("delete", Some(&body), None, false);
    let model_name = extract_required_model_name(&body)?;
    log_request("DELETE", "/api/delete", Some(model_name));

    if context.virtual_models.get(model_name).await.is_none() {
        return Err(ProxyError::not_found(&format!(
            "model '{}' not managed by this proxy",
            model_name
        )));
    }

    let removed = context.virtual_models.delete(model_name).await?;
    log_timed(LOG_PREFIX_SUCCESS, "Ollama delete", start_time);
    let response = removed.to_response();
    log_handler_io("delete", None, Some(&response), false);
    Ok(json_response(&response))
}

pub async fn handle_ollama_push(
    context: RequestContext<'_>,
    model_resolver: ModelResolverType,
    body: Value,
    cancellation_token: CancellationToken,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();
    log_handler_io("push", Some(&body), None, false);
    let model_name = extract_required_model_name(&body)?;
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
        log_handler_io("push", None, None, true);
        return stream_status_messages(statuses, "failed to stream push status");
    }

    let response = json!({
        "status": "success",
        "model": model_name,
        "detail": "push is a no-op when targeting LM Studio",
        "target_model_id": resolved_model_id
    });
    log_handler_io("push", None, Some(&response), false);
    Ok(json_response(&response))
}
