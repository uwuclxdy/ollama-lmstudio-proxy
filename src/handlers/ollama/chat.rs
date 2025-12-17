use std::borrow::Cow;
use std::time::Instant;

use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::constants::{ERROR_MISSING_MESSAGES, LM_STUDIO_NATIVE_CHAT, LOG_PREFIX_SUCCESS};
use crate::error::ProxyError;
use crate::handlers::RequestContext;
use crate::handlers::retry::execute_request_with_retry;
use crate::handlers::transform::ResponseTransformer;
use crate::http::client::handle_json_response;
use crate::http::json_response;
use crate::http::request::LMStudioRequestType;
use crate::logging::{LogConfig, log_timed};
use crate::server::ModelResolverType;
use crate::streaming::handle_streaming_response;

use super::utils::{
    LMStudioRequestParams, execute_lmstudio_request, extract_model_name, normalize_chat_messages,
    parse_keep_alive_seconds, resolve_model_with_context,
};

pub async fn handle_ollama_chat(
    context: RequestContext<'_>,
    model_resolver: ModelResolverType,
    body: Value,
    cancellation_token: CancellationToken,
    load_timeout_seconds: u64,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();
    let ollama_model_name = extract_model_name(&body, "model")?;
    let keep_alive_seconds = parse_keep_alive_seconds(body.get("keep_alive"))?;

    let resolved_model_id_for_retry = ollama_model_name.to_string();

    let operation = || {
        let context = context.clone();
        let model_resolver = model_resolver.clone();
        let body_clone = body.clone();
        let cancellation_token_clone = cancellation_token.clone();
        let ollama_model_name_clone = ollama_model_name.to_string();
        let keep_alive_seconds_for_request = keep_alive_seconds;

        async move {
            if LogConfig::get().debug_enabled {
                log::debug!(
                    "chat request: {}",
                    serde_json::to_string_pretty(&body_clone).unwrap_or_default()
                );
            }

            let current_ollama_model_name = extract_model_name(&body_clone, "model")?;

            let messages = body_clone
                .get("messages")
                .and_then(|m| m.as_array())
                .ok_or_else(|| ProxyError::bad_request(ERROR_MISSING_MESSAGES))?;

            let stream = body_clone
                .get("stream")
                .and_then(|s| s.as_bool())
                .unwrap_or(false);

            let ollama_tools = body_clone.get("tools");
            let ollama_images = body_clone.get("images");

            let resolution_ctx = resolve_model_with_context(
                &context,
                &model_resolver,
                current_ollama_model_name,
                &body_clone,
                cancellation_token_clone.clone(),
            )
            .await?;

            let message_count = messages.len();

            let normalized_messages =
                normalize_chat_messages(messages, resolution_ctx.system_prompt.as_deref());
            let messages_with_images = if let Some(images) = ollama_images {
                inject_images_into_messages(normalized_messages, images)
            } else {
                normalized_messages
            };

            let response = execute_lmstudio_request(
                &context,
                LMStudioRequestParams {
                    endpoint: LM_STUDIO_NATIVE_CHAT,
                    model_id: &resolution_ctx.lm_studio_model_id,
                    request_type: LMStudioRequestType::Chat {
                        messages: &messages_with_images,
                        stream,
                    },
                    options: resolution_ctx.effective_options.as_ref(),
                    tools: ollama_tools,
                    format: resolution_ctx.effective_format.as_ref(),
                    keep_alive: keep_alive_seconds_for_request,
                },
                cancellation_token_clone.clone(),
            )
            .await?;

            if stream {
                handle_streaming_response(
                    response,
                    true,
                    &ollama_model_name_clone,
                    start_time,
                    cancellation_token_clone.clone(),
                    60,
                )
                .await
            } else {
                let lm_response_value =
                    handle_json_response(response, cancellation_token_clone).await?;
                let ollama_response = ResponseTransformer::convert_to_ollama_chat(
                    &lm_response_value,
                    &ollama_model_name_clone,
                    message_count,
                    start_time,
                    matches!(model_resolver, ModelResolverType::Native(_)),
                );
                if LogConfig::get().debug_enabled {
                    log::debug!(
                        "chat response: {}",
                        serde_json::to_string_pretty(&ollama_response).unwrap_or_default()
                    );
                }
                Ok(json_response(&ollama_response))
            }
        }
    };

    let result = execute_request_with_retry(
        &context,
        &resolved_model_id_for_retry,
        operation,
        true,
        load_timeout_seconds,
        cancellation_token.clone(),
    )
    .await?;

    log_timed(LOG_PREFIX_SUCCESS, "Ollama chat", start_time);
    Ok(result)
}

fn inject_images_into_messages(messages: Value, images: &Value) -> Value {
    let Some(image_array) = images.as_array() else {
        return messages;
    };
    if image_array.is_empty() {
        return messages;
    }

    let Some(msg_array) = messages.as_array() else {
        return messages;
    };

    let image_parts: Vec<Value> = image_array
        .iter()
        .filter_map(|img| {
            img.as_str().map(|base64_data| {
                json!({
                    "type": "image_url",
                    "image_url": {
                        "url": format!("data:image/jpeg;base64,{}", base64_data)
                    }
                })
            })
        })
        .collect();

    if image_parts.is_empty() {
        return messages;
    }

    let mut updated = msg_array.clone();
    if let Some(last_msg) = updated.last_mut()
        && let Some(obj) = last_msg.as_object_mut()
        && let Some(content) = obj.get("content")
    {
        let text_part = json!({
            "type": "text",
            "text": content.as_str().map(Cow::Borrowed).unwrap_or(Cow::Owned(content.to_string()))
        });
        let mut parts = vec![text_part];
        parts.extend(image_parts);
        obj.insert("content".to_string(), Value::Array(parts));
    }

    Value::Array(updated)
}
