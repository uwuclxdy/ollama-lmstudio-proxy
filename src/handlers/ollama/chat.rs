use std::time::Instant;

use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::constants::{ERROR_MISSING_MESSAGES, LM_STUDIO_NATIVE_CHAT, LOG_PREFIX_SUCCESS};
use crate::error::ProxyError;
use crate::handlers::RequestContext;
use crate::handlers::retry::execute_request_with_retry;
use crate::http::request::LMStudioRequestType;
use crate::logging::{LogConfig, log_timed};
use crate::server::ModelResolverType;

use super::utils::{parse_keep_alive_seconds, resolve_model_with_context};
use crate::handlers::ollama::images::inject_images_into_messages;
use crate::handlers::response::{ResponseContext, ResponseParams, handle_response};
use crate::handlers::transform::normalize_chat_messages;
use crate::model::utils::extract_required_model_name;

pub async fn handle_ollama_chat(
    context: RequestContext<'_>,
    model_resolver: ModelResolverType,
    body: Value,
    cancellation_token: CancellationToken,
    load_timeout_seconds: u64,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();
    let ollama_model_name = extract_required_model_name(&body)?;
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

            let current_ollama_model_name = extract_required_model_name(&body_clone)?;

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

            let mut lm_request = crate::http::request::build_lm_studio_request(
                &resolution_ctx.lm_studio_model_id,
                LMStudioRequestType::Chat {
                    messages: &messages_with_images,
                    stream,
                },
                resolution_ctx.effective_options.as_ref(),
                ollama_tools,
                resolution_ctx.effective_format.as_ref(),
            );

            // Apply keep-alive TTL
            crate::handlers::ollama::keep_alive::apply_keep_alive_ttl(
                &mut lm_request,
                keep_alive_seconds_for_request,
            );

            let response = crate::http::client::CancellableRequest::new(
                context.client,
                cancellation_token_clone.clone(),
            )
            .make_request(
                reqwest::Method::POST,
                &context.endpoint_url(LM_STUDIO_NATIVE_CHAT),
                Some(lm_request),
            )
            .await?;

            handle_response(ResponseParams {
                response,
                stream,
                is_chat: true,
                model_name: &ollama_model_name_clone,
                start_time,
                context: ResponseContext::Chat { message_count },
                model_resolver: &model_resolver,
                cancellation_token: cancellation_token_clone,
            })
            .await
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
