use std::borrow::Cow;
use std::time::Instant;

use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::constants::{
    ERROR_MISSING_PROMPT, LM_STUDIO_NATIVE_CHAT, LM_STUDIO_NATIVE_COMPLETIONS, LOG_PREFIX_SUCCESS,
};
use crate::error::ProxyError;
use crate::handlers::RequestContext;
use crate::handlers::retry::with_retry_and_cancellation;
use crate::http::request::LMStudioRequestType;
use crate::logging::{LogConfig, log_timed};
use crate::model::ModelResolver;
use std::sync::Arc;

use super::shared::make_top_level_params;
use super::utils::{parse_keep_alive_seconds, resolve_model_with_context};
use crate::handlers::ollama::images::build_vision_chat_messages;
use crate::handlers::response::{ResponseContext, ResponseParams, handle_response};
use crate::model::utils::extract_required_model_name;

pub async fn handle_ollama_generate(
    context: RequestContext<'_>,
    model_resolver: Arc<ModelResolver>,
    body: Value,
    cancellation_token: CancellationToken,
    load_timeout_seconds: u64,
) -> Result<axum::response::Response, ProxyError> {
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
                    "generate request: {}",
                    serde_json::to_string_pretty(&body_clone).unwrap_or_default()
                );
            }

            let current_ollama_model_name = extract_required_model_name(&body_clone)?;
            let current_prompt = body_clone
                .get("prompt")
                .and_then(|p| p.as_str())
                .ok_or_else(|| ProxyError::bad_request(ERROR_MISSING_PROMPT))?;

            let stream = body_clone
                .get("stream")
                .and_then(|s| s.as_bool())
                .unwrap_or(false);

            let current_images = body_clone.get("images");
            let raw = body_clone
                .get("raw")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            let resolution_ctx = resolve_model_with_context(
                &context,
                &model_resolver,
                current_ollama_model_name,
                &body_clone,
                cancellation_token_clone.clone(),
            )
            .await?;

            let mut prompt_for_estimation = current_prompt;
            let mut prompt_override_storage: Option<String> = None;
            let chat_messages_payload: Option<Value>;

            let (lm_studio_endpoint, lm_request_type) = if current_images.is_some() {
                let system_for_vision = if raw {
                    None
                } else {
                    resolution_ctx.system_prompt.as_deref()
                };
                chat_messages_payload = Some(build_vision_chat_messages(
                    system_for_vision,
                    current_prompt,
                    current_images,
                ));
                let messages_ref = chat_messages_payload.as_ref().unwrap();

                (
                    LM_STUDIO_NATIVE_CHAT,
                    LMStudioRequestType::Chat {
                        messages: messages_ref,
                        stream,
                    },
                )
            } else {
                if !raw && let Some(system_text) = resolution_ctx.system_prompt.as_deref() {
                    let trimmed = system_text.trim();
                    if !trimmed.is_empty() {
                        let combined = if current_prompt.is_empty() {
                            trimmed.to_string()
                        } else {
                            format!("{trimmed}\n\n{current_prompt}")
                        };
                        prompt_override_storage = Some(combined);
                    }
                }

                if let Some(override_str) = prompt_override_storage.as_deref() {
                    prompt_for_estimation = override_str;
                }

                let effective_prompt = prompt_override_storage.as_deref().unwrap_or(current_prompt);

                (
                    LM_STUDIO_NATIVE_COMPLETIONS,
                    LMStudioRequestType::Completion {
                        prompt: Cow::Borrowed(effective_prompt),
                        stream,
                    },
                )
            };

            let top_level_params = make_top_level_params(&body_clone);
            let suffix_val = body_clone.get("suffix");

            if current_images.is_some() && suffix_val.is_some() {
                log::debug!("unsupported on vision path: suffix");
            }

            let mut lm_request = crate::http::request::build_lm_studio_request(
                &resolution_ctx.lm_studio_model_id,
                lm_request_type,
                resolution_ctx.effective_options.as_ref(),
                None,
                resolution_ctx.effective_format.as_ref(),
                Some(&top_level_params),
            );

            if current_images.is_none()
                && let Some(s) = suffix_val
                && let Some(obj) = lm_request.as_object_mut()
            {
                obj.insert("suffix".to_string(), s.clone());
            }

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
                &context.endpoint_url(lm_studio_endpoint),
                Some(lm_request),
            )
            .await?;

            handle_response(ResponseParams {
                response,
                stream,
                is_chat: false,
                model_name: &ollama_model_name_clone,
                start_time,
                context: ResponseContext::Generate {
                    prompt: prompt_for_estimation.to_string(),
                },
                cancellation_token: cancellation_token_clone,
            })
            .await
        }
    };

    let result = with_retry_and_cancellation(
        &context,
        &resolved_model_id_for_retry,
        load_timeout_seconds,
        operation,
        cancellation_token.clone(),
    )
    .await?;

    let is_streaming = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let unload_delay = if is_streaming {
        crate::constants::DEFAULT_STREAM_TIMEOUT_SECONDS
    } else {
        0
    };
    super::keep_alive::spawn_model_unload_if_needed(
        context.client.clone(),
        context.lmstudio_url.to_string(),
        model_resolver.clone(),
        ollama_model_name.to_string(),
        keep_alive_seconds,
        unload_delay,
    );

    log_timed(LOG_PREFIX_SUCCESS, "Ollama generate", start_time);
    Ok(result)
}

#[cfg(test)]
#[path = "../../../tests/unit/handlers_ollama_generate.rs"]
mod tests;
