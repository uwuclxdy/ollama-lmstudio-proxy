use std::borrow::Cow;
use std::time::Instant;

use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::constants::{
    ERROR_MISSING_PROMPT, LM_STUDIO_NATIVE_CHAT, LM_STUDIO_NATIVE_COMPLETIONS, LOG_PREFIX_SUCCESS,
};
use crate::error::ProxyError;
use crate::handlers::RequestContext;
use crate::handlers::retry::execute_request_with_retry;
use crate::http::request::LMStudioRequestType;
use crate::logging::{LogConfig, log_timed};
use crate::server::ModelResolverType;

use super::utils::{parse_keep_alive_seconds, resolve_model_with_context};
use crate::handlers::ollama::images::build_vision_chat_messages;
use crate::handlers::response::{ResponseContext, ResponseParams, handle_response};
use crate::model::utils::extract_required_model_name;

pub async fn handle_ollama_generate(
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
            let mut chat_messages_payload: Option<Value> = None;

            let (lm_studio_endpoint, lm_request_type) = if current_images.is_some() {
                chat_messages_payload = Some(build_vision_chat_messages(
                    resolution_ctx.system_prompt.as_deref(),
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
                if let Some(system_text) = resolution_ctx.system_prompt.as_deref() {
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

            let _ = &chat_messages_payload;

            let mut lm_request = crate::http::request::build_lm_studio_request(
                &resolution_ctx.lm_studio_model_id,
                lm_request_type,
                resolution_ctx.effective_options.as_ref(),
                None,
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

    log_timed(LOG_PREFIX_SUCCESS, "Ollama generate", start_time);
    Ok(result)
}
