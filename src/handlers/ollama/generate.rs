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
use crate::http::request::{LMStudioRequestType, TopLevelParams};
use crate::logging::{LogConfig, log_timed};
use crate::server::ModelResolverType;

use super::utils::{parse_keep_alive_seconds, resolve_model_with_context};
use crate::handlers::ollama::images::build_vision_chat_messages;
use crate::handlers::response::{ResponseContext, ResponseParams, handle_response};
use crate::model::utils::extract_required_model_name;

fn make_top_level_params(body: &serde_json::Value) -> TopLevelParams<'_> {
    TopLevelParams {
        think: body.get("think"),
        logprobs: body.get("logprobs"),
        top_logprobs: body.get("top_logprobs"),
    }
}

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
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extracts_think_from_generate_body() {
        let body = json!({ "think": "high", "model": "x", "prompt": "hi" });
        let top = make_top_level_params(&body);
        assert_eq!(top.think, Some(&json!("high")));
    }

    #[test]
    fn suffix_inserted_into_lm_request() {
        use crate::http::request::{LMStudioRequestType, TopLevelParams, build_lm_studio_request};
        use std::borrow::Cow;

        let body = json!({ "suffix": "world", "model": "test", "prompt": "hello" });
        let suffix_val = body.get("suffix");
        let top_level = TopLevelParams {
            think: None,
            logprobs: None,
            top_logprobs: None,
        };

        let mut lm_request = build_lm_studio_request(
            "test",
            LMStudioRequestType::Completion {
                prompt: Cow::Borrowed("hello"),
                stream: false,
            },
            None,
            None,
            None,
            Some(&top_level),
        );

        if let Some(s) = suffix_val
            && let Some(obj) = lm_request.as_object_mut()
        {
            obj.insert("suffix".to_string(), s.clone());
        }

        assert_eq!(lm_request.get("suffix"), Some(&json!("world")));
    }

    #[test]
    fn suffix_not_inserted_on_vision_path() {
        let body = json!({ "suffix": "world", "model": "test", "prompt": "hello",
                           "images": ["base64data"] });
        let current_images = body.get("images");
        let suffix_val = body.get("suffix");
        let mut lm_request = json!({ "model": "test" });

        if current_images.is_none()
            && let Some(s) = suffix_val
            && let Some(obj) = lm_request.as_object_mut()
        {
            obj.insert("suffix".to_string(), s.clone());
        }

        assert!(
            lm_request.get("suffix").is_none(),
            "suffix must be absent on vision path"
        );
    }

    #[test]
    fn absent_think_gives_none_in_generate() {
        let body = json!({ "model": "x", "prompt": "hi" });
        let top = make_top_level_params(&body);
        assert!(top.think.is_none());
    }
}
