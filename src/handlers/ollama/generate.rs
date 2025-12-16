use std::borrow::Cow;
use std::time::Instant;

use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::constants::{
    ERROR_MISSING_PROMPT, LM_STUDIO_NATIVE_CHAT, LM_STUDIO_NATIVE_COMPLETIONS, LOG_PREFIX_SUCCESS,
};
use crate::error::ProxyError;
use crate::handlers::RequestContext;
use crate::handlers::retry::execute_request_with_retry;
use crate::handlers::transform::{ResponseTransformer, extract_system_prompt};
use crate::http::client::handle_json_response;
use crate::http::request::{LMStudioRequestType, build_lm_studio_request};
use crate::http::{CancellableRequest, json_response};
use crate::logging::{LogConfig, log_request, log_timed};
use crate::server::ModelResolverType;
use crate::streaming::handle_streaming_response;

use super::utils::{
    apply_keep_alive_ttl, extract_model_name, merge_option_maps, parse_keep_alive_seconds,
    resolve_model_target,
};

pub async fn handle_ollama_generate(
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
                    "generate request: {}",
                    serde_json::to_string_pretty(&body_clone).unwrap_or_default()
                );
            }

            let current_ollama_model_name = extract_model_name(&body_clone, "model")?;
            let current_prompt = body_clone
                .get("prompt")
                .and_then(|p| p.as_str())
                .ok_or_else(|| ProxyError::bad_request(ERROR_MISSING_PROMPT))?;

            let stream = body_clone
                .get("stream")
                .and_then(|s| s.as_bool())
                .unwrap_or(false);

            let ollama_options = body_clone.get("options");
            let current_images = body_clone.get("images");
            let structured_format = body_clone.get("format");

            let (lm_studio_model_id, virtual_model_entry) = resolve_model_target(
                &context,
                &model_resolver,
                current_ollama_model_name,
                cancellation_token_clone.clone(),
            )
            .await?;

            let merged_options_owned = merge_option_maps(
                virtual_model_entry
                    .as_ref()
                    .and_then(|entry| entry.metadata.parameters.as_ref()),
                ollama_options,
            );
            let effective_options_value = merged_options_owned.clone();

            let merged_format_owned = virtual_model_entry
                .as_ref()
                .and_then(|entry| entry.metadata.parameters.as_ref())
                .and_then(|params| params.get("format"))
                .cloned()
                .or_else(|| structured_format.cloned());
            let effective_format_value = merged_format_owned;

            let system_from_body = extract_system_prompt(&body_clone);
            let system_from_virtual = virtual_model_entry
                .as_ref()
                .and_then(|entry| entry.metadata.system_prompt.clone());
            let applied_system_prompt = system_from_body.or(system_from_virtual);

            let endpoint_url_base = context.lmstudio_url;
            let mut prompt_for_estimation = current_prompt;
            let mut prompt_override_storage: Option<String> = None;
            let mut chat_messages_payload: Option<Value> = None;

            let (lm_studio_target_url, lm_request_type) = if current_images.is_some() {
                let mut message_list = Vec::new();
                if let Some(system_text) = applied_system_prompt.as_deref() {
                    message_list.push(json!({
                        "role": "system",
                        "content": system_text,
                    }));
                }

                let mut user_message = json!({
                    "role": "user",
                    "content": current_prompt,
                });
                if let Some(images_val) = current_images
                    && let Some(obj) = user_message.as_object_mut()
                {
                    obj.insert("images".to_string(), images_val.clone());
                }
                message_list.push(user_message);
                chat_messages_payload = Some(Value::Array(message_list));
                let messages_ref = chat_messages_payload.as_ref().unwrap();

                let chat_endpoint = LM_STUDIO_NATIVE_CHAT;
                (
                    format!("{}{}", endpoint_url_base, chat_endpoint),
                    LMStudioRequestType::Chat {
                        messages: messages_ref,
                        stream,
                    },
                )
            } else {
                if let Some(system_text) = applied_system_prompt.as_deref() {
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

                let completions_endpoint = LM_STUDIO_NATIVE_COMPLETIONS;
                (
                    format!("{}{}", endpoint_url_base, completions_endpoint),
                    LMStudioRequestType::Completion {
                        prompt: Cow::Borrowed(effective_prompt),
                        stream,
                    },
                )
            };

            let _ = &chat_messages_payload;

            let effective_options_ref: Option<&Value> =
                effective_options_value.as_ref().or(ollama_options);
            let effective_format_ref: Option<&Value> =
                effective_format_value.as_ref().or(structured_format);

            let mut lm_request = build_lm_studio_request(
                &lm_studio_model_id,
                lm_request_type,
                effective_options_ref,
                None,
                effective_format_ref,
            );
            apply_keep_alive_ttl(&mut lm_request, keep_alive_seconds_for_request);

            let request_obj =
                CancellableRequest::new(context.client, cancellation_token_clone.clone());
            log_request("POST", &lm_studio_target_url, Some(&lm_studio_model_id));

            let response = request_obj
                .make_request(
                    reqwest::Method::POST,
                    &lm_studio_target_url,
                    Some(lm_request),
                )
                .await?;

            if stream {
                handle_streaming_response(
                    response,
                    false,
                    &ollama_model_name_clone,
                    start_time,
                    cancellation_token_clone.clone(),
                    60,
                )
                .await
            } else {
                let lm_response_value =
                    handle_json_response(response, cancellation_token_clone).await?;
                let ollama_response = ResponseTransformer::convert_to_ollama_generate(
                    &lm_response_value,
                    &ollama_model_name_clone,
                    prompt_for_estimation,
                    start_time,
                    matches!(model_resolver, ModelResolverType::Native(_)),
                );
                if LogConfig::get().debug_enabled {
                    log::debug!(
                        "generate response: {}",
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

    log_timed(LOG_PREFIX_SUCCESS, "Ollama generate", start_time);
    Ok(result)
}
