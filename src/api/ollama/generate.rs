use std::borrow::Cow;
use std::sync::Arc;
use std::time::Instant;

use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::api::RequestContext;
use crate::api::pipeline::ChatLikeCall;
use crate::api::response::{ResponseContext, ResponseParams, handle_response};
use crate::constants::{ERROR_MISSING_PROMPT, LM_STUDIO_NATIVE_CHAT, LM_STUDIO_NATIVE_COMPLETIONS};
use crate::error::ProxyError;
use crate::http::client::CancellableRequest;
use crate::lmstudio::images::build_vision_chat_messages;
use crate::lmstudio::keep_alive::{apply_keep_alive_ttl, parse_keep_alive_seconds};
use crate::lmstudio::request::{LMStudioRequestType, build_lm_studio_request};
use crate::logging::LogConfig;
use crate::model::ModelResolver;
use crate::model::naming::extract_required_model_name;

use super::resolution::{make_top_level_params, resolve_model_with_context};
use super::unload_only::{UnloadOnlyCall, is_generate_unload_only, respond_unload_only};

pub async fn handle_ollama_generate(
    context: RequestContext<'_>,
    model_resolver: Arc<ModelResolver>,
    body: Value,
    cancellation_token: CancellationToken,
    load_timeout_seconds: u64,
) -> Result<axum::response::Response, ProxyError> {
    let start_time = Instant::now();
    let ollama_model_name = extract_required_model_name(&body)?.to_string();
    let keep_alive_seconds = parse_keep_alive_seconds(body.get("keep_alive"))?;

    // Spec: `{"model":"x","keep_alive":0}` (no/empty `prompt`) is an
    // unload-only call. Short-circuit before the inference path — firing the
    // request would race the TTL=0 teardown and return "Model is unloaded."
    if is_generate_unload_only(&body, keep_alive_seconds) {
        let stream = body.get("stream").and_then(|s| s.as_bool()).unwrap_or(true);
        return respond_unload_only(UnloadOnlyCall {
            context: &context,
            model_resolver,
            ollama_model_name: &ollama_model_name,
            keep_alive_seconds,
            is_chat: false,
            stream,
            start_time,
            cancellation_token,
        })
        .await;
    }

    let operation = {
        let context = context.clone();
        let model_resolver = model_resolver.clone();
        let body = body.clone();
        let cancellation_token = cancellation_token.clone();
        let ollama_model_name = ollama_model_name.clone();
        move || {
            let context = context.clone();
            let model_resolver = model_resolver.clone();
            let body = body.clone();
            let cancellation_token = cancellation_token.clone();
            let ollama_model_name = ollama_model_name.clone();
            async move {
                if LogConfig::get().debug_enabled {
                    log::debug!(
                        "generate request: {}",
                        serde_json::to_string_pretty(&body).unwrap_or_default()
                    );
                }

                let current_prompt = body
                    .get("prompt")
                    .and_then(|p| p.as_str())
                    .ok_or_else(|| ProxyError::bad_request(ERROR_MISSING_PROMPT))?;

                let stream = body.get("stream").and_then(|s| s.as_bool()).unwrap_or(true);

                let current_images = body.get("images");
                let raw = body.get("raw").and_then(|v| v.as_bool()).unwrap_or(false);

                let resolution_ctx = resolve_model_with_context(
                    &context,
                    &model_resolver,
                    &ollama_model_name,
                    &body,
                    cancellation_token.clone(),
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

                    let effective_prompt =
                        prompt_override_storage.as_deref().unwrap_or(current_prompt);

                    (
                        LM_STUDIO_NATIVE_COMPLETIONS,
                        LMStudioRequestType::Completion {
                            prompt: Cow::Borrowed(effective_prompt),
                            stream,
                        },
                    )
                };

                let top_level_params = make_top_level_params(&body);
                let suffix_val = body.get("suffix");

                if current_images.is_some() && suffix_val.is_some() {
                    log::debug!("unsupported on vision path: suffix");
                }

                let mut lm_request = build_lm_studio_request(
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

                apply_keep_alive_ttl(&mut lm_request, keep_alive_seconds);

                let response = CancellableRequest::new(context.client, cancellation_token.clone())
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
                    model_name: &ollama_model_name,
                    start_time,
                    context: ResponseContext::Generate {
                        prompt: prompt_for_estimation.to_string(),
                    },
                    cancellation_token,
                })
                .await
            }
        }
    };

    ChatLikeCall {
        context,
        resolver: model_resolver,
        body,
        cancellation: cancellation_token,
        load_timeout_seconds,
        ollama_model_name,
        keep_alive_seconds,
        start_time,
        op_label: "Ollama generate",
        spawn_unload: true,
    }
    .run(operation)
    .await
}
