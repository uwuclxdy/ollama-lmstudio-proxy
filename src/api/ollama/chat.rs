use std::sync::Arc;
use std::time::Instant;

use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::api::RequestContext;
use crate::api::pipeline::ChatLikeCall;
use crate::api::response::{ResponseContext, ResponseParams, handle_response};
use crate::config::get_runtime_config;
use crate::constants::{
    DEFAULT_STREAM_TIMEOUT_SECONDS, ERROR_MISSING_MESSAGES, LM_STUDIO_NATIVE_CHAT,
    LM_STUDIO_V1_CHAT,
};
use crate::error::ProxyError;
use crate::http::client::{CancellableRequest, handle_json_response};
use crate::http::json_response;
use crate::lmstudio::ensure_context_length;
use crate::lmstudio::images::{convert_per_message_images, inject_images_into_messages};
use crate::lmstudio::keep_alive::{apply_keep_alive_ttl, parse_keep_alive_seconds};
use crate::lmstudio::native_chat::{
    NativeChatRequestParams, build_native_chat_request, convert_native_to_ollama_chat,
};
use crate::lmstudio::request::{LMStudioRequestType, build_lm_studio_request};
use crate::lmstudio::response::normalize_chat_messages;
use crate::logging::LogConfig;
use crate::model::ModelResolver;
use crate::model::naming::extract_required_model_name;
use crate::streaming::handle_native_streaming_response;

use super::resolution::{make_top_level_params, resolve_model_with_context};
use super::unload_only::{UnloadOnlyCall, is_chat_unload_only, respond_unload_only};

pub async fn handle_ollama_chat(
    context: RequestContext<'_>,
    model_resolver: Arc<ModelResolver>,
    body: Value,
    cancellation_token: CancellationToken,
    load_timeout_seconds: u64,
    use_native_chat: bool,
) -> Result<axum::response::Response, ProxyError> {
    let start_time = Instant::now();
    let ollama_model_name = extract_required_model_name(&body)?.to_string();
    let keep_alive_seconds = parse_keep_alive_seconds(body.get("keep_alive"))?;

    // Spec: `{"model":"x","keep_alive":0}` (no/empty `messages`) is an
    // unload-only call. Short-circuit before the inference path so we don't
    // race the TTL=0 teardown against the chat completion.
    if is_chat_unload_only(&body, keep_alive_seconds) {
        let stream = body.get("stream").and_then(|s| s.as_bool()).unwrap_or(true);
        return respond_unload_only(UnloadOnlyCall {
            context: &context,
            model_resolver,
            ollama_model_name: &ollama_model_name,
            keep_alive_seconds,
            is_chat: true,
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
                        "chat request: {}",
                        serde_json::to_string_pretty(&body).unwrap_or_default()
                    );
                }

                let messages = body
                    .get("messages")
                    .and_then(|m| m.as_array())
                    .ok_or_else(|| ProxyError::bad_request(ERROR_MISSING_MESSAGES))?;

                let stream = body.get("stream").and_then(|s| s.as_bool()).unwrap_or(true);

                let ollama_tools = body.get("tools");
                let ollama_images = body.get("images");

                let resolution_ctx = resolve_model_with_context(
                    &context,
                    &model_resolver,
                    &ollama_model_name,
                    &body,
                    cancellation_token.clone(),
                )
                .await?;

                // Honor Ollama `num_ctx`: reload the model at the requested
                // context window before inference. No-op when unset or already
                // satisfied; best-effort, never fails the request.
                ensure_context_length(
                    &context,
                    &resolution_ctx.lm_studio_model_id,
                    resolution_ctx.effective_options.as_ref(),
                    get_runtime_config(),
                    &cancellation_token,
                )
                .await;

                let message_count = messages.len();

                // Native /api/v1/chat path: build the request from the raw Ollama
                // messages (the native builder owns its own `input`/image shaping)
                // and dispatch to the native converter / streaming driver.
                if use_native_chat {
                    // Only forward `integrations` on the native path; the default
                    // OpenAI-compat path never reads this field.
                    let integrations = body.get("integrations").filter(|v| v.is_array());
                    let mut native_request = build_native_chat_request(NativeChatRequestParams {
                        model_lm_studio_id: &resolution_ctx.lm_studio_model_id,
                        messages: body.get("messages").unwrap_or(&Value::Null),
                        system_prompt: resolution_ctx.system_prompt.as_deref(),
                        ollama_options: resolution_ctx.effective_options.as_ref(),
                        think: make_top_level_params(&body).think,
                        stream,
                        integrations,
                    });
                    apply_keep_alive_ttl(&mut native_request, keep_alive_seconds);

                    let response =
                        CancellableRequest::new(context.client, cancellation_token.clone())
                            .make_request(
                                reqwest::Method::POST,
                                &context.endpoint_url(LM_STUDIO_V1_CHAT),
                                Some(native_request),
                            )
                            .await?;

                    return if stream {
                        handle_native_streaming_response(
                            response,
                            &ollama_model_name,
                            start_time,
                            cancellation_token,
                            DEFAULT_STREAM_TIMEOUT_SECONDS,
                        )
                        .await
                    } else {
                        let native_value =
                            handle_json_response(response, cancellation_token).await?;
                        let ollama_response = convert_native_to_ollama_chat(
                            &native_value,
                            &ollama_model_name,
                            start_time,
                        );
                        Ok(json_response(&ollama_response))
                    };
                }

                let normalized_messages =
                    normalize_chat_messages(messages, resolution_ctx.system_prompt.as_deref());
                // /api/chat: each message may carry its own `images` array — pull those
                // into OpenAI content parts on the same message first.
                let with_per_message_images = convert_per_message_images(normalized_messages);
                // Top-level `images` (rare on /api/chat, common on vision /api/generate
                // bridged through chat) attaches to the LAST user message.
                let messages_with_images = if let Some(images) = ollama_images {
                    inject_images_into_messages(with_per_message_images, images)
                } else {
                    with_per_message_images
                };

                let top_level_params = make_top_level_params(&body);
                let mut lm_request = build_lm_studio_request(
                    &resolution_ctx.lm_studio_model_id,
                    LMStudioRequestType::Chat {
                        messages: &messages_with_images,
                        stream,
                    },
                    resolution_ctx.effective_options.as_ref(),
                    ollama_tools,
                    resolution_ctx.effective_format.as_ref(),
                    Some(&top_level_params),
                );

                // Forward OpenAI-compat `tool_choice` alongside `tools`; LM
                // Studio accepts it on /api/v0/chat/completions.
                if let Some(tool_choice) = body.get("tool_choice")
                    && let Some(obj) = lm_request.as_object_mut()
                {
                    obj.insert("tool_choice".to_string(), tool_choice.clone());
                }

                apply_keep_alive_ttl(&mut lm_request, keep_alive_seconds);

                let response = CancellableRequest::new(context.client, cancellation_token.clone())
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
                    model_name: &ollama_model_name,
                    start_time,
                    context: ResponseContext::Chat { message_count },
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
        op_label: "Ollama chat",
        spawn_unload: true,
    }
    .run(operation)
    .await
}
