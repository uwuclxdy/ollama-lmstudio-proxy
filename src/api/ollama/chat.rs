use std::sync::Arc;
use std::time::Instant;

use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::api::RequestContext;
use crate::api::pipeline::ChatLikeCall;
use crate::api::response::{ResponseContext, ResponseParams, handle_response};
use crate::constants::{ERROR_MISSING_MESSAGES, LM_STUDIO_NATIVE_CHAT};
use crate::error::ProxyError;
use crate::http::client::CancellableRequest;
use crate::lmstudio::images::{convert_per_message_images, inject_images_into_messages};
use crate::lmstudio::keep_alive::{apply_keep_alive_ttl, parse_keep_alive_seconds};
use crate::lmstudio::request::{LMStudioRequestType, build_lm_studio_request};
use crate::lmstudio::response::normalize_chat_messages;
use crate::logging::LogConfig;
use crate::model::ModelResolver;
use crate::model::naming::extract_required_model_name;

use super::resolution::{make_top_level_params, resolve_model_with_context};

pub async fn handle_ollama_chat(
    context: RequestContext<'_>,
    model_resolver: Arc<ModelResolver>,
    body: Value,
    cancellation_token: CancellationToken,
    load_timeout_seconds: u64,
) -> Result<axum::response::Response, ProxyError> {
    let start_time = Instant::now();
    let ollama_model_name = extract_required_model_name(&body)?.to_string();
    let keep_alive_seconds = parse_keep_alive_seconds(body.get("keep_alive"))?;

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

                let message_count = messages.len();

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
