use std::time::Instant;

use serde_json::json;

use crate::error::ProxyError;
use crate::http::client::handle_json_response;
use crate::http::json_response;
use crate::lmstudio::response::{
    ResponseTransformer, estimate_token_count, estimate_tokens_from_bytes,
};
use crate::logging::log_handler_io;
use crate::streaming::handle_streaming_response;
use crate::streaming::sse::CompatFinalizeOptions;
use tokio_util::sync::CancellationToken;

#[derive(Debug)]
pub enum ResponseContext {
    Chat {
        message_count: usize,
        /// Total request text length, for the streaming prompt-token estimate
        /// (streaming has no usage block to read a real count from).
        input_chars: usize,
    },
    Generate {
        prompt: String,
    },
}

#[derive(Debug)]
pub struct ResponseParams<'a> {
    pub response: reqwest::Response,
    pub stream: bool,
    pub is_chat: bool,
    pub model_name: &'a str,
    pub start_time: Instant,
    pub context: ResponseContext,
    pub cancellation_token: CancellationToken,
    pub stream_timeout_seconds: u64,
    /// Client-facing dropped-field notice; the non-streaming body and the
    /// streaming final chunk carry it, matching the native path.
    pub warning: Option<String>,
}

pub async fn handle_response(
    params: ResponseParams<'_>,
) -> Result<axum::response::Response, ProxyError> {
    let ResponseParams {
        response,
        stream,
        is_chat,
        model_name,
        start_time,
        context,
        cancellation_token,
        stream_timeout_seconds,
        warning,
    } = params;

    if stream {
        let prompt_tokens_estimate = match &context {
            ResponseContext::Chat { input_chars, .. } => estimate_tokens_from_bytes(*input_chars),
            ResponseContext::Generate { prompt } => estimate_token_count(prompt),
        };
        handle_streaming_response(
            response,
            is_chat,
            model_name,
            start_time,
            cancellation_token,
            stream_timeout_seconds,
            CompatFinalizeOptions {
                prompt_tokens_estimate,
                warning,
            },
        )
        .await
    } else {
        let lm_response_value = handle_json_response(response, cancellation_token).await?;

        let mut ollama_response = match context {
            ResponseContext::Chat { message_count, .. } => {
                ResponseTransformer::convert_to_ollama_chat(
                    &lm_response_value,
                    model_name,
                    message_count,
                    start_time,
                )
            }
            ResponseContext::Generate { prompt } => {
                ResponseTransformer::convert_to_ollama_generate(
                    &lm_response_value,
                    model_name,
                    &prompt,
                    start_time,
                )
            }
        };

        if let Some(warning) = warning.as_deref()
            && let Some(obj) = ollama_response.as_object_mut()
        {
            obj.insert("warning".to_string(), json!(warning));
        }

        log_handler_io(
            if is_chat { "chat" } else { "generate" },
            None,
            Some(&ollama_response),
        );

        Ok(json_response(&ollama_response))
    }
}

#[cfg(test)]
#[path = "../../tests/unit/handlers_response.rs"]
mod tests;
