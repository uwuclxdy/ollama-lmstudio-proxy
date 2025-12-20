use std::time::Instant;

use crate::constants::DEFAULT_STREAM_TIMEOUT_SECONDS;
use crate::error::ProxyError;
use crate::handlers::transform::ResponseTransformer;
use crate::http::client::handle_json_response;
use crate::http::json_response;
use crate::logging::{LogConfig, log_handler_io};
use crate::server::ModelResolverType;
use crate::streaming::handle_streaming_response;
use tokio_util::sync::CancellationToken;

pub enum ResponseContext {
    Chat { message_count: usize },
    Generate { prompt: String },
}

pub struct ResponseParams<'a> {
    pub response: reqwest::Response,
    pub stream: bool,
    pub is_chat: bool,
    pub model_name: &'a str,
    pub start_time: Instant,
    pub context: ResponseContext,
    pub model_resolver: &'a ModelResolverType,
    pub cancellation_token: CancellationToken,
}

pub async fn handle_response(
    params: ResponseParams<'_>,
) -> Result<warp::reply::Response, ProxyError> {
    let ResponseParams {
        response,
        stream,
        is_chat,
        model_name,
        start_time,
        context,
        model_resolver,
        cancellation_token,
    } = params;

    if stream {
        handle_streaming_response(
            response,
            is_chat,
            model_name,
            start_time,
            cancellation_token,
            DEFAULT_STREAM_TIMEOUT_SECONDS,
        )
        .await
    } else {
        let lm_response_value = handle_json_response(response, cancellation_token).await?;
        let use_native_stats = matches!(model_resolver, ModelResolverType::Native(_));

        let ollama_response = match context {
            ResponseContext::Chat { message_count } => ResponseTransformer::convert_to_ollama_chat(
                &lm_response_value,
                model_name,
                message_count,
                start_time,
                use_native_stats,
            ),
            ResponseContext::Generate { prompt } => {
                ResponseTransformer::convert_to_ollama_generate(
                    &lm_response_value,
                    model_name,
                    &prompt,
                    start_time,
                    use_native_stats,
                )
            }
        };

        if LogConfig::get().debug_enabled {
            log::debug!(
                "{} response: {}",
                if is_chat { "chat" } else { "generate" },
                serde_json::to_string_pretty(&ollama_response).unwrap_or_default()
            );
        }

        // Log the handler I/O
        log_handler_io(
            if is_chat { "chat" } else { "generate" },
            None,
            Some(&ollama_response),
            false,
        );

        Ok(json_response(&ollama_response))
    }
}
