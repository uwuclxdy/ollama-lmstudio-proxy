use std::time::Instant;

use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::constants::{ERROR_MISSING_INPUT, LM_STUDIO_NATIVE_EMBEDDINGS, LOG_PREFIX_SUCCESS};
use crate::error::ProxyError;
use crate::handlers::RequestContext;
use crate::handlers::retry::with_retry_and_cancellation;
use crate::handlers::transform::ResponseTransformer;
use crate::http::client::handle_json_response;
use crate::http::json_response;
use crate::http::request::LMStudioRequestType;
use crate::logging::{LogConfig, log_timed};
use crate::model::ModelResolver;
use std::sync::Arc;

use super::utils::{parse_keep_alive_seconds, resolve_model_with_context};
use crate::model::utils::extract_required_model_name;

#[derive(Debug, Clone, Copy)]
pub enum EmbeddingResponseMode {
    Embed,
    LegacyEmbeddings,
}

pub async fn handle_ollama_embeddings(
    context: RequestContext<'_>,
    model_resolver: Arc<ModelResolver>,
    body: Value,
    response_mode: EmbeddingResponseMode,
    cancellation_token: CancellationToken,
    load_timeout_seconds: u64,
) -> Result<axum::response::Response, ProxyError> {
    let start_time = Instant::now();
    let mut body = body;
    lift_embed_top_level_params(&mut body);
    let ollama_model_name = extract_required_model_name(&body)?;
    let keep_alive_seconds = parse_keep_alive_seconds(body.get("keep_alive"))?;

    let operation = || {
        let context = context.clone();
        let model_resolver = model_resolver.clone();
        let body_clone = body.clone();
        let cancellation_token_clone = cancellation_token.clone();
        let ollama_model_name_clone = ollama_model_name.to_string();
        let response_mode_for_request = response_mode;
        let keep_alive_seconds_for_request = keep_alive_seconds;

        async move {
            if LogConfig::get().debug_enabled {
                log::debug!(
                    "embeddings request: {}",
                    serde_json::to_string_pretty(&body_clone).unwrap_or_default()
                );
            }

            let current_ollama_model_name = extract_required_model_name(&body_clone)?;
            let input_value = body_clone
                .get("input")
                .or_else(|| body_clone.get("prompt"))
                .cloned()
                .ok_or_else(|| ProxyError::bad_request(ERROR_MISSING_INPUT))?;

            if let Some(s) = input_value.as_str()
                && s.is_empty()
            {
                return Err(ProxyError::bad_request(ERROR_MISSING_INPUT));
            }

            let resolution_ctx = resolve_model_with_context(
                &context,
                &model_resolver,
                current_ollama_model_name,
                &body_clone,
                cancellation_token_clone.clone(),
            )
            .await?;

            let mut lm_request = crate::http::request::build_lm_studio_request(
                &resolution_ctx.lm_studio_model_id,
                LMStudioRequestType::Embeddings {
                    input: &input_value,
                },
                resolution_ctx.effective_options.as_ref(),
                None,
                None,
                None, // TopLevelParams — not applicable for embeddings
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
                &context.endpoint_url(LM_STUDIO_NATIVE_EMBEDDINGS),
                Some(lm_request),
            )
            .await?;
            let lm_response_value =
                handle_json_response(response, cancellation_token_clone).await?;

            let ollama_response = ResponseTransformer::convert_to_ollama_embeddings(
                &lm_response_value,
                &ollama_model_name_clone,
                start_time,
            );
            let final_payload =
                finalize_embedding_response(ollama_response, response_mode_for_request);
            if LogConfig::get().debug_enabled {
                log::debug!(
                    "embeddings response: {}",
                    serde_json::to_string_pretty(&final_payload).unwrap_or_default()
                );
            }
            Ok(json_response(&final_payload))
        }
    };

    let result = with_retry_and_cancellation(
        &context,
        ollama_model_name,
        load_timeout_seconds,
        operation,
        cancellation_token.clone(),
    )
    .await?;

    log_timed(LOG_PREFIX_SUCCESS, "Ollama embeddings", start_time);
    Ok(result)
}

fn finalize_embedding_response(mut response: Value, mode: EmbeddingResponseMode) -> Value {
    if matches!(mode, EmbeddingResponseMode::LegacyEmbeddings) {
        let fallback = Value::Array(Vec::new());
        let first_vector = response
            .get("embeddings")
            .and_then(|value| value.as_array())
            .and_then(|arr| arr.first().cloned())
            .unwrap_or(fallback);

        if let Some(obj) = response.as_object_mut() {
            obj.insert("embedding".to_string(), first_vector);
            obj.remove("embeddings");
        }
    }

    response
}

/// Lift Ollama's top-level `/api/embed` advanced parameters (`truncate`, `dimensions`)
/// into the `options` map so the shared option-mapper picks them up.
///
/// Per Ollama spec (api_docs/ollama.md §"Generate Embeddings"), `truncate` and
/// `dimensions` sit at the top level of the request body, peers of `model` and
/// `input`. Values inside an existing `options` object take precedence.
pub fn lift_embed_top_level_params(body: &mut Value) {
    let Some(obj) = body.as_object_mut() else {
        return;
    };

    let truncate = obj.remove("truncate");
    let dimensions = obj.remove("dimensions");
    if truncate.is_none() && dimensions.is_none() {
        return;
    }

    let options_entry = obj
        .entry("options")
        .or_insert_with(|| serde_json::json!({}));
    let Some(options) = options_entry.as_object_mut() else {
        // `options` is set to a non-object value; restore top-level fields and bail.
        if let Some(t) = truncate {
            obj.insert("truncate".to_string(), t);
        }
        if let Some(d) = dimensions {
            obj.insert("dimensions".to_string(), d);
        }
        return;
    };

    if let Some(t) = truncate {
        options.entry("truncate".to_string()).or_insert(t);
    }
    if let Some(d) = dimensions {
        options.entry("dimensions".to_string()).or_insert(d);
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/handlers_ollama_embed_params.rs"]
mod tests_embed_params;
