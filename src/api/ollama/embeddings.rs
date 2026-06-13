use std::sync::Arc;
use std::time::Instant;

use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::api::RequestContext;
use crate::api::pipeline::ChatLikeCall;
use crate::constants::{
    ERROR_EMBED_INPUT_EMPTY, ERROR_EMBED_INPUT_REQUIRED, ERROR_EMBEDDINGS_PROMPT_EMPTY,
    ERROR_EMBEDDINGS_PROMPT_REQUIRED, LM_STUDIO_NATIVE_EMBEDDINGS,
};
use crate::error::ProxyError;
use crate::http::client::{CancellableRequest, handle_json_response};
use crate::http::json_response;
use crate::lmstudio::keep_alive::{apply_keep_alive_ttl, parse_keep_alive_seconds};
use crate::lmstudio::request::{LMStudioRequestType, build_lm_studio_request};
use crate::lmstudio::response::ResponseTransformer;
use crate::logging::LogConfig;
use crate::model::ModelResolver;
use crate::model::naming::extract_required_model_name;

use super::resolution::resolve_model_with_context;

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
                        "embeddings request: {}",
                        serde_json::to_string_pretty(&body).unwrap_or_default()
                    );
                }

                let input_value = extract_embedding_input(&body, response_mode)?;

                let resolution_ctx = resolve_model_with_context(
                    &context,
                    &model_resolver,
                    &ollama_model_name,
                    &body,
                    cancellation_token.clone(),
                )
                .await?;

                let mut lm_request = build_lm_studio_request(
                    &resolution_ctx.lm_studio_model_id,
                    LMStudioRequestType::Embeddings {
                        input: &input_value,
                    },
                    resolution_ctx.effective_options.as_ref(),
                    None,
                    None,
                    None,
                );

                apply_keep_alive_ttl(&mut lm_request, keep_alive_seconds);

                let response = CancellableRequest::new(context.client, cancellation_token.clone())
                    .make_request(
                        reqwest::Method::POST,
                        &context.endpoint_url(LM_STUDIO_NATIVE_EMBEDDINGS),
                        Some(lm_request),
                    )
                    .await?;
                let lm_response_value = handle_json_response(response, cancellation_token).await?;

                let ollama_response = ResponseTransformer::convert_to_ollama_embeddings(
                    &lm_response_value,
                    &ollama_model_name,
                    start_time,
                );
                let final_payload = finalize_embedding_response(ollama_response, response_mode);
                if LogConfig::get().debug_enabled {
                    log::debug!(
                        "embeddings response: {}",
                        serde_json::to_string_pretty(&final_payload).unwrap_or_default()
                    );
                }
                Ok(json_response(&final_payload))
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
        op_label: "Ollama embeddings",
        spawn_unload: true,
    }
    .run(operation)
    .await
}

/// Extract the embedding input value from the request body, gated by endpoint mode.
///
/// `/api/embed` (Embed) requires `input` (string or string[]) and rejects the
/// legacy `prompt` field. `/api/embeddings` (LegacyEmbeddings) requires `prompt`
/// (string) and rejects the new `input` field. Empty values — empty string,
/// empty array, or an array of only empty strings — are rejected with 400.
fn extract_embedding_input(body: &Value, mode: EmbeddingResponseMode) -> Result<Value, ProxyError> {
    match mode {
        EmbeddingResponseMode::Embed => {
            if body.get("prompt").is_some() && body.get("input").is_none() {
                return Err(ProxyError::bad_request(ERROR_EMBED_INPUT_REQUIRED));
            }
            let input = body
                .get("input")
                .cloned()
                .ok_or_else(|| ProxyError::bad_request(ERROR_EMBED_INPUT_REQUIRED))?;
            if is_empty_embedding_input(&input) {
                return Err(ProxyError::bad_request(ERROR_EMBED_INPUT_EMPTY));
            }
            Ok(input)
        }
        EmbeddingResponseMode::LegacyEmbeddings => {
            if body.get("input").is_some() && body.get("prompt").is_none() {
                return Err(ProxyError::bad_request(ERROR_EMBEDDINGS_PROMPT_REQUIRED));
            }
            let prompt = body
                .get("prompt")
                .cloned()
                .ok_or_else(|| ProxyError::bad_request(ERROR_EMBEDDINGS_PROMPT_REQUIRED))?;
            if is_empty_embedding_input(&prompt) {
                return Err(ProxyError::bad_request(ERROR_EMBEDDINGS_PROMPT_EMPTY));
            }
            Ok(prompt)
        }
    }
}

/// True when the value is an empty string, an empty array, or an array whose
/// every string element is empty. Non-string values are treated as non-empty
/// so callers can let upstream surface a typed error.
fn is_empty_embedding_input(value: &Value) -> bool {
    match value {
        Value::String(s) => s.is_empty(),
        Value::Array(items) => {
            items.is_empty()
                || items
                    .iter()
                    .all(|item| item.as_str().is_some_and(str::is_empty))
        }
        _ => false,
    }
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
/// Per Ollama spec (api-docs/ollama.md §"Generate Embeddings"), `truncate` and
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
