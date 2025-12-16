use std::time::Instant;

use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::constants::{ERROR_MISSING_INPUT, LM_STUDIO_NATIVE_EMBEDDINGS, LOG_PREFIX_SUCCESS};
use crate::error::ProxyError;
use crate::handlers::RequestContext;
use crate::handlers::retry::execute_request_with_retry;
use crate::handlers::transform::ResponseTransformer;
use crate::http::client::handle_json_response;
use crate::http::request::{LMStudioRequestType, build_lm_studio_request};
use crate::http::{CancellableRequest, json_response};
use crate::logging::{LogConfig, log_request, log_timed};
use crate::server::ModelResolverType;

use super::utils::{
    apply_keep_alive_ttl, extract_model_name, merge_option_maps, parse_keep_alive_seconds,
    resolve_model_target,
};

#[derive(Debug, Clone, Copy)]
pub enum EmbeddingResponseMode {
    Embed,
    LegacyEmbeddings,
}

pub async fn handle_ollama_embeddings(
    context: RequestContext<'_>,
    model_resolver: ModelResolverType,
    body: Value,
    response_mode: EmbeddingResponseMode,
    cancellation_token: CancellationToken,
    load_timeout_seconds: u64,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();
    let ollama_model_name = extract_model_name(&body, "model")?;
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

            let current_ollama_model_name = extract_model_name(&body_clone, "model")?;
            let input_value = body_clone
                .get("input")
                .or_else(|| body_clone.get("prompt"))
                .cloned()
                .ok_or_else(|| ProxyError::bad_request(ERROR_MISSING_INPUT))?;

            let (lm_studio_model_id, virtual_model_entry) = resolve_model_target(
                &context,
                &model_resolver,
                current_ollama_model_name,
                cancellation_token_clone.clone(),
            )
            .await?;
            let endpoint_url = format!("{}{}", context.lmstudio_url, LM_STUDIO_NATIVE_EMBEDDINGS);

            let merged_options_owned = merge_option_maps(
                virtual_model_entry
                    .as_ref()
                    .and_then(|entry| entry.metadata.parameters.as_ref()),
                body_clone.get("options"),
            );
            let effective_options_ref = merged_options_owned
                .as_ref()
                .or_else(|| body_clone.get("options"));

            let mut lm_request = build_lm_studio_request(
                &lm_studio_model_id,
                LMStudioRequestType::Embeddings {
                    input: &input_value,
                },
                effective_options_ref,
                None,
                None,
            );
            apply_keep_alive_ttl(&mut lm_request, keep_alive_seconds_for_request);

            let request_obj =
                CancellableRequest::new(context.client, cancellation_token_clone.clone());
            log_request("POST", &endpoint_url, Some(&lm_studio_model_id));

            let response = request_obj
                .make_request(reqwest::Method::POST, &endpoint_url, Some(lm_request))
                .await?;
            let lm_response_value =
                handle_json_response(response, cancellation_token_clone).await?;

            let ollama_response = ResponseTransformer::convert_to_ollama_embeddings(
                &lm_response_value,
                &ollama_model_name_clone,
                start_time,
                matches!(model_resolver, ModelResolverType::Native(_)),
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

    let result = execute_request_with_retry(
        &context,
        ollama_model_name,
        operation,
        true,
        load_timeout_seconds,
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
