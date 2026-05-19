use std::time::Instant;

use moka::future::Cache;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::constants::{ERROR_LM_STUDIO_UNAVAILABLE, LM_STUDIO_NATIVE_MODELS, LOG_PREFIX_SUCCESS};
use crate::error::ProxyError;
use crate::http::CancellableRequest;
use crate::logging::log_timed;
use crate::model::matcher::{ModelMatchView, find_best_match};
use crate::model::types::{ModelInfo, NativeModelsResponse};
use crate::model::utils::clean_model_name;

pub struct ModelResolver {
    lmstudio_url: String,
    cache: Cache<String, String>,
}

impl ModelResolver {
    pub fn new(lmstudio_url: String, cache: Cache<String, String>) -> Self {
        Self {
            lmstudio_url,
            cache,
        }
    }

    pub async fn resolve_model_name(
        &self,
        ollama_model_name_requested: &str,
        client: &reqwest::Client,
        cancellation_token: CancellationToken,
    ) -> Result<String, ProxyError> {
        let start_time = Instant::now();
        let cleaned_ollama_request = clean_model_name(ollama_model_name_requested).to_string();

        if let Some(cached_lm_studio_id) = self.cache.get(&cleaned_ollama_request).await {
            log::debug!(
                "cache hit: '{}' -> '{}'",
                cleaned_ollama_request,
                cached_lm_studio_id
            );
            return Ok(cached_lm_studio_id);
        }

        log::debug!(
            "cache miss, fetching '{}' from LM Studio",
            cleaned_ollama_request
        );

        match self.get_available_models(client, cancellation_token).await {
            Ok(available_models) => {
                if let Some(matched_model) =
                    Self::resolve_match(&cleaned_ollama_request, &available_models)
                {
                    if !matched_model.is_loaded {
                        log::warn!(
                            "'{}' found but not loaded (state: {})",
                            matched_model.id,
                            matched_model.state
                        );
                    }

                    self.cache
                        .insert(cleaned_ollama_request.clone(), matched_model.id.clone())
                        .await;
                    log_timed(
                        LOG_PREFIX_SUCCESS,
                        &format!(
                            "resolved: '{}' -> '{}' ({})",
                            cleaned_ollama_request, matched_model.id, matched_model.state
                        ),
                        start_time,
                    );
                    Ok(matched_model.id)
                } else {
                    Err(ProxyError::not_found(&format!(
                        "model '{}' not found in LM Studio. Available models can be listed via /api/tags",
                        cleaned_ollama_request
                    )))
                }
            }
            Err(e) => {
                if e.message.contains("404") || e.message.contains("not found") {
                    Err(ProxyError::new(
                        format!(
                            "LM Studio native API not available. Please update to LM Studio 0.3.6+. Original error: {}",
                            e.message
                        ),
                        503,
                    ))
                } else {
                    Err(e)
                }
            }
        }
    }

    async fn get_available_models(
        &self,
        client: &reqwest::Client,
        cancellation_token: CancellationToken,
    ) -> Result<Vec<ModelInfo>, ProxyError> {
        let url = format!("{}{}", self.lmstudio_url, LM_STUDIO_NATIVE_MODELS);

        let request = CancellableRequest::new(client, cancellation_token);

        let response = request
            .make_request(reqwest::Method::GET, &url, None::<Value>)
            .await?;

        if !response.status().is_success() {
            return Err(ProxyError::new(
                format!(
                    "native API error ({}): {}. Ensure LM Studio 0.3.6+ is installed",
                    response.status(),
                    ERROR_LM_STUDIO_UNAVAILABLE
                ),
                response.status().as_u16(),
            ));
        }

        let native_response = response.json::<NativeModelsResponse>().await.map_err(|e| {
            ProxyError::internal_server_error(&format!(
                "invalid JSON from {}: {}. Ensure LM Studio 0.3.6+ is running",
                LM_STUDIO_NATIVE_MODELS, e
            ))
        })?;

        let models = native_response
            .models
            .iter()
            .map(ModelInfo::from_native_data)
            .collect();

        Ok(models)
    }

    fn resolve_match(query: &str, available_models: &[ModelInfo]) -> Option<ModelInfo> {
        let views: Vec<ModelMatchView> = available_models
            .iter()
            .map(|m| ModelMatchView {
                id: m.id.clone(),
                arch: m.arch.clone(),
                model_type: m.model_type.clone(),
                is_loaded: m.is_loaded,
            })
            .collect();
        let matched = find_best_match(query, &views)?;
        available_models
            .iter()
            .find(|m| m.id == matched.id)
            .cloned()
    }

    pub async fn get_all_models(
        &self,
        client: &reqwest::Client,
        cancellation_token: CancellationToken,
    ) -> Result<Vec<ModelInfo>, ProxyError> {
        self.get_available_models(client, cancellation_token).await
    }

    pub async fn get_loaded_models(
        &self,
        client: &reqwest::Client,
        cancellation_token: CancellationToken,
    ) -> Result<Vec<ModelInfo>, ProxyError> {
        let all_models = self.get_all_models(client, cancellation_token).await?;
        Ok(all_models.into_iter().filter(|m| m.is_loaded).collect())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/model_resolver.rs"]
mod tests;
