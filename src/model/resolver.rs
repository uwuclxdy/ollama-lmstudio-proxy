use std::time::Instant;

use moka::future::Cache;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::constants::{ERROR_LM_STUDIO_UNAVAILABLE, LM_STUDIO_NATIVE_MODELS, LOG_PREFIX_SUCCESS};
use crate::error::ProxyError;
use crate::http::CancellableRequest;
use crate::logging::log_timed;
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
                    self.find_best_match(&cleaned_ollama_request, &available_models)
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

    fn find_best_match(
        &self,
        ollama_name_cleaned: &str,
        available_models: &[ModelInfo],
    ) -> Option<ModelInfo> {
        let lower_ollama = ollama_name_cleaned.to_lowercase();

        for model in available_models {
            if model.id.to_lowercase() == lower_ollama {
                return Some(model.clone());
            }
        }

        for model in available_models {
            if model.id.to_lowercase().contains(&lower_ollama)
                && (lower_ollama.len() > model.id.len() / 2 || lower_ollama.len() > 10)
            {
                return Some(model.clone());
            }
        }

        let mut best_match = None;
        let mut best_score = 0;
        for model in available_models {
            let score = self.calculate_match_score(&lower_ollama, model);
            if score > best_score && score >= 3 {
                best_score = score;
                best_match = Some(model.clone());
            }
        }

        best_match
    }

    fn calculate_match_score(&self, ollama_name: &str, model: &ModelInfo) -> usize {
        let model_name_lower = model.id.to_lowercase();
        let ollama_parts: Vec<&str> = ollama_name
            .split(&['-', '_', ':', '.', '/', ' '])
            .filter(|s| !s.is_empty() && s.len() > 1)
            .collect();
        let model_parts: Vec<&str> = model_name_lower
            .split(&['-', '_', ':', '.', '/', ' '])
            .filter(|s| !s.is_empty() && s.len() > 1)
            .collect();

        let mut score = 0;

        for ollama_part in &ollama_parts {
            for model_part in &model_parts {
                if ollama_part == model_part {
                    score += ollama_part.len() * 2;
                } else if model_part.contains(ollama_part) || ollama_part.contains(model_part) {
                    score += ollama_part.len().min(model_part.len());
                }
            }
        }

        if model
            .arch
            .to_lowercase()
            .contains(&ollama_name.to_lowercase())
        {
            score += 5;
        }

        if model.model_type == "llm"
            && (ollama_name.contains("chat") || ollama_name.contains("instruct"))
        {
            score += 3;
        }
        if model.model_type == "vlm"
            && (ollama_name.contains("vision") || ollama_name.contains("llava"))
        {
            score += 3;
        }
        if model.model_type == "embeddings" && ollama_name.contains("embed") {
            score += 3;
        }

        if model.is_loaded {
            score += 2;
        }

        if model_name_lower.starts_with(ollama_name) {
            score += ollama_name.len();
        }

        score
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
