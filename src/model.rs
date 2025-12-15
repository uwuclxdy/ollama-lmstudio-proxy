use moka::future::Cache;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::common::CancellableRequest;
use crate::constants::*;
use crate::utils::{ProxyError, log_timed, log_warning};

/// Native LM Studio model data from /api/v1/models
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NativeModelData {
    pub key: String,
    #[serde(rename = "type")]
    pub model_type: String,
    pub publisher: String,
    pub architecture: Option<String>,
    pub format: Option<String>,
    pub quantization: Option<NativeQuantization>,
    pub max_context_length: u64,
    pub loaded_instances: Vec<NativeLoadedInstance>,
    #[serde(default)]
    pub capabilities: Option<NativeCapabilities>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NativeCapabilities {
    #[serde(default)]
    pub vision: Option<bool>,
    #[serde(default)]
    pub trained_for_tool_use: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NativeQuantization {
    pub name: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NativeLoadedInstance {
    pub id: String,
}

/// Native LM Studio models response
#[derive(Debug, Deserialize)]
pub struct NativeModelsResponse {
    pub models: Vec<NativeModelData>,
}

/// Enhanced model information using real LM Studio data
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub ollama_name: String,
    pub model_type: String,
    pub publisher: String,
    pub arch: String,
    pub compatibility_type: String,
    pub quantization: String,
    pub state: String,
    pub max_context_length: u64,
    pub is_loaded: bool,
    pub supports_vision: bool,
    pub supports_tools: bool,
}

impl ModelInfo {
    /// Create model info from native LM Studio data
    pub fn from_native_data(native_data: &NativeModelData) -> Self {
        let is_loaded = !native_data.loaded_instances.is_empty();
        let state = if is_loaded { "loaded" } else { "not-loaded" };

        let ollama_name = if native_data.key.contains(':') {
            native_data.key.clone()
        } else {
            format!("{}:latest", native_data.key)
        };

        let quantization = native_data
            .quantization
            .as_ref()
            .and_then(|q| q.name.clone())
            .unwrap_or_else(|| "unknown".to_string());

        let supports_vision = native_data
            .capabilities
            .as_ref()
            .and_then(|c| c.vision)
            .unwrap_or(false);

        let supports_tools = native_data
            .capabilities
            .as_ref()
            .and_then(|c| c.trained_for_tool_use)
            .unwrap_or(false);

        Self {
            id: native_data.key.clone(),
            ollama_name,
            model_type: native_data.model_type.clone(),
            publisher: native_data.publisher.clone(),
            arch: native_data
                .architecture
                .clone()
                .unwrap_or_else(|| "unknown".to_string()),
            compatibility_type: native_data
                .format
                .clone()
                .unwrap_or_else(|| "unknown".to_string()),
            quantization,
            state: state.to_string(),
            max_context_length: native_data.max_context_length,
            is_loaded,
            supports_vision,
            supports_tools,
        }
    }

    /// Clone model info but override the Ollama-visible name
    pub fn with_alias_name(&self, alias_name: &str) -> Self {
        let mut cloned = self.clone();
        cloned.ollama_name = alias_name.to_string();
        cloned
    }

    /// Determine model capabilities based on LM Studio data and model type
    fn determine_capabilities(&self) -> Vec<String> {
        let mut caps = Vec::new();

        match self.model_type.as_str() {
            "llm" => {
                caps.push("completion".to_string());
                if self.arch.contains("instruct")
                    || self.id.contains("instruct")
                    || self.id.contains("chat")
                {
                    caps.push("chat".to_string());
                }
                if self.supports_vision {
                    caps.push("vision".to_string());
                }
                if self.supports_tools {
                    caps.push("tools".to_string());
                }
            }
            "vlm" => {
                caps.push("completion".to_string());
                caps.push("chat".to_string());
                caps.push("vision".to_string());
                if self.supports_tools {
                    caps.push("tools".to_string());
                }
            }
            "embeddings" | "embedding" => {
                caps.push("embedding".to_string());
            }
            _ => {
                caps.push("completion".to_string());
                if self.supports_vision {
                    caps.push("vision".to_string());
                }
                if self.supports_tools {
                    caps.push("tools".to_string());
                }
            }
        }

        if caps.is_empty() {
            caps.push("completion".to_string());
        }

        caps
    }

    /// Calculate estimated file size based on architecture and quantization
    fn calculate_estimated_size(&self) -> u64 {
        // Extract parameter count from model ID if possible
        let lower_id = self.id.to_lowercase();
        let base_params: u64 = if lower_id.contains("0.5b") || lower_id.contains("500m") {
            500_000_000
        } else if lower_id.contains("1b") && !lower_id.contains("11b") {
            1_000_000_000
        } else if lower_id.contains("2b") && !lower_id.contains("22b") {
            2_000_000_000
        } else if lower_id.contains("3b") && !lower_id.contains("13b") {
            3_000_000_000
        } else if lower_id.contains("7b") {
            7_000_000_000
        } else if lower_id.contains("8b") {
            8_000_000_000
        } else if lower_id.contains("13b") {
            13_000_000_000
        } else if lower_id.contains("70b") {
            70_000_000_000
        } else {
            4_000_000_000 // Default estimate
        };

        // Apply quantization factor
        let multiplier = match self.quantization.to_lowercase().as_str() {
            q if q.contains("2bit") || q.contains("q2") => 0.35,
            q if q.contains("3bit") || q.contains("q3") => 0.45,
            q if q.contains("4bit") || q.contains("q4") => 0.55,
            q if q.contains("5bit") || q.contains("q5") => 0.65,
            q if q.contains("6bit") || q.contains("q6") => 0.75,
            q if q.contains("8bit") || q.contains("q8") => 1.0,
            q if q.contains("f16") || q.contains("fp16") => 2.0,
            q if q.contains("f32") || q.contains("fp32") => 4.0,
            _ => 0.55, // Default to Q4 estimate
        };

        ((base_params as f64) * multiplier) as u64
    }

    /// Generate Ollama-compatible model entry for /api/tags
    pub fn to_ollama_tags_model(&self) -> Value {
        let estimated_size = self.calculate_estimated_size();

        json!({
            "name": self.ollama_name,
            "model": self.ollama_name,
            "modified_at": chrono::Utc::now().to_rfc3339(),
            "size": estimated_size,
            "digest": format!("{:x}", md5::compute(self.ollama_name.as_bytes())),
            "details": {
                "parent_model": "",
                "format": self.compatibility_type,
                "family": self.arch,
                "families": [self.arch],
                "parameter_size": self.extract_parameter_size_string(),
                "quantization_level": self.quantization
            }
        })
    }

    /// Generate Ollama-compatible model entry for /api/ps (running models)
    pub fn to_ollama_ps_model(&self) -> Value {
        let estimated_size = self.calculate_estimated_size();

        json!({
            "name": self.ollama_name,
            "model": self.ollama_name,
            "size": estimated_size,
            "digest": format!("{:x}", md5::compute(self.ollama_name.as_bytes())),
            "details": {
                "parent_model": "",
                "format": self.compatibility_type,
                "family": self.arch,
                "families": [self.arch],
                "parameter_size": self.extract_parameter_size_string(),
                "quantization_level": self.quantization
            },
            "expires_at": (chrono::Utc::now() + chrono::Duration::minutes(DEFAULT_KEEP_ALIVE_MINUTES)).to_rfc3339(),
            "size_vram": estimated_size
        })
    }

    /// Generate model show response for /api/show
    pub fn to_show_response(&self) -> Value {
        let estimated_size = self.calculate_estimated_size();
        let capabilities = self.determine_capabilities();
        let param_size_str = self.extract_parameter_size_string();

        json!({
            "modelfile": format!("# Modelfile for {}\nFROM {} # (Real data from LM Studio)\n\nPARAMETER temperature {}\nPARAMETER top_p {}\nPARAMETER top_k {}\n\nTEMPLATE \"\"\"{{ if .System }}{{ .System }} {{ end }}{{ .Prompt }}\"\"\"",
                self.ollama_name, self.ollama_name, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K
            ),
            "parameters": format!("temperature {}\ntop_p {}\ntop_k {}\nrepeat_penalty {}",
                DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K, DEFAULT_REPEAT_PENALTY),
            "template": "{{ if .System }}{{ .System }}\\n{{ end }}{{ .Prompt }}",
            "details": {
                "parent_model": "",
                "format": self.compatibility_type,
                "family": self.arch,
                "families": [self.arch],
                "parameter_size": param_size_str,
                "quantization_level": self.quantization
            },
            "model_info": {
                "general.architecture": self.arch,
                "general.file_type": 2,
                "general.quantization_version": 2,
                "lmstudio.publisher": self.publisher,
                "lmstudio.model_type": self.model_type,
                "lmstudio.state": self.state,
                "lmstudio.max_context_length": self.max_context_length,
                "lmstudio.compatibility_type": self.compatibility_type,
                "lmstudio.supports_vision": self.supports_vision,
                "lmstudio.supports_tools": self.supports_tools
            },
            "capabilities": capabilities,
            "digest": format!("{:x}", md5::compute(self.ollama_name.as_bytes())),
            "size": estimated_size,
            "modified_at": chrono::Utc::now().to_rfc3339()
        })
    }

    /// Extract parameter size string from model ID
    fn extract_parameter_size_string(&self) -> String {
        let lower_id = self.id.to_lowercase();

        if lower_id.contains("0.5b") || lower_id.contains("500m") {
            "0.5B".to_string()
        } else if lower_id.contains("1b") && !lower_id.contains("11b") {
            "1B".to_string()
        } else if lower_id.contains("2b") && !lower_id.contains("22b") {
            "2B".to_string()
        } else if lower_id.contains("3b") && !lower_id.contains("13b") {
            "3B".to_string()
        } else if lower_id.contains("7b") {
            "7B".to_string()
        } else if lower_id.contains("8b") {
            "8B".to_string()
        } else if lower_id.contains("13b") {
            "13B".to_string()
        } else if lower_id.contains("70b") {
            "70B".to_string()
        } else {
            "unknown".to_string()
        }
    }
}

/// Optimized model name cleaning
pub fn clean_model_name(name: &str) -> &str {
    if name.is_empty() {
        return name;
    }
    let after_latest = if let Some(pos) = name.rfind(":latest") {
        &name[..pos]
    } else {
        name
    };
    if let Some(colon_pos) = after_latest.rfind(':') {
        let suffix = &after_latest[colon_pos + 1..];
        if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) && colon_pos > 0 {
            return &after_latest[..colon_pos];
        }
    }
    after_latest
}

/// ModelResolver for handling model resolution with native LM Studio API
pub struct ModelResolver {
    lmstudio_url: String,
    cache: Cache<String, String>,
}

impl ModelResolver {
    /// Create new model resolver for native API
    pub fn new(lmstudio_url: String, cache: Cache<String, String>) -> Self {
        Self {
            lmstudio_url,
            cache,
        }
    }

    /// Direct model resolution using native API with strict error handling
    pub async fn resolve_model_name(
        &self,
        ollama_model_name_requested: &str,
        client: &reqwest::Client,
        cancellation_token: CancellationToken,
    ) -> Result<String, ProxyError> {
        let start_time = Instant::now();
        let cleaned_ollama_request = clean_model_name(ollama_model_name_requested).to_string();

        // Check cache first
        if let Some(cached_lm_studio_id) = self.cache.get(&cleaned_ollama_request).await {
            log_timed(
                LOG_PREFIX_SUCCESS,
                &format!(
                    "Cache hit: '{}' -> '{}'",
                    cleaned_ollama_request, cached_lm_studio_id
                ),
                start_time,
            );
            return Ok(cached_lm_studio_id);
        }

        log_warning(
            "Cache miss",
            &format!("Fetching '{}' from LM Studio", cleaned_ollama_request),
        );

        match self
            .get_available_lm_studio_models_native(client, cancellation_token)
            .await
        {
            Ok(available_models) => {
                if let Some(matched_model) =
                    self.find_best_match_native(&cleaned_ollama_request, &available_models)
                {
                    // Check if model is loaded for strict error handling
                    if !matched_model.is_loaded {
                        log_warning(
                            "Model state",
                            &format!(
                                "'{}' found but not loaded (state: {})",
                                matched_model.id, matched_model.state
                            ),
                        );
                    }

                    self.cache
                        .insert(cleaned_ollama_request.clone(), matched_model.id.clone())
                        .await;
                    log_timed(
                        LOG_PREFIX_SUCCESS,
                        &format!(
                            "Resolved: '{}' -> '{}' ({})",
                            cleaned_ollama_request, matched_model.id, matched_model.state
                        ),
                        start_time,
                    );
                    Ok(matched_model.id)
                } else {
                    // Strict error handling - don't allow unknown models
                    Err(ProxyError::not_found(&format!(
                        "Model '{}' not found in LM Studio. Available models can be listed via /api/tags",
                        cleaned_ollama_request
                    )))
                }
            }
            Err(e) => {
                // Provide helpful error message for native API issues
                if e.message.contains("404") || e.message.contains("not found") {
                    Err(ProxyError::new(
                        format!(
                            "LM Studio native API not available. Please update to LM Studio 0.3.6+ (legacy mode has been removed). Original error: {}",
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

    /// Get available models from LM Studio native API
    async fn get_available_lm_studio_models_native(
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
                    "Native API error ({}): {}. Ensure LM Studio 0.3.6+ is installed",
                    response.status(),
                    ERROR_LM_STUDIO_UNAVAILABLE
                ),
                response.status().as_u16(),
            ));
        }

        let native_response = response.json::<NativeModelsResponse>().await.map_err(|e| {
            ProxyError::internal_server_error(&format!(
                "Invalid JSON from {}: {}. Ensure LM Studio 0.3.6+ is running",
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

    /// Find best matching model using native model data
    fn find_best_match_native(
        &self,
        ollama_name_cleaned: &str,
        available_models: &[ModelInfo],
    ) -> Option<ModelInfo> {
        let lower_ollama = ollama_name_cleaned.to_lowercase();

        // Exact match first
        for model in available_models {
            if model.id.to_lowercase() == lower_ollama {
                return Some(model.clone());
            }
        }

        // Substring match
        for model in available_models {
            if model.id.to_lowercase().contains(&lower_ollama)
                && (lower_ollama.len() > model.id.len() / 2 || lower_ollama.len() > 10)
            {
                return Some(model.clone());
            }
        }

        // Enhanced scoring match
        let mut best_match = None;
        let mut best_score = 0;
        for model in available_models {
            let score = self.calculate_match_score_native(&lower_ollama, model);
            if score > best_score && score >= 3 {
                best_score = score;
                best_match = Some(model.clone());
            }
        }

        best_match
    }

    /// Calculate match score using native model data
    fn calculate_match_score_native(&self, ollama_name: &str, model: &ModelInfo) -> usize {
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

        // Part matching
        for ollama_part in &ollama_parts {
            for model_part in &model_parts {
                if ollama_part == model_part {
                    score += ollama_part.len() * 2; // Exact part match
                } else if model_part.contains(ollama_part) || ollama_part.contains(model_part) {
                    score += ollama_part.len().min(model_part.len()); // Partial match
                }
            }
        }

        // Architecture matching bonus
        if model
            .arch
            .to_lowercase()
            .contains(&ollama_name.to_lowercase())
        {
            score += 5;
        }

        // Model type matching bonus
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

        // Loaded model bonus (prefer loaded models)
        if model.is_loaded {
            score += 2;
        }

        // Prefix matching bonus
        if model_name_lower.starts_with(ollama_name) {
            score += ollama_name.len();
        }

        score
    }

    /// Get all available models (for /api/tags and /api/ps)
    pub async fn get_all_models(
        &self,
        client: &reqwest::Client,
        cancellation_token: CancellationToken,
    ) -> Result<Vec<ModelInfo>, ProxyError> {
        self.get_available_lm_studio_models_native(client, cancellation_token)
            .await
    }

    /// Get only loaded models (for /api/ps)
    pub async fn get_loaded_models(
        &self,
        client: &reqwest::Client,
        cancellation_token: CancellationToken,
    ) -> Result<Vec<ModelInfo>, ProxyError> {
        let all_models = self.get_all_models(client, cancellation_token).await?;
        Ok(all_models.into_iter().filter(|m| m.is_loaded).collect())
    }
}
