use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::constants::DEFAULT_KEEP_ALIVE_MINUTES;
use crate::storage::VirtualModelEntry;
use crate::storage::virtual_models::VirtualModelMetadata;

#[derive(Debug, Clone)]
pub struct ModelParameters {
    pub size_string: String,
}

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
    #[serde(default)]
    pub size_bytes: Option<u64>,
    #[serde(default)]
    pub params_string: Option<String>,
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NativeCapabilities {
    #[serde(default)]
    pub vision: Option<bool>,
    #[serde(default)]
    pub trained_for_tool_use: Option<bool>,
    #[serde(default)]
    pub reasoning: Option<NativeReasoningCapability>,
}

/// LM Studio 0.4.0+ surfaces public reasoning config on tool-capable models
/// (e.g. `openai/gpt-oss-20b`). When `allowed_options` contains anything other
/// than `"off"`, the model can think — promote it to the `thinking` capability.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NativeReasoningCapability {
    #[serde(default)]
    pub allowed_options: Vec<String>,
    #[serde(default)]
    pub default: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NativeQuantization {
    pub name: Option<String>,
    #[serde(default)]
    pub bits_per_weight: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NativeLoadedInstanceConfig {
    pub context_length: Option<u64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NativeLoadedInstance {
    pub id: String,
    #[serde(default)]
    pub config: Option<NativeLoadedInstanceConfig>,
}

#[derive(Debug, Deserialize)]
pub struct NativeModelsResponse {
    pub models: Vec<NativeModelData>,
}

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
    pub context_length: u64,
    pub is_loaded: bool,
    pub supports_vision: bool,
    pub supports_tools: bool,
    pub supports_reasoning: bool,
    pub size_bytes: Option<u64>,
    pub params_string: Option<String>,
    pub display_name: Option<String>,
    pub description: Option<String>,
}

impl ModelInfo {
    pub fn merge_with_virtuals<F>(
        base_models: &[ModelInfo],
        virtual_entries: &[VirtualModelEntry],
        transform_fn: F,
    ) -> Vec<Value>
    where
        F: Fn(&ModelInfo) -> Value,
    {
        let mut result: Vec<_> = base_models.iter().map(&transform_fn).collect();

        for entry in virtual_entries {
            if let Some(base_model) = base_models.iter().find(|m| m.id == entry.target_model_id) {
                let aliased = base_model.with_alias_name(&entry.name);
                result.push(transform_fn(&aliased));
            }
        }

        result
    }

    pub fn from_native_data(native_data: &NativeModelData) -> Self {
        let is_loaded = !native_data.loaded_instances.is_empty();
        let state = if is_loaded { "loaded" } else { "not-loaded" };

        let context_length = native_data
            .loaded_instances
            .first()
            .and_then(|inst| inst.config.as_ref())
            .and_then(|cfg| cfg.context_length)
            .unwrap_or(native_data.max_context_length);

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

        let supports_reasoning = native_data
            .capabilities
            .as_ref()
            .and_then(|c| c.reasoning.as_ref())
            .map(|r| {
                r.allowed_options
                    .iter()
                    .any(|opt| !opt.eq_ignore_ascii_case("off"))
            })
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
            context_length,
            is_loaded,
            supports_vision,
            supports_tools,
            supports_reasoning,
            size_bytes: native_data.size_bytes,
            params_string: native_data.params_string.clone(),
            display_name: native_data.display_name.clone(),
            description: native_data.description.clone(),
        }
    }

    pub fn with_alias_name(&self, alias_name: &str) -> Self {
        let mut cloned = self.clone();
        cloned.ollama_name = alias_name.to_string();
        cloned
    }

    fn is_thinking_model(&self) -> bool {
        if self.supports_reasoning {
            return true;
        }
        let lower = self.id.to_lowercase();
        lower.contains("reasoning")
            || lower.contains("thinking")
            || lower.contains("-r1")
            || lower.starts_with("r1-")
            || lower.contains("qwq")
            || lower.contains("qvq")
            || lower.contains("-cot-")
            || lower.contains("deepseek-r")
            || lower.contains("reflect")
    }

    fn determine_capabilities(&self) -> Vec<&'static str> {
        let mut caps = Vec::with_capacity(5);

        match self.model_type.as_str() {
            "llm" => {
                caps.push("completion");
                caps.push("chat");
                if self.is_thinking_model() {
                    caps.push("thinking");
                }
                if self.supports_vision {
                    caps.push("vision");
                }
                if self.supports_tools {
                    caps.push("tools");
                }
            }
            "vlm" => {
                caps.push("completion");
                caps.push("chat");
                caps.push("vision");
                if self.is_thinking_model() {
                    caps.push("thinking");
                }
                if self.supports_tools {
                    caps.push("tools");
                }
            }
            "embeddings" | "embedding" => {
                caps.push("embedding");
            }
            _ => {
                caps.push("completion");
                caps.push("chat");
                if self.is_thinking_model() {
                    caps.push("thinking");
                }
                if self.supports_vision {
                    caps.push("vision");
                }
                if self.supports_tools {
                    caps.push("tools");
                }
            }
        }

        if caps.is_empty() {
            caps.push("completion");
        }

        caps
    }

    pub(crate) fn calculate_estimated_size(&self) -> u64 {
        if let Some(bytes) = self.size_bytes {
            return bytes;
        }
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
            4_000_000_000
        };

        let multiplier = match self.quantization.to_lowercase().as_str() {
            q if q.contains("2bit") || q.contains("q2") => 0.35,
            q if q.contains("3bit") || q.contains("q3") => 0.45,
            q if q.contains("4bit") || q.contains("q4") => 0.55,
            q if q.contains("5bit") || q.contains("q5") => 0.65,
            q if q.contains("6bit") || q.contains("q6") => 0.75,
            q if q.contains("8bit") || q.contains("q8") => 1.0,
            q if q.contains("f16") || q.contains("fp16") => 2.0,
            q if q.contains("f32") || q.contains("fp32") => 4.0,
            _ => 0.55,
        };

        ((base_params as f64) * multiplier) as u64
    }

    pub(crate) fn parse_parameters(&self) -> ModelParameters {
        if let Some(ref s) = self.params_string {
            return ModelParameters {
                size_string: s.clone(),
            };
        }
        let lower_id = self.id.to_lowercase();

        let size_string = if lower_id.contains("0.5b") || lower_id.contains("500m") {
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
        };

        ModelParameters { size_string }
    }

    fn base_ollama_representation(&self) -> Value {
        let estimated_size = self.calculate_estimated_size();
        let params = self.parse_parameters();

        json!({
            "name": self.ollama_name,
            "model": self.ollama_name,
            "size": estimated_size,
            // LM Studio exposes no blob hash; derive a deterministic SHA-256
            // from the model key (stable unique identifier in LM Studio's API).
            "digest": hex::encode(Sha256::digest(self.id.as_bytes())),
            "context_length": self.context_length,
            "max_context_length": self.max_context_length,
            "details": {
                "format": self.compatibility_type,
                "family": self.arch,
                "families": [self.arch],
                "parameter_size": params.size_string,
                "quantization_level": self.quantization,
                "context_length": self.context_length,
                "max_context_length": self.max_context_length
            }
        })
    }

    pub fn to_ollama_tags_model(&self) -> Value {
        // LM Studio's model list exposes no per-model mtime. Real Ollama
        // returns the model file's mtime in `modified_at`; the proxy omits
        // the field rather than fabricate one. Ollama's tags schema allows
        // absence — clients treat it as "unknown, refresh as needed".
        self.base_ollama_representation()
    }

    pub fn to_ollama_ps_model(&self) -> Value {
        let mut base = self.base_ollama_representation();

        if let Some(obj) = base.as_object_mut() {
            obj.insert(
                "expires_at".to_string(),
                (chrono::Utc::now() + chrono::Duration::minutes(DEFAULT_KEEP_ALIVE_MINUTES))
                    .to_rfc3339()
                    .into(),
            );
            // LM Studio exposes KV-cache GPU offload only, not model-weight VRAM usage.
            obj.insert("size_vram".to_string(), json!(0));
        }

        base
    }

    /// Build the `model_info` block for `/api/show`.
    ///
    /// Concise (`verbose: false`) emits the GGUF-style keys real Ollama clients
    /// rely on: `general.*` and the architecture-scoped `<arch>.context_length`.
    /// LM Studio doesn't expose the rest of the GGUF metadata, so we stop there.
    ///
    /// Verbose (`verbose: true`) is the proxy's "tell me everything you know"
    /// mode. Real Ollama emits GGUF tokenizer arrays here; we can't (no GGUF
    /// access), so instead we add every `lmstudio.*` field we can derive from
    /// the native model record.
    fn build_model_info(&self, verbose: bool) -> Value {
        let mut map = serde_json::Map::new();
        map.insert("general.architecture".into(), json!(self.arch));
        map.insert("general.file_type".into(), json!(2));
        // Ollama clients (api-docs/ollama.md line 1485) use general.parameter_count
        // to size memory budgets. Derive from params_string ("7B" → 7e9) or fall
        // back to the heuristic on the model id.
        let params = self.parse_parameters();
        if let Some(count) = crate::model::param_count::parse_parameter_count(&params.size_string) {
            map.insert("general.parameter_count".into(), json!(count));
        }
        map.insert("general.quantization_version".into(), json!(2));
        map.insert(
            format!("{}.context_length", self.arch),
            json!(self.max_context_length),
        );

        if verbose {
            map.insert("lmstudio.publisher".into(), json!(self.publisher));
            map.insert("lmstudio.model_type".into(), json!(self.model_type));
            map.insert("lmstudio.state".into(), json!(self.state));
            map.insert("lmstudio.context_length".into(), json!(self.context_length));
            map.insert(
                "lmstudio.max_context_length".into(),
                json!(self.max_context_length),
            );
            map.insert(
                "lmstudio.compatibility_type".into(),
                json!(self.compatibility_type),
            );
            map.insert("lmstudio.quantization".into(), json!(self.quantization));
            map.insert(
                "lmstudio.supports_vision".into(),
                json!(self.supports_vision),
            );
            map.insert("lmstudio.supports_tools".into(), json!(self.supports_tools));
            map.insert(
                "lmstudio.supports_reasoning".into(),
                json!(self.supports_reasoning),
            );
            map.insert("lmstudio.is_loaded".into(), json!(self.is_loaded));
            if let Some(ref ps) = self.params_string {
                map.insert("lmstudio.params_string".into(), json!(ps));
            }
            if let Some(ref dn) = self.display_name {
                map.insert("lmstudio.display_name".into(), json!(dn));
            }
            if let Some(ref desc) = self.description {
                map.insert("lmstudio.description".into(), json!(desc));
            }
            if let Some(bytes) = self.size_bytes {
                map.insert("lmstudio.size_bytes".into(), json!(bytes));
            }
        }

        Value::Object(map)
    }

    /// Build the `/api/show` response, honouring virtual-alias metadata and
    /// the request's `verbose` flag.
    ///
    /// `parameters` and `template` are only surfaced when the caller passed a
    /// virtual alias that supplies them — LM Studio exposes no Modelfile, so
    /// the proxy refuses to fabricate values that would mislead clients into
    /// thinking these are the model's real generation defaults.
    pub fn to_show_response(
        &self,
        alias_metadata: Option<&VirtualModelMetadata>,
        verbose: bool,
    ) -> Value {
        let capabilities = self.determine_capabilities();
        let mut details = self.base_ollama_representation()["details"].clone();
        if let Some(obj) = details.as_object_mut() {
            obj.insert("parent_model".to_string(), json!(""));
            // /api/show is the stable model description; runtime-loaded context
            // belongs to /api/ps. Always report the model's max here so two
            // calls don't return different numbers depending on load state.
            obj.insert("context_length".to_string(), json!(self.max_context_length));
        }

        // `modified_at` is intentionally absent: LM Studio surfaces no per-model
        // mtime, so the proxy declines to fabricate one. Virtual aliases that
        // have a real `updated_at` are layered in by the /api/show handler.
        let mut response = json!({
            "details": details,
            "capabilities": capabilities,
            "model_info": self.build_model_info(verbose),
        });

        if let Some(obj) = response.as_object_mut() {
            if let Some(name) = &self.display_name {
                obj.insert("display_name".to_string(), json!(name));
            }
            if let Some(desc) = &self.description {
                obj.insert("description".to_string(), json!(desc));
            }

            if let Some(meta) = alias_metadata {
                if let Some(params) = meta.parameters.as_ref() {
                    let value = match params {
                        Value::String(s) => json!(s),
                        other => json!(other),
                    };
                    obj.insert("parameters".to_string(), value);
                }
                if let Some(template) = &meta.template {
                    obj.insert("template".to_string(), json!(template));
                }
            }
        }

        response
    }
}

#[cfg(test)]
#[path = "../../tests/unit/model_types.rs"]
mod tests;
