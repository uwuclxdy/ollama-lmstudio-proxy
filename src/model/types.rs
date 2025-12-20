use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::constants::{
    DEFAULT_KEEP_ALIVE_MINUTES, DEFAULT_REPEAT_PENALTY, DEFAULT_TEMPERATURE, DEFAULT_TOP_K,
    DEFAULT_TOP_P,
};
use crate::storage::VirtualModelEntry;

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
    pub is_loaded: bool,
    pub supports_vision: bool,
    pub supports_tools: bool,
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

    pub fn with_alias_name(&self, alias_name: &str) -> Self {
        let mut cloned = self.clone();
        cloned.ollama_name = alias_name.to_string();
        cloned
    }

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

    fn calculate_estimated_size(&self) -> u64 {
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

    fn parse_parameters(&self) -> ModelParameters {
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
            "digest": format!("{:x}", md5::compute(self.ollama_name.as_bytes())),
            "details": {
                "parent_model": "",
                "format": self.compatibility_type,
                "family": self.arch,
                "families": [self.arch],
                "parameter_size": params.size_string,
                "quantization_level": self.quantization
            }
        })
    }

    pub fn to_ollama_tags_model(&self) -> Value {
        let mut base = self.base_ollama_representation();

        if let Some(obj) = base.as_object_mut() {
            obj.insert(
                "modified_at".to_string(),
                chrono::Utc::now().to_rfc3339().into(),
            );
        }

        base
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
            obj.insert("size_vram".to_string(), obj["size"].clone());
        }

        base
    }

    pub fn to_show_response(&self) -> Value {
        let capabilities = self.determine_capabilities();

        json!({
            "modelfile": format!("# Modelfile for {}\nFROM {} # (Real data from LM Studio)\n\nPARAMETER temperature {}\nPARAMETER top_p {}\nPARAMETER top_k {}\n\nTEMPLATE \"\"\"{{ if .System }}{{ .System }} {{ end }}{{ .Prompt }}\"\"\"",
                self.ollama_name, self.ollama_name, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K
            ),
            "parameters": format!("temperature {}\ntop_p {}\ntop_k {}\nrepeat_penalty {}",
                DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K, DEFAULT_REPEAT_PENALTY),
            "template": "{{ if .System }}{{ .System }}\n{{ end }}{{ .Prompt }}",
            "details": self.base_ollama_representation()["details"].clone(),
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
            "size": self.base_ollama_representation()["size"].as_u64().unwrap_or(0),
            "modified_at": chrono::Utc::now().to_rfc3339()
        })
    }
}
