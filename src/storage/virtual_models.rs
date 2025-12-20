use std::borrow::Cow;
use std::collections::HashMap;
use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::fs;
use tokio::sync::RwLock;

use crate::error::ProxyError;
use crate::model::clean_model_name;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VirtualModelMetadata {
    pub system_prompt: Option<String>,
    pub template: Option<String>,
    pub parameters: Option<Value>,
    pub license: Option<Value>,
    pub adapters: Option<Value>,
    pub messages: Option<Vec<Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualModelEntry {
    pub name: String,
    pub source_model: String,
    pub target_model_id: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub metadata: VirtualModelMetadata,
}

pub struct VirtualModelStore {
    path: PathBuf,
    entries: RwLock<HashMap<String, VirtualModelEntry>>,
}

impl VirtualModelEntry {
    pub fn to_response(&self) -> Value {
        serde_json::json!({
            "status": "success",
            "model": self.name,
            "virtual": true,
            "source_model": self.source_model,
            "target_model_id": self.target_model_id,
            "created_at": self.created_at.to_rfc3339(),
            "updated_at": self.updated_at.to_rfc3339(),
        })
    }
}

impl VirtualModelStore {
    pub fn build_metadata_from_request(
        body: &Value,
        base: Option<VirtualModelMetadata>,
    ) -> VirtualModelMetadata {
        let mut metadata = base.unwrap_or_default();

        if let Some(system_prompt) = body.get("system").and_then(|v| v.as_str()) {
            metadata.system_prompt = Some(system_prompt.to_string());
        }

        if let Some(template) = body.get("template").and_then(|v| v.as_str()) {
            metadata.template = Some(template.to_string());
        }

        if let Some(parameters) = body.get("parameters") {
            metadata.parameters = Some(parameters.clone());
        }

        if let Some(license) = body.get("license") {
            metadata.license = Some(license.clone());
        }

        if let Some(adapters) = body.get("adapters") {
            metadata.adapters = Some(adapters.clone());
        }

        if let Some(messages) = body.get("messages").and_then(|m| m.as_array()).cloned() {
            metadata.messages = Some(messages);
        }

        metadata
    }

    pub async fn create_from_request(
        &self,
        context: &crate::handlers::RequestContext<'_>,
        model_resolver: &crate::server::ModelResolverType,
        alias_name: &str,
        source_name: &str,
        body: &Value,
        cancellation_token: tokio_util::sync::CancellationToken,
    ) -> Result<VirtualModelEntry, ProxyError> {
        if let Some(files) = body.get("files") {
            let has_content = match files {
                Value::Object(map) => !map.is_empty(),
                Value::Array(arr) => !arr.is_empty(),
                Value::Null => false,
                _ => true,
            };
            if has_content {
                return Err(ProxyError::not_implemented(
                    "creating models from custom blobs is not supported via LM Studio proxy",
                ));
            }
        }

        if body.get("quantize").is_some() {
            return Err(ProxyError::not_implemented(
                "quantization is not available via LM Studio proxy",
            ));
        }

        let (resolved_id, source_virtual_entry) =
            crate::handlers::ollama::utils::resolve_model_target(
                context,
                model_resolver,
                source_name,
                cancellation_token,
            )
            .await?;

        let base_metadata = source_virtual_entry.map(|entry| entry.metadata);
        let metadata = Self::build_metadata_from_request(body, base_metadata);

        self.create_alias(alias_name, source_name.to_string(), resolved_id, metadata)
            .await
    }

    pub fn load<P: Into<PathBuf>>(path: P) -> Result<Self, ProxyError> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                ProxyError::internal_server_error(&format!(
                    "failed to create state directory: {}",
                    e
                ))
            })?;
        }

        let map = if path.exists() {
            match std::fs::read(&path) {
                Ok(bytes) if !bytes.is_empty() => {
                    serde_json::from_slice(&bytes).unwrap_or_default()
                }
                Ok(_) => HashMap::new(),
                Err(e) => {
                    return Err(ProxyError::internal_server_error(&format!(
                        "failed to read {}: {}",
                        path.display(),
                        e
                    )));
                }
            }
        } else {
            HashMap::new()
        };

        Ok(Self {
            path,
            entries: RwLock::new(map),
        })
    }

    fn canonical(model_name: &str) -> Cow<'_, str> {
        let cleaned = clean_model_name(model_name);
        if cleaned.len() == model_name.len() {
            Cow::Borrowed(cleaned)
        } else {
            Cow::Owned(cleaned.to_string())
        }
    }

    pub async fn get(&self, model_name: &str) -> Option<VirtualModelEntry> {
        let key = Self::canonical(model_name);
        let guard = self.entries.read().await;
        guard.get(key.as_ref()).cloned()
    }

    pub async fn create_alias(
        &self,
        alias: &str,
        source_model: String,
        target_model_id: String,
        metadata: VirtualModelMetadata,
    ) -> Result<VirtualModelEntry, ProxyError> {
        let alias_key = Self::canonical(alias).into_owned();
        let mut guard = self.entries.write().await;
        if guard.contains_key(&alias_key) {
            return Err(ProxyError::bad_request(&format!(
                "model '{}' already exists",
                alias
            )));
        }

        let now = Utc::now();
        let entry = VirtualModelEntry {
            name: alias.to_string(),
            source_model,
            target_model_id,
            created_at: now,
            updated_at: now,
            metadata,
        };
        guard.insert(alias_key, entry.clone());
        self.persist_locked(&guard).await?;
        Ok(entry)
    }

    pub async fn delete(&self, alias: &str) -> Result<VirtualModelEntry, ProxyError> {
        let alias_key = Self::canonical(alias).into_owned();
        let mut guard = self.entries.write().await;
        let removed = guard.remove(&alias_key).ok_or_else(|| {
            ProxyError::not_found(&format!("model '{}' not managed by proxy", alias))
        })?;
        self.persist_locked(&guard).await?;
        Ok(removed)
    }

    pub async fn list(&self) -> Vec<VirtualModelEntry> {
        let guard = self.entries.read().await;
        guard.values().cloned().collect()
    }

    async fn persist_locked(
        &self,
        entries: &HashMap<String, VirtualModelEntry>,
    ) -> Result<(), ProxyError> {
        let tmp_path = self.path.with_extension("tmp");
        let data = serde_json::to_vec_pretty(entries).map_err(|e| {
            ProxyError::internal_server_error(&format!("failed to serialize store: {}", e))
        })?;
        fs::write(&tmp_path, data).await.map_err(|e| {
            ProxyError::internal_server_error(&format!(
                "failed to write {}: {}",
                tmp_path.display(),
                e
            ))
        })?;
        fs::rename(&tmp_path, &self.path).await.map_err(|e| {
            ProxyError::internal_server_error(&format!(
                "failed to atomic write {}: {}",
                self.path.display(),
                e
            ))
        })?;
        Ok(())
    }
}
