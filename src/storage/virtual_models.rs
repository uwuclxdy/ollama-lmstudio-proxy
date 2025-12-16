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

impl VirtualModelStore {
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
