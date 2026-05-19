use std::time::Duration;

use serde_json::json;
use tokio_util::sync::CancellationToken;

use crate::constants::{LM_STUDIO_NATIVE_MODELS, LM_STUDIO_NATIVE_UNLOAD};
use crate::error::ProxyError;
use crate::model::types::NativeModelsResponse;
use crate::server::ModelResolverType;

pub use super::keep_alive_parse::{
    apply_keep_alive_ttl, keep_alive_requests_unload, parse_keep_alive_seconds,
};

/// Spawns a background task to explicitly unload the model via LMStudio's native
/// unload endpoint when `keep_alive: 0` is requested.
///
/// The `delay_seconds` parameter allows callers to defer the unload (e.g. to let
/// a streaming response finish before tearing down the model instance).
pub fn spawn_model_unload_if_needed(
    client: reqwest::Client,
    base_url: String,
    model_resolver: ModelResolverType,
    ollama_model_name: String,
    keep_alive_seconds: Option<i64>,
    delay_seconds: u64,
) {
    if !matches!(keep_alive_seconds, Some(0)) {
        return;
    }

    tokio::spawn(async move {
        if delay_seconds > 0 {
            tokio::time::sleep(Duration::from_secs(delay_seconds)).await;
        }
        if let Err(e) =
            unload_model_instances(&client, &base_url, &model_resolver, &ollama_model_name).await
        {
            log::warn!(
                "model unload failed for '{}': {}",
                ollama_model_name,
                e.message
            );
        }
    });
}

async fn unload_model_instances(
    client: &reqwest::Client,
    base_url: &str,
    model_resolver: &ModelResolverType,
    ollama_model_name: &str,
) -> Result<(), ProxyError> {
    let lm_studio_id = match model_resolver {
        ModelResolverType::Native(resolver) => {
            resolver
                .resolve_model_name(ollama_model_name, client, CancellationToken::new())
                .await?
        }
    };

    let models_url = format!("{}{}", base_url, LM_STUDIO_NATIVE_MODELS);
    let native: NativeModelsResponse = client
        .get(&models_url)
        .send()
        .await
        .map_err(|e| {
            ProxyError::internal_server_error(&format!("unload: fetch models failed: {e}"))
        })?
        .json()
        .await
        .map_err(|e| {
            ProxyError::internal_server_error(&format!("unload: parse models response failed: {e}"))
        })?;

    let unload_url = format!("{}{}", base_url, LM_STUDIO_NATIVE_UNLOAD);
    for model in native.models.iter().filter(|m| m.key == lm_studio_id) {
        for instance in &model.loaded_instances {
            match client
                .post(&unload_url)
                .json(&json!({ "instance_id": instance.id }))
                .send()
                .await
            {
                Ok(_) => log::debug!("unloaded model instance '{}'", instance.id),
                Err(e) => log::warn!("failed to unload instance '{}': {}", instance.id, e),
            }
        }
    }

    Ok(())
}

#[cfg(test)]
#[path = "../../../tests/unit/handlers_ollama_keep_alive.rs"]
mod tests;
