//! Keep-alive translation and explicit model-unload spawning.
//!
//! Ollama keep_alive accepts:
//!   - integer N: seconds (0 = unload now; negative = stay forever)
//!   - string: Go-style duration ("5m", "1h30m", "500ms", optionally negative)
//!
//! LM Studio's `ttl` field is a non-negative seconds count; we normalize
//! negative values to a `-1` sentinel and omit `ttl` from the request,
//! letting LM Studio's default ("loaded indefinitely") apply.

use std::sync::Arc;
use std::time::Duration;

use humantime::parse_duration;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::constants::{LM_STUDIO_NATIVE_MODELS, LM_STUDIO_NATIVE_UNLOAD};
use crate::error::ProxyError;
use crate::model::ModelResolver;
use crate::model::types::NativeModelsResponse;

const FOREVER_SENTINEL: i64 = -1;

pub fn parse_keep_alive_seconds(raw_value: Option<&Value>) -> Result<Option<i64>, ProxyError> {
    let Some(value) = raw_value else {
        return Ok(None);
    };

    match value {
        Value::Null => Ok(None),
        Value::Number(num) => parse_numeric(num),
        Value::String(text) => parse_string(text),
        _ => Err(ProxyError::bad_request(
            "invalid keep_alive value. Use numeric seconds or durations like '5m'",
        )),
    }
}

fn parse_numeric(num: &serde_json::Number) -> Result<Option<i64>, ProxyError> {
    if let Some(signed) = num.as_i64() {
        return Ok(Some(normalize(signed)));
    }
    if let Some(unsigned) = num.as_u64() {
        if unsigned <= i64::MAX as u64 {
            return Ok(Some(unsigned as i64));
        }
        return Err(ProxyError::bad_request(
            "keep_alive value exceeds supported range",
        ));
    }
    Err(ProxyError::bad_request("keep_alive must be integral"))
}

fn parse_string(text: &str) -> Result<Option<i64>, ProxyError> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }

    // Explicit negative durations: "-1", "-1s", "-30m", ...
    if let Some(rest) = trimmed.strip_prefix('-') {
        if rest.parse::<i64>().is_ok() || parse_duration(rest).is_ok() {
            return Ok(Some(FOREVER_SENTINEL));
        }
        return Err(ProxyError::bad_request(
            "invalid keep_alive value. Use numeric seconds or durations like '5m'",
        ));
    }

    if let Ok(duration) = parse_duration(trimmed) {
        let secs = duration.as_secs();
        // Sub-second durations must NOT round down to 0 (would trigger unload).
        let effective = if duration.is_zero() {
            0
        } else if secs == 0 {
            1
        } else if secs > i64::MAX as u64 {
            return Err(ProxyError::bad_request(
                "keep_alive duration exceeds supported range",
            ));
        } else {
            secs as i64
        };
        return Ok(Some(effective));
    }

    trimmed
        .parse::<i64>()
        .map(|n| Some(normalize(n)))
        .map_err(|_| {
            ProxyError::bad_request(
                "invalid keep_alive value. Use numeric seconds or durations like '5m'",
            )
        })
}

fn normalize(secs: i64) -> i64 {
    if secs < 0 { FOREVER_SENTINEL } else { secs }
}

pub fn apply_keep_alive_ttl(target: &mut Value, keep_alive_seconds: Option<i64>) {
    let Some(ttl) = keep_alive_seconds else {
        return;
    };
    if ttl < 0 {
        // "stay loaded forever" → omit ttl (LM Studio default applies)
        return;
    }
    if let Some(obj) = target.as_object_mut() {
        obj.insert("ttl".to_string(), Value::from(ttl));
    }
}

pub fn keep_alive_requests_unload(ttl: Option<i64>) -> bool {
    matches!(ttl, Some(value) if value == 0)
}

/// Spawns a background task to explicitly unload the model via LMStudio's native
/// unload endpoint when `keep_alive: 0` is requested.
///
/// The `delay_seconds` parameter allows callers to defer the unload (e.g. to let
/// a streaming response finish before tearing down the model instance).
pub fn spawn_model_unload_if_needed(
    client: reqwest::Client,
    base_url: String,
    model_resolver: Arc<ModelResolver>,
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
    model_resolver: &Arc<ModelResolver>,
    ollama_model_name: &str,
) -> Result<(), ProxyError> {
    let lm_studio_id = model_resolver
        .resolve_model_name(ollama_model_name, client, CancellationToken::new())
        .await?;

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

/// Core eviction logic operating on an already-fetched models list.
///
/// Unloads every loaded instance of every model whose `key` != `keep_model_key`.
/// Best-effort: per-instance unload failures are logged and skipped.
pub async fn unload_instances_except(
    client: &reqwest::Client,
    models: &NativeModelsResponse,
    keep_model_key: &str,
    unload_url: &str,
) {
    for model in models
        .models
        .iter()
        .filter(|m| m.key != keep_model_key && !m.loaded_instances.is_empty())
    {
        for instance in &model.loaded_instances {
            match client
                .post(unload_url)
                .json(&json!({ "instance_id": instance.id }))
                .send()
                .await
            {
                Ok(_) => log::debug!(
                    "auto-evict: unloaded instance '{}' of '{}'",
                    instance.id,
                    model.key
                ),
                Err(e) => log::warn!(
                    "auto-evict: failed to unload instance '{}' of '{}': {}",
                    instance.id,
                    model.key,
                    e
                ),
            }
        }
    }
}

/// Best-effort: unload every OTHER model's loaded instances (those whose
/// `key` != `keep_model_key`) before a fresh load, mirroring Ollama's
/// single-model default + LM Studio's JIT auto-evict. Never propagates a
/// failure that would abort the caller's load: each per-instance error is
/// logged and skipped, and the function returns `Ok(())` after best-effort
/// attempts.
pub async fn unload_other_models(
    client: &reqwest::Client,
    base_url: &str,
    keep_model_key: &str,
) -> Result<(), ProxyError> {
    let models_url = format!("{}{}", base_url, LM_STUDIO_NATIVE_MODELS);
    let native: NativeModelsResponse = match client.get(&models_url).send().await {
        Ok(response) => match response.json().await {
            Ok(parsed) => parsed,
            Err(e) => {
                log::warn!("auto-evict: parse models response failed, skipping: {}", e);
                return Ok(());
            }
        },
        Err(e) => {
            log::warn!("auto-evict: fetch models failed, skipping: {}", e);
            return Ok(());
        }
    };

    let unload_url = format!("{}{}", base_url, LM_STUDIO_NATIVE_UNLOAD);
    unload_instances_except(client, &native, keep_model_key, &unload_url).await;
    Ok(())
}

/// Proactive eviction: if `keep_model_key` is NOT currently loaded (no
/// `loaded_instances`), unload all other models' instances and return `true`
/// (eviction ran). If the target is already loaded, return `false` (nothing to
/// do). Always best-effort — any fetch/parse failure logs and returns `false`.
///
/// Takes ownership of an already-fetched `NativeModelsResponse` so the caller
/// can share the single GET with its own logic (e.g. checking whether the
/// target needs a context-length reload) without a redundant round trip.
pub async fn proactive_evict_if_unloaded(
    client: &reqwest::Client,
    models: &NativeModelsResponse,
    keep_model_key: &str,
    unload_url: &str,
) -> bool {
    let target_loaded = models
        .models
        .iter()
        .find(|m| m.key == keep_model_key)
        .map(|m| !m.loaded_instances.is_empty())
        .unwrap_or(false);

    if target_loaded {
        return false;
    }

    unload_instances_except(client, models, keep_model_key, unload_url).await;
    true
}

#[cfg(test)]
#[path = "../../tests/unit/lmstudio_keep_alive_spawn.rs"]
mod tests_spawn;

#[cfg(test)]
#[path = "../../tests/unit/lmstudio_keep_alive_parse.rs"]
mod tests_parse;
