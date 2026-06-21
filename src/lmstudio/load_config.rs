use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use serde_json::{Value, json};
use tokio::sync::Mutex as AsyncMutex;
use tokio_util::sync::CancellationToken;

use crate::api::RequestContext;
use crate::config::RuntimeConfig;
use crate::constants::{LM_STUDIO_MODELS_LOAD, LM_STUDIO_NATIVE_MODELS, LM_STUDIO_NATIVE_UNLOAD};
use crate::http::CancellableRequest;
use crate::model::types::NativeModelsResponse;

/// Build the body for `POST /api/v1/models/load` from the current runtime flags
/// and an optional per-request `context_length` (Ollama `num_ctx`).
///
/// Returns `None` only when nothing would be sent — no `context_length` and none
/// of the three load-tuning flags set — so the caller can skip the load call
/// entirely and preserve byte-for-byte identical behavior with the no-flags
/// default.
pub fn build_load_config_body(
    model: &str,
    rc: &RuntimeConfig,
    context_length: Option<u64>,
) -> Option<Value> {
    if context_length.is_none()
        && !rc.flash_attention
        && !rc.offload_kv_cache
        && rc.eval_batch_size.is_none()
    {
        return None;
    }

    let mut obj = serde_json::Map::new();
    obj.insert("model".to_string(), json!(model));

    if let Some(ctx) = context_length {
        obj.insert("context_length".to_string(), json!(ctx));
    }
    if rc.flash_attention {
        obj.insert("flash_attention".to_string(), json!(true));
    }
    if rc.offload_kv_cache {
        obj.insert("offload_kv_cache_to_gpu".to_string(), json!(true));
    }
    if let Some(batch) = rc.eval_batch_size {
        obj.insert("eval_batch_size".to_string(), json!(batch));
    }

    Some(Value::Object(obj))
}

/// Pull a usable `num_ctx` (positive integer) out of merged Ollama options.
/// Absent, non-integer, or non-positive values yield `None`.
pub fn extract_num_ctx(options: Option<&Value>) -> Option<u64> {
    options
        .and_then(|o| o.get("num_ctx"))
        .and_then(|v| v.as_u64())
        .filter(|n| *n > 0)
}

/// Resolve the context window to enforce: a per-request `num_ctx` wins, else the
/// server-wide `default` (Ollama's `OLLAMA_CONTEXT_LENGTH`), else `None`. Zero or
/// non-positive values from either source are discarded.
fn resolve_requested_ctx(options: Option<&Value>, default: Option<u64>) -> Option<u64> {
    extract_num_ctx(options).or(default).filter(|n| *n > 0)
}

/// Per-model async lock serializing `ensure_context_length` for one model key.
///
/// Without it, two concurrent requests carrying different `num_ctx` for the same
/// model both observe the same instance list, both unload it, and both load —
/// racing LM Studio's "chat routes to the first instance" rule and leaving
/// duplicate instances. The lock makes the read→unload→load sequence atomic per
/// model; different models still proceed concurrently.
fn context_lock_for(model: &str) -> Arc<AsyncMutex<()>> {
    static LOCKS: OnceLock<Mutex<HashMap<String, Arc<AsyncMutex<()>>>>> = OnceLock::new();
    let locks = LOCKS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = locks
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    guard
        .entry(model.to_string())
        .or_insert_with(|| Arc::new(AsyncMutex::new(())))
        .clone()
}

/// Best-effort: make `lm_studio_model_id` serve at the requested `num_ctx` by
/// ensuring exactly one loaded instance at that `context_length`.
///
/// `context_length` is a load-time parameter in LM Studio: the chat body cannot
/// change it. Worse, `POST /api/v1/models/load` spawns a *new* instance instead
/// of reconfiguring an existing one, and chat requests route to the first loaded
/// instance — so honoring Ollama's `num_ctx` (which reloads the model whenever
/// the context changes) means collapsing to a single instance at the requested
/// size. The request is clamped to the model's trained maximum, serialized per
/// model (see `context_lock_for`), and skips the reload if any unload fails (so
/// it never stacks a fresh instance on top of ones still loaded). No-op when
/// `num_ctx` is absent/zero or already satisfied. Every failure is logged and
/// swallowed so this can never fail the user's request.
pub async fn ensure_context_length(
    context: &RequestContext<'_>,
    lm_studio_model_id: &str,
    effective_options: Option<&Value>,
    rc: &RuntimeConfig,
    cancellation: &CancellationToken,
) {
    let Some(mut requested) = resolve_requested_ctx(effective_options, rc.default_context_length)
    else {
        return;
    };
    if cancellation.is_cancelled() {
        return;
    }

    // Serialize read→unload→load for this model so concurrent differing-num_ctx
    // requests can't both reload it and leave duplicate instances.
    let lock = context_lock_for(lm_studio_model_id);
    let _guard = lock.lock().await;

    let models_url = context.endpoint_url(LM_STUDIO_NATIVE_MODELS);
    let native: NativeModelsResponse = match context.client.get(&models_url).send().await {
        Ok(resp) => match resp.json().await {
            Ok(parsed) => parsed,
            Err(e) => {
                log::warn!("num_ctx: parse models response failed: {e}");
                return;
            }
        },
        Err(e) => {
            log::warn!("num_ctx: fetch models failed: {e}");
            return;
        }
    };

    let model = native.models.iter().find(|m| m.key == lm_studio_model_id);
    let instances = model
        .map(|m| m.loaded_instances.iter().collect::<Vec<_>>())
        .unwrap_or_default();

    // Clamp to the model's trained maximum: loading above it fails, and since we
    // unload first that would leave the model with NO instance at all.
    if let Some(max) = model.map(|m| m.max_context_length).filter(|&m| m > 0)
        && requested > max
    {
        log::warn!(
            "num_ctx {requested} exceeds model max {max} for '{lm_studio_model_id}'; clamping to {max}"
        );
        requested = max;
    }

    // All loaded instances already at requested context → nothing to do.
    // (`!is_empty` guard: an empty `.all()` is vacuously true and would wrongly
    // skip the initial load when no instance exists yet.)
    if !instances.is_empty()
        && instances
            .iter()
            .all(|i| i.config.as_ref().and_then(|c| c.context_length) == Some(requested))
    {
        return;
    }

    // Collapse to one instance at the requested context: unload every existing
    // instance, then load once. If an unload errors, skip the load so we never
    // stack a fresh instance on top of ones that are still loaded.
    let unload_url = context.endpoint_url(LM_STUDIO_NATIVE_UNLOAD);
    let mut unloads_ok = true;
    for instance in &instances {
        if let Err(e) = context
            .client
            .post(&unload_url)
            .json(&json!({ "instance_id": instance.id }))
            .send()
            .await
        {
            log::warn!("num_ctx: unload '{}' failed: {e}", instance.id);
            unloads_ok = false;
        }
    }
    if !unloads_ok {
        log::warn!(
            "num_ctx: unload incomplete for '{lm_studio_model_id}', skipping reload to avoid duplicate instances"
        );
        return;
    }

    let Some(load_body) = build_load_config_body(lm_studio_model_id, rc, Some(requested)) else {
        return;
    };
    let load_url = context.endpoint_url(LM_STUDIO_MODELS_LOAD);
    match CancellableRequest::new(context.client, cancellation.clone())
        .make_request(reqwest::Method::POST, &load_url, Some(load_body))
        .await
    {
        Ok(_) => {
            log::debug!("num_ctx: loaded '{lm_studio_model_id}' at context_length={requested}");
            // Record the reload so /api/ps can report a real expires_at. Keyed
            // on the LM Studio model id (= ModelInfo.id). Unknown here: the
            // reload path carries no keep_alive; a subsequent inference request's
            // keep_alive will refresh the deadline when it arrives.
            context.load_tracker.record(
                lm_studio_model_id,
                crate::model::load_tracker::KeepAlive::Unknown,
            );
        }
        Err(e) => log::warn!(
            "num_ctx: load '{lm_studio_model_id}' at {requested} failed: {}",
            e.message
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rc_none() -> RuntimeConfig {
        RuntimeConfig {
            flash_attention: false,
            offload_kv_cache: false,
            eval_batch_size: None,
            ..RuntimeConfig::default()
        }
    }

    #[test]
    fn returns_none_when_no_flags_and_no_context_length() {
        assert!(build_load_config_body("some-model", &rc_none(), None).is_none());
    }

    #[test]
    fn context_length_alone_yields_body() {
        let body = build_load_config_body("m", &rc_none(), Some(4096)).expect("some");
        assert_eq!(body["model"], "m");
        assert_eq!(body["context_length"], 4096);
        assert!(body.get("flash_attention").is_none());
        assert!(body.get("offload_kv_cache_to_gpu").is_none());
        assert!(body.get("eval_batch_size").is_none());
    }

    #[test]
    fn flash_attention_only() {
        let rc = RuntimeConfig {
            flash_attention: true,
            ..rc_none()
        };
        let body = build_load_config_body("m", &rc, None).expect("some");
        assert_eq!(body["model"], "m");
        assert_eq!(body["flash_attention"], true);
        assert!(body.get("context_length").is_none());
        assert!(body.get("offload_kv_cache_to_gpu").is_none());
        assert!(body.get("eval_batch_size").is_none());
    }

    #[test]
    fn offload_kv_cache_only() {
        let rc = RuntimeConfig {
            offload_kv_cache: true,
            ..rc_none()
        };
        let body = build_load_config_body("m", &rc, None).expect("some");
        assert_eq!(body["offload_kv_cache_to_gpu"], true);
        assert!(body.get("flash_attention").is_none());
        assert!(body.get("eval_batch_size").is_none());
    }

    #[test]
    fn eval_batch_size_only() {
        let rc = RuntimeConfig {
            eval_batch_size: Some(512),
            ..rc_none()
        };
        let body = build_load_config_body("m", &rc, None).expect("some");
        assert_eq!(body["eval_batch_size"], 512);
        assert!(body.get("flash_attention").is_none());
        assert!(body.get("offload_kv_cache_to_gpu").is_none());
    }

    #[test]
    fn context_length_alongside_all_flags() {
        let rc = RuntimeConfig {
            flash_attention: true,
            offload_kv_cache: true,
            eval_batch_size: Some(256),
            ..rc_none()
        };
        let body = build_load_config_body("mymodel", &rc, Some(8192)).expect("some");
        assert_eq!(body["model"], "mymodel");
        assert_eq!(body["context_length"], 8192);
        assert_eq!(body["flash_attention"], true);
        assert_eq!(body["offload_kv_cache_to_gpu"], true);
        assert_eq!(body["eval_batch_size"], 256);
    }

    #[test]
    fn model_name_preserved_exactly() {
        let rc = RuntimeConfig {
            flash_attention: true,
            ..rc_none()
        };
        let body =
            build_load_config_body("lmstudio-community/some-model-q4", &rc, None).expect("some");
        assert_eq!(body["model"], "lmstudio-community/some-model-q4");
    }

    #[test]
    fn extract_num_ctx_reads_positive_integer() {
        assert_eq!(
            extract_num_ctx(Some(&json!({ "num_ctx": 4096 }))),
            Some(4096)
        );
    }

    #[test]
    fn extract_num_ctx_rejects_absent_zero_negative_and_non_integer() {
        assert_eq!(extract_num_ctx(None), None);
        assert_eq!(extract_num_ctx(Some(&json!({}))), None);
        assert_eq!(extract_num_ctx(Some(&json!({ "num_ctx": 0 }))), None);
        assert_eq!(extract_num_ctx(Some(&json!({ "num_ctx": -1 }))), None);
        assert_eq!(extract_num_ctx(Some(&json!({ "num_ctx": 1.5 }))), None);
        assert_eq!(extract_num_ctx(Some(&json!({ "num_ctx": "4096" }))), None);
    }

    #[test]
    fn resolve_requested_ctx_request_beats_default() {
        assert_eq!(
            resolve_requested_ctx(Some(&json!({ "num_ctx": 4096 })), Some(8192)),
            Some(4096)
        );
    }

    #[test]
    fn resolve_requested_ctx_falls_back_to_default() {
        assert_eq!(
            resolve_requested_ctx(Some(&json!({})), Some(8192)),
            Some(8192)
        );
    }

    #[test]
    fn resolve_requested_ctx_none_when_neither_set() {
        assert_eq!(resolve_requested_ctx(Some(&json!({})), None), None);
    }

    #[test]
    fn resolve_requested_ctx_rejects_zero_default() {
        assert_eq!(resolve_requested_ctx(Some(&json!({})), Some(0)), None);
    }
}
