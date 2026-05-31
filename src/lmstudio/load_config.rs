use serde_json::{Value, json};

use crate::config::RuntimeConfig;

/// Build the body for `POST /api/v1/models/load` from the current runtime flags.
///
/// Returns `None` when none of the three load-tuning flags are set, so the
/// caller can skip the load call entirely and preserve byte-for-byte identical
/// behavior with the no-flags default.
pub fn build_load_config_body(model: &str, rc: &RuntimeConfig) -> Option<Value> {
    if !rc.flash_attention && !rc.offload_kv_cache && rc.eval_batch_size.is_none() {
        return None;
    }

    let mut body = json!({ "model": model });
    let obj = body.as_object_mut().expect("json object");

    if rc.flash_attention {
        obj.insert("flash_attention".to_string(), json!(true));
    }
    if rc.offload_kv_cache {
        obj.insert("offload_kv_cache_to_gpu".to_string(), json!(true));
    }
    if let Some(batch) = rc.eval_batch_size {
        obj.insert("eval_batch_size".to_string(), json!(batch));
    }

    Some(body)
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
    fn returns_none_when_no_flags_set() {
        assert!(build_load_config_body("some-model", &rc_none()).is_none());
    }

    #[test]
    fn flash_attention_only() {
        let rc = RuntimeConfig {
            flash_attention: true,
            ..rc_none()
        };
        let body = build_load_config_body("m", &rc).expect("some");
        assert_eq!(body["model"], "m");
        assert_eq!(body["flash_attention"], true);
        assert!(body.get("offload_kv_cache_to_gpu").is_none());
        assert!(body.get("eval_batch_size").is_none());
    }

    #[test]
    fn offload_kv_cache_only() {
        let rc = RuntimeConfig {
            offload_kv_cache: true,
            ..rc_none()
        };
        let body = build_load_config_body("m", &rc).expect("some");
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
        let body = build_load_config_body("m", &rc).expect("some");
        assert_eq!(body["eval_batch_size"], 512);
        assert!(body.get("flash_attention").is_none());
        assert!(body.get("offload_kv_cache_to_gpu").is_none());
    }

    #[test]
    fn all_flags_set() {
        let rc = RuntimeConfig {
            flash_attention: true,
            offload_kv_cache: true,
            eval_batch_size: Some(256),
            ..rc_none()
        };
        let body = build_load_config_body("mymodel", &rc).expect("some");
        assert_eq!(body["model"], "mymodel");
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
        let body = build_load_config_body("lmstudio-community/some-model-q4", &rc).expect("some");
        assert_eq!(body["model"], "lmstudio-community/some-model-q4");
    }
}
