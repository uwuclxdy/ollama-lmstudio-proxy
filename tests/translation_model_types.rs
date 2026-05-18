//! Tests for translation of LM Studio native model data into the Ollama
//! list/show/ps shapes.
//!
//! Reference docs:
//!   - `api_docs/ollama.md`         — `/api/tags`, `/api/show`, `/api/ps` shape
//!   - `api_docs/lmstudio/1_developer/2_rest/list.md` — native `/api/v1/models`
//!
//! The source modules are mounted via `#[path]` and live alongside small
//! shims for `crate::constants`, `crate::error`, and `crate::storage` so the
//! `use crate::...` lines inside the real modules resolve here too. The shims
//! mirror only the surface area the model layer actually pulls in.

#![allow(dead_code)]

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

// ── crate::constants ────────────────────────────────────────────────────────
// Mirror values from src/constants.rs; only the symbols touched by model/* are
// reproduced. If src/constants.rs diverges, these copies stay stable for the
// test surface — the assertions below do not rely on the literal values, only
// on the structural behavior they drive.
pub mod constants {
    pub const DEFAULT_TEMPERATURE: f64 = 0.7;
    pub const DEFAULT_TOP_P: f64 = 0.9;
    pub const DEFAULT_TOP_K: u32 = 40;
    pub const DEFAULT_REPEAT_PENALTY: f64 = 1.1;
    pub const DEFAULT_KEEP_ALIVE_MINUTES: i64 = 5;
    pub const ERROR_MISSING_MODEL: &str = "Missing 'model' field";
}

// ── crate::error ────────────────────────────────────────────────────────────
// Minimum ProxyError surface used by model::utils::extract_required_model_name.
pub mod error {
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ProxyError {
        pub message: String,
        pub status_code: u16,
    }

    impl ProxyError {
        pub fn bad_request(message: &str) -> Self {
            Self {
                message: message.to_string(),
                status_code: 400,
            }
        }
    }
}

// ── crate::storage ──────────────────────────────────────────────────────────
// Only the data shape touched by ModelInfo::merge_with_virtuals matters:
// `.name` and `.target_model_id`. The real struct in src/storage/virtual_models.rs
// carries additional fields used by other layers, but the model layer reads
// only these two — see src/model/types.rs around the `merge_with_virtuals` loop.
pub mod storage {
    use super::*;

    #[derive(Debug, Clone, Serialize, Deserialize, Default)]
    pub struct VirtualModelMetadata {
        pub system_prompt: Option<String>,
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
}

// ── crate::model ────────────────────────────────────────────────────────────
// Source files are mounted as top-level mods (relative #[path] resolves from
// the directory of the declaring file — keeping them flat avoids needing to
// thread an extra `../`).
#[path = "../src/model/param_count.rs"]
pub mod _param_count;
#[path = "../src/model/timestamps.rs"]
pub mod _timestamps;
#[path = "../src/model/types.rs"]
pub mod _types;
#[path = "../src/model/utils.rs"]
pub mod _utils;

pub mod model {
    pub use super::_param_count as param_count;
    pub use super::_timestamps as timestamps;
    pub use super::_types as types;
    pub use super::_utils as utils;

    pub use super::_types::{
        ModelInfo, NativeCapabilities, NativeLoadedInstance, NativeLoadedInstanceConfig,
        NativeModelData, NativeQuantization, NativeReasoningCapability,
    };
    pub use super::_utils::{clean_model_name, extract_required_model_name};
}

use model::types::{
    ModelInfo, NativeCapabilities, NativeLoadedInstance, NativeLoadedInstanceConfig,
    NativeModelData, NativeQuantization, NativeReasoningCapability,
};
use model::utils::{clean_model_name, extract_required_model_name};
use storage::{VirtualModelEntry, VirtualModelMetadata};

// ── fixtures ────────────────────────────────────────────────────────────────

fn native(key: &str) -> NativeModelData {
    NativeModelData {
        key: key.to_string(),
        model_type: "llm".to_string(),
        publisher: "publisher".to_string(),
        architecture: Some("llama".to_string()),
        format: Some("gguf".to_string()),
        quantization: Some(NativeQuantization {
            name: Some("Q4_K_M".to_string()),
            bits_per_weight: Some(4.0),
        }),
        max_context_length: 4096,
        loaded_instances: vec![],
        capabilities: None,
        size_bytes: None,
        params_string: None,
        display_name: None,
        description: None,
    }
}

fn loaded_instance(ctx: Option<u64>) -> NativeLoadedInstance {
    NativeLoadedInstance {
        id: "inst-1".to_string(),
        config: Some(NativeLoadedInstanceConfig {
            context_length: ctx,
        }),
    }
}

fn virt(name: &str, target_id: &str) -> VirtualModelEntry {
    let now = Utc::now();
    VirtualModelEntry {
        name: name.to_string(),
        source_model: "src".to_string(),
        target_model_id: target_id.to_string(),
        created_at: now,
        updated_at: now,
        metadata: VirtualModelMetadata::default(),
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ModelInfo::from_native_data
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn from_native_unloaded_state() {
    let info = ModelInfo::from_native_data(&native("foo"));
    assert!(!info.is_loaded);
    assert_eq!(info.state, "not-loaded");
}

#[test]
fn from_native_loaded_state() {
    let mut n = native("foo");
    n.loaded_instances.push(loaded_instance(None));
    let info = ModelInfo::from_native_data(&n);
    assert!(info.is_loaded);
    assert_eq!(info.state, "loaded");
}

#[test]
fn from_native_ollama_name_appends_latest_when_no_tag() {
    // LM Studio keys are typically `publisher/model-id` with no colon; Ollama
    // clients expect a `:tag` suffix on every name (api_docs/ollama.md tags).
    let info = ModelInfo::from_native_data(&native("publisher/model"));
    assert_eq!(info.ollama_name, "publisher/model:latest");
}

#[test]
fn from_native_ollama_name_preserves_existing_tag() {
    let info = ModelInfo::from_native_data(&native("publisher/model:custom"));
    assert_eq!(info.ollama_name, "publisher/model:custom");
}

#[test]
fn from_native_quantization_falls_back_to_unknown() {
    let mut n = native("foo");
    n.quantization = None;
    let info = ModelInfo::from_native_data(&n);
    assert_eq!(info.quantization, "unknown");
}

#[test]
fn from_native_quantization_uses_name_when_present() {
    let info = ModelInfo::from_native_data(&native("foo"));
    assert_eq!(info.quantization, "Q4_K_M");
}

#[test]
fn from_native_capabilities_vision_tools_reasoning() {
    let mut n = native("foo");
    n.capabilities = Some(NativeCapabilities {
        vision: Some(true),
        trained_for_tool_use: Some(true),
        reasoning: Some(NativeReasoningCapability {
            allowed_options: vec!["off".into(), "high".into()],
            default: Some("high".into()),
        }),
    });
    let info = ModelInfo::from_native_data(&n);
    assert!(info.supports_vision);
    assert!(info.supports_tools);
    assert!(info.supports_reasoning);
}

#[test]
fn from_native_capabilities_default_false_when_absent() {
    let info = ModelInfo::from_native_data(&native("foo"));
    assert!(!info.supports_vision);
    assert!(!info.supports_tools);
    assert!(!info.supports_reasoning);
}

#[test]
fn from_native_context_length_prefers_loaded_instance_config() {
    // When an instance is loaded with its own context_length, that overrides
    // the model's max_context_length — clients see the live runtime value.
    let mut n = native("foo");
    n.max_context_length = 4096;
    n.loaded_instances.push(loaded_instance(Some(8192)));
    let info = ModelInfo::from_native_data(&n);
    assert_eq!(info.context_length, 8192);
    assert_eq!(info.max_context_length, 4096);
}

#[test]
fn from_native_context_length_falls_back_to_max_when_not_loaded() {
    let mut n = native("foo");
    n.max_context_length = 4096;
    let info = ModelInfo::from_native_data(&n);
    assert_eq!(info.context_length, 4096);
}

#[test]
fn from_native_context_length_falls_back_when_instance_has_no_config() {
    let mut n = native("foo");
    n.max_context_length = 4096;
    n.loaded_instances.push(NativeLoadedInstance {
        id: "x".into(),
        config: None,
    });
    let info = ModelInfo::from_native_data(&n);
    assert_eq!(info.context_length, 4096);
}

// ════════════════════════════════════════════════════════════════════════════
// ModelInfo::to_ollama_tags_model — api_docs/ollama.md /api/tags
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn tags_model_has_all_spec_keys() {
    let info = ModelInfo::from_native_data(&native("publisher/model"));
    let v = info.to_ollama_tags_model();
    for key in ["name", "model", "modified_at", "size", "digest", "details"] {
        assert!(v.get(key).is_some(), "missing key {key} in {v}");
    }
    let details = &v["details"];
    for key in [
        "parent_model",
        "format",
        "family",
        "families",
        "parameter_size",
        "quantization_level",
    ] {
        assert!(details.get(key).is_some(), "missing details.{key}");
    }
    assert!(
        v["size"].is_u64(),
        "size must serialize as integer, got {}",
        v["size"]
    );
}

#[test]
fn tags_model_digest_is_md5_of_ollama_name() {
    let info = ModelInfo::from_native_data(&native("publisher/model"));
    let v = info.to_ollama_tags_model();
    let expected = format!("{:x}", md5::compute(info.ollama_name.as_bytes()));
    assert_eq!(v["digest"].as_str().unwrap(), expected);
}

#[test]
fn tags_model_digest_is_deterministic() {
    let info = ModelInfo::from_native_data(&native("publisher/model"));
    let a = info.to_ollama_tags_model();
    let b = info.to_ollama_tags_model();
    assert_eq!(a["digest"], b["digest"]);
}

#[test]
fn tags_model_details_format_mirrors_compatibility_type() {
    let info = ModelInfo::from_native_data(&native("publisher/model"));
    let v = info.to_ollama_tags_model();
    assert_eq!(v["details"]["format"], json!(info.compatibility_type));
    assert_eq!(v["details"]["format"], json!("gguf"));
}

#[test]
fn tags_model_family_and_families_both_use_arch() {
    let info = ModelInfo::from_native_data(&native("publisher/model"));
    let v = info.to_ollama_tags_model();
    assert_eq!(v["details"]["family"], json!("llama"));
    assert_eq!(v["details"]["families"], json!(["llama"]));
}

// ════════════════════════════════════════════════════════════════════════════
// ModelInfo::to_ollama_ps_model — api_docs/ollama.md /api/ps adds expires_at + size_vram
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn ps_model_includes_tags_fields_plus_expires_and_vram() {
    let info = ModelInfo::from_native_data(&native("publisher/model"));
    let v = info.to_ollama_ps_model();
    for key in [
        "name",
        "model",
        "size",
        "digest",
        "details",
        "expires_at",
        "size_vram",
    ] {
        assert!(v.get(key).is_some(), "missing key {key}");
    }
    assert_eq!(v["size_vram"], v["size"]);
}

#[test]
fn ps_model_expires_at_is_rfc3339_in_future() {
    let info = ModelInfo::from_native_data(&native("publisher/model"));
    let v = info.to_ollama_ps_model();
    let s = v["expires_at"].as_str().expect("expires_at must be string");
    let ts = DateTime::parse_from_rfc3339(s).expect("must parse as RFC3339");
    assert!(ts > Utc::now(), "expires_at must be in the future, got {s}");
}

// ════════════════════════════════════════════════════════════════════════════
// ModelInfo::to_show_response — api_docs/ollama.md /api/show
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn show_response_has_all_top_level_keys() {
    let info = ModelInfo::from_native_data(&native("publisher/model"));
    let v = info.to_show_response();
    for key in [
        "modelfile",
        "parameters",
        "template",
        "details",
        "model_info",
        "capabilities",
        "digest",
        "size",
        "modified_at",
    ] {
        assert!(v.get(key).is_some(), "missing key {key} in show response");
    }
}

#[test]
fn show_response_parameters_is_multiline_string() {
    let info = ModelInfo::from_native_data(&native("publisher/model"));
    let v = info.to_show_response();
    let params = v["parameters"].as_str().expect("parameters must be string");
    assert!(
        params.contains('\n'),
        "parameters must be multi-line, got {params:?}"
    );
    assert!(params.contains("temperature"));
    assert!(params.contains("top_p"));
}

#[test]
fn show_response_modelfile_contains_from_with_ollama_name() {
    let info = ModelInfo::from_native_data(&native("publisher/model"));
    let v = info.to_show_response();
    let mf = v["modelfile"].as_str().unwrap();
    assert!(
        mf.contains(&format!("FROM {}", info.ollama_name)),
        "modelfile must declare FROM <ollama_name>, got: {mf}"
    );
}

#[test]
fn show_response_model_info_has_general_keys_and_arch_context_length() {
    // api_docs/ollama.md /api/show: model_info uses dotted keys like
    // `general.architecture`, `general.parameter_count`, `<arch>.context_length`.
    let info = ModelInfo::from_native_data(&native("publisher/model"));
    let v = info.to_show_response();
    let mi = &v["model_info"];
    assert_eq!(mi["general.architecture"], json!("llama"));
    assert_eq!(mi["general.file_type"], json!(2));
    assert_eq!(mi["general.quantization_version"], json!(2));
    assert!(
        mi.get("llama.context_length").is_some(),
        "missing <arch>.context_length: {mi}"
    );
}

#[test]
fn show_response_model_info_has_parameter_count_when_params_known() {
    // Recent fix: emit general.parameter_count derived from params_string.
    let mut n = native("foo");
    n.params_string = Some("7B".to_string());
    let info = ModelInfo::from_native_data(&n);
    let v = info.to_show_response();
    let mi = &v["model_info"];
    assert_eq!(mi["general.parameter_count"], json!(7_000_000_000_u64));
}

#[test]
fn show_response_model_info_omits_parameter_count_when_unknown() {
    // No params_string and no recognizable substring → parse_parameter_count
    // returns None and the field is left out.
    let info = ModelInfo::from_native_data(&native("publisher/wholly-unknown-shape"));
    let v = info.to_show_response();
    assert!(v["model_info"].get("general.parameter_count").is_none());
}

#[test]
fn show_response_capabilities_reflects_flags() {
    // vlm always picks up vision; tools flag adds tools; reasoning adds thinking.
    let mut n = native("vision-tools");
    n.model_type = "vlm".to_string();
    n.capabilities = Some(NativeCapabilities {
        vision: Some(true),
        trained_for_tool_use: Some(true),
        reasoning: Some(NativeReasoningCapability {
            allowed_options: vec!["high".into()],
            default: Some("high".into()),
        }),
    });
    let info = ModelInfo::from_native_data(&n);
    let v = info.to_show_response();
    let caps: Vec<&str> = v["capabilities"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c.as_str().unwrap())
        .collect();
    assert!(caps.contains(&"vision"), "{caps:?}");
    assert!(caps.contains(&"tools"), "{caps:?}");
    assert!(caps.contains(&"thinking"), "{caps:?}");
    assert!(caps.contains(&"completion"));
    assert!(caps.contains(&"chat"));
}

// ════════════════════════════════════════════════════════════════════════════
// parse_parameters
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn parse_parameters_uses_params_string_when_present() {
    let mut n = native("anything");
    n.params_string = Some("42B".to_string());
    let info = ModelInfo::from_native_data(&n);
    assert_eq!(info.parse_parameters().size_string, "42B");
}

#[test]
fn parse_parameters_id_heuristic_500m_and_half_b() {
    let info = ModelInfo::from_native_data(&native("qwen-0.5b-chat"));
    assert_eq!(info.parse_parameters().size_string, "0.5B");

    let info = ModelInfo::from_native_data(&native("tinyllama-500m"));
    assert_eq!(info.parse_parameters().size_string, "0.5B");
}

#[test]
fn parse_parameters_id_heuristic_1b_excludes_11b() {
    let info = ModelInfo::from_native_data(&native("granite-1b"));
    assert_eq!(info.parse_parameters().size_string, "1B");

    let info = ModelInfo::from_native_data(&native("solar-11b-instruct"));
    assert_ne!(info.parse_parameters().size_string, "1B");
}

#[test]
fn parse_parameters_id_heuristic_common_sizes() {
    assert_eq!(
        ModelInfo::from_native_data(&native("llama-7b"))
            .parse_parameters()
            .size_string,
        "7B"
    );
    assert_eq!(
        ModelInfo::from_native_data(&native("llama-70b"))
            .parse_parameters()
            .size_string,
        "70B"
    );
}

#[test]
fn parse_parameters_unknown_yields_unknown_literal() {
    let info = ModelInfo::from_native_data(&native("mystery/shape"));
    assert_eq!(info.parse_parameters().size_string, "unknown");
}

// ════════════════════════════════════════════════════════════════════════════
// calculate_estimated_size
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn calculate_size_uses_size_bytes_when_present() {
    let mut n = native("anything");
    n.size_bytes = Some(123_456_789);
    let info = ModelInfo::from_native_data(&n);
    assert_eq!(info.calculate_estimated_size(), 123_456_789);
}

#[test]
fn calculate_size_scales_with_quant_for_7b() {
    // 7B * 0.55 (Q4) = 3_850_000_000
    let n = native("llama-7b");
    let info = ModelInfo::from_native_data(&n);
    assert_eq!(info.calculate_estimated_size(), 3_850_000_000);
}

#[test]
fn calculate_size_scales_with_quant_for_70b_q8() {
    // 70B * 1.0 (Q8) = 70_000_000_000
    let mut n = native("llama-70b");
    n.quantization = Some(NativeQuantization {
        name: Some("Q8_0".to_string()),
        bits_per_weight: Some(8.0),
    });
    let info = ModelInfo::from_native_data(&n);
    assert_eq!(info.calculate_estimated_size(), 70_000_000_000);
}

#[test]
fn calculate_size_fp16_doubles_base() {
    // 7B * 2.0 (fp16) = 14_000_000_000
    let mut n = native("llama-7b");
    n.quantization = Some(NativeQuantization {
        name: Some("F16".to_string()),
        bits_per_weight: Some(16.0),
    });
    let info = ModelInfo::from_native_data(&n);
    assert_eq!(info.calculate_estimated_size(), 14_000_000_000);
}

// ════════════════════════════════════════════════════════════════════════════
// determine_capabilities — exercised through to_show_response so we only need
// to verify a few branches not already covered by the inline test module in
// src/model/types.rs.
// ════════════════════════════════════════════════════════════════════════════

fn caps_of(info: &ModelInfo) -> Vec<String> {
    info.to_show_response()["capabilities"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect()
}

#[test]
fn capabilities_embeddings_type_yields_embedding_only() {
    let mut n = native("text-embed-3");
    n.model_type = "embeddings".to_string();
    let info = ModelInfo::from_native_data(&n);
    let c = caps_of(&info);
    assert_eq!(c, vec!["embedding".to_string()]);
}

#[test]
fn capabilities_embedding_singular_also_yields_embedding() {
    let mut n = native("text-embed-3");
    n.model_type = "embedding".to_string();
    let info = ModelInfo::from_native_data(&n);
    assert_eq!(caps_of(&info), vec!["embedding".to_string()]);
}

#[test]
fn capabilities_vlm_includes_vision_always() {
    let mut n = native("llava-7b");
    n.model_type = "vlm".to_string();
    let info = ModelInfo::from_native_data(&n);
    let c = caps_of(&info);
    assert!(c.contains(&"vision".to_string()), "{c:?}");
    assert!(c.contains(&"completion".to_string()));
    assert!(c.contains(&"chat".to_string()));
}

#[test]
fn capabilities_llm_with_tools_includes_tools() {
    let mut n = native("llama-3-8b");
    n.capabilities = Some(NativeCapabilities {
        vision: Some(false),
        trained_for_tool_use: Some(true),
        reasoning: None,
    });
    let info = ModelInfo::from_native_data(&n);
    let c = caps_of(&info);
    assert!(c.contains(&"tools".to_string()), "{c:?}");
}

#[test]
fn capabilities_unknown_model_type_falls_back_to_chat_completion() {
    let mut n = native("foo");
    n.model_type = "what-is-this".to_string();
    let info = ModelInfo::from_native_data(&n);
    let c = caps_of(&info);
    assert!(c.contains(&"completion".to_string()), "{c:?}");
    assert!(c.contains(&"chat".to_string()), "{c:?}");
}

// ════════════════════════════════════════════════════════════════════════════
// ModelInfo::with_alias_name
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn with_alias_name_replaces_ollama_name_only() {
    let info = ModelInfo::from_native_data(&native("publisher/model"));
    let aliased = info.with_alias_name("my-alias:latest");
    assert_eq!(aliased.ollama_name, "my-alias:latest");
    // other fields stay intact
    assert_eq!(aliased.id, info.id);
    assert_eq!(aliased.arch, info.arch);
    assert_eq!(aliased.quantization, info.quantization);
}

// ════════════════════════════════════════════════════════════════════════════
// ModelInfo::merge_with_virtuals
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn merge_with_virtuals_emits_base_models() {
    let base = vec![
        ModelInfo::from_native_data(&native("a/one")),
        ModelInfo::from_native_data(&native("a/two")),
    ];
    let out = ModelInfo::merge_with_virtuals(&base, &[], |m| json!({ "n": m.ollama_name }));
    assert_eq!(out.len(), 2);
    let names: Vec<&str> = out.iter().map(|v| v["n"].as_str().unwrap()).collect();
    assert!(names.contains(&"a/one:latest"));
    assert!(names.contains(&"a/two:latest"));
}

#[test]
fn merge_with_virtuals_appends_known_aliases() {
    let base = vec![
        ModelInfo::from_native_data(&native("a/one")),
        ModelInfo::from_native_data(&native("a/two")),
    ];
    let virtuals = vec![virt("my-alias:latest", "a/one")];
    let out = ModelInfo::merge_with_virtuals(&base, &virtuals, |m| json!({ "n": m.ollama_name }));
    let names: Vec<&str> = out.iter().map(|v| v["n"].as_str().unwrap()).collect();
    assert_eq!(out.len(), 3);
    assert!(names.contains(&"my-alias:latest"), "{names:?}");
    assert!(names.contains(&"a/one:latest"));
    assert!(names.contains(&"a/two:latest"));
}

#[test]
fn merge_with_virtuals_skips_unknown_targets() {
    // target_model_id "ghost" does not match any base ModelInfo.id, so the
    // virtual entry is silently dropped — see src/model/types.rs.
    let base = vec![ModelInfo::from_native_data(&native("a/one"))];
    let virtuals = vec![virt("dangling:latest", "ghost")];
    let out = ModelInfo::merge_with_virtuals(&base, &virtuals, |m| json!({ "n": m.ollama_name }));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0]["n"], "a/one:latest");
}

// ════════════════════════════════════════════════════════════════════════════
// clean_model_name — extra cases beyond what src/model/utils.rs covers inline
// (the inline tests already exercise the alphanumeric-tag and bare-name paths)
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn clean_model_name_strips_latest_suffix() {
    assert_eq!(clean_model_name("model:latest"), "model");
}

#[test]
fn clean_model_name_preserves_alphanumeric_tag() {
    // "7b" is not all-digits → kept.
    assert_eq!(clean_model_name("model:7b"), "model:7b");
}

#[test]
fn clean_model_name_strips_numeric_tag() {
    // Current implementation: suffix that is all ASCII digits gets stripped.
    // This diverges from a naive read of the Ollama spec (`model:tag` is opaque),
    // but matches src/model/utils.rs line 22-26 — documenting the behavior.
    assert_eq!(clean_model_name("model:123"), "model");
}

#[test]
fn clean_model_name_no_colon_unchanged() {
    assert_eq!(clean_model_name("model"), "model");
}

#[test]
fn clean_model_name_empty_unchanged() {
    assert_eq!(clean_model_name(""), "");
}

#[test]
fn clean_model_name_strips_latest_then_evaluates_remaining() {
    // "model:7b:latest" → strip ":latest" → "model:7b" (kept; alphanumeric).
    assert_eq!(clean_model_name("model:7b:latest"), "model:7b");
}

// ════════════════════════════════════════════════════════════════════════════
// extract_required_model_name
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn extract_required_model_name_ok_when_present() {
    let body = json!({ "model": "llama-7b" });
    let got = extract_required_model_name(&body).expect("must accept");
    assert_eq!(got, "llama-7b");
}

#[test]
fn extract_required_model_name_err_when_missing() {
    let body = json!({});
    let err = extract_required_model_name(&body).expect_err("must reject");
    assert_eq!(err.message, constants::ERROR_MISSING_MODEL);
    assert_eq!(err.status_code, 400);
}

#[test]
fn extract_required_model_name_err_when_null() {
    let body = json!({ "model": Value::Null });
    let err = extract_required_model_name(&body).expect_err("must reject");
    assert_eq!(err.message, constants::ERROR_MISSING_MODEL);
}

#[test]
fn extract_required_model_name_err_when_empty_string() {
    // The implementation filters out empty strings — see src/model/utils.rs.
    let body = json!({ "model": "" });
    let err = extract_required_model_name(&body).expect_err("must reject empty string");
    assert_eq!(err.message, constants::ERROR_MISSING_MODEL);
}

#[test]
fn extract_required_model_name_err_when_non_string() {
    let body = json!({ "model": 42 });
    let err = extract_required_model_name(&body).expect_err("must reject non-string");
    assert_eq!(err.message, constants::ERROR_MISSING_MODEL);
}
