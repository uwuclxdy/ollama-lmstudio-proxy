use super::*;
use crate::storage::VirtualModelEntry;
use chrono::Utc;
use serde_json::json;

// ─── helpers shared across test sections ────────────────────────────────────

/// Build a minimal NativeModelData with the given key (used by translation_model_types tests).
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
        metadata: Default::default(),
    }
}

fn make_native_with_caps(
    key: &str,
    model_type: &str,
    vision: bool,
    tools: bool,
) -> NativeModelData {
    NativeModelData {
        key: key.to_string(),
        model_type: model_type.to_string(),
        publisher: "test".to_string(),
        architecture: Some("llama".to_string()),
        format: Some("gguf".to_string()),
        quantization: Some(NativeQuantization {
            name: Some("Q4_K_M".to_string()),
            bits_per_weight: Some(4.0),
        }),
        max_context_length: 4096,
        loaded_instances: vec![],
        capabilities: Some(NativeCapabilities {
            vision: Some(vision),
            trained_for_tool_use: Some(tools),
            reasoning: None,
        }),
        size_bytes: None,
        params_string: None,
        display_name: None,
        description: None,
    }
}

fn caps(info: &ModelInfo) -> Vec<&'static str> {
    info.determine_capabilities()
}

#[test]
fn llm_without_instruct_in_name_still_gets_chat() {
    let native = make_native_with_caps("qwen3.5-9b-reasoning-distilled", "llm", false, false);
    let info = ModelInfo::from_native_data(&native);
    assert!(
        caps(&info).contains(&"chat"),
        "expected 'chat' but got {:?}",
        caps(&info)
    );
}

#[test]
fn reasoning_model_gets_thinking_capability() {
    let native = make_native_with_caps("qwen3.5-9b-opus-reasoning-distilled", "llm", false, false);
    let info = ModelInfo::from_native_data(&native);
    assert!(
        caps(&info).contains(&"thinking"),
        "expected 'thinking' but got {:?}",
        caps(&info)
    );
}

#[test]
fn r1_model_gets_thinking_capability() {
    let native = make_native_with_caps("deepseek-r1-7b", "llm", false, false);
    let info = ModelInfo::from_native_data(&native);
    assert!(
        caps(&info).contains(&"thinking"),
        "expected 'thinking' but got {:?}",
        caps(&info)
    );
}

#[test]
fn vision_llm_gets_vision_capability() {
    let native = make_native_with_caps("qwen3.5-9b-instruct-heretic", "llm", true, false);
    let info = ModelInfo::from_native_data(&native);
    assert!(
        caps(&info).contains(&"vision"),
        "expected 'vision' but got {:?}",
        caps(&info)
    );
}

#[test]
fn reasoning_vision_model_gets_both() {
    let native = make_native_with_caps("qvq-72b-preview", "llm", true, false);
    let info = ModelInfo::from_native_data(&native);
    let c = caps(&info);
    assert!(
        c.contains(&"thinking"),
        "expected 'thinking' but got {:?}",
        c
    );
    assert!(c.contains(&"vision"), "expected 'vision' but got {:?}", c);
}

#[test]
fn non_reasoning_llm_does_not_get_thinking() {
    let native = make_native_with_caps("llama-3-8b-instruct", "llm", false, false);
    let info = ModelInfo::from_native_data(&native);
    assert!(
        !caps(&info).contains(&"thinking"),
        "unexpected 'thinking' in {:?}",
        caps(&info)
    );
}

#[test]
fn falls_back_to_heuristic_when_size_bytes_absent() {
    let info = ModelInfo::from_native_data(&native("llama-7b"));
    assert!(info.calculate_estimated_size() > 0);
    assert_ne!(info.calculate_estimated_size(), 4_200_000_000);
}

#[test]
fn falls_back_to_inferred_params_when_absent() {
    let info = ModelInfo::from_native_data(&native("llama-7b-instruct"));
    assert_eq!(info.parse_parameters().size_string, "7B");
}

#[test]
fn reasoning_capability_promotes_to_thinking_even_without_keyword() {
    let mut native = make_native_with_caps("openai/gpt-oss-20b", "llm", false, true);
    native.capabilities.as_mut().unwrap().reasoning = Some(NativeReasoningCapability {
        allowed_options: vec!["off".into(), "low".into(), "medium".into(), "high".into()],
        default: Some("medium".into()),
    });
    let info = ModelInfo::from_native_data(&native);
    assert!(info.supports_reasoning);
    assert!(
        caps(&info).contains(&"thinking"),
        "expected thinking via capabilities, got {:?}",
        caps(&info)
    );
}

#[test]
fn reasoning_only_off_does_not_promote_thinking() {
    let mut native = make_native_with_caps("plain-model", "llm", false, false);
    native.capabilities.as_mut().unwrap().reasoning = Some(NativeReasoningCapability {
        allowed_options: vec!["off".into()],
        default: Some("off".into()),
    });
    let info = ModelInfo::from_native_data(&native);
    assert!(!info.supports_reasoning);
    assert!(!caps(&info).contains(&"thinking"));
}

#[test]
fn show_response_surfaces_display_name_and_description() {
    let mut native = make_native_with_caps("publisher/model", "llm", false, false);
    native.display_name = Some("Pretty Model".into());
    native.description = Some("a description".into());
    let info = ModelInfo::from_native_data(&native);
    let show = info.to_show_response_verbose(false);
    assert_eq!(
        show.get("display_name").and_then(|v| v.as_str()),
        Some("Pretty Model")
    );
    assert_eq!(
        show.get("description").and_then(|v| v.as_str()),
        Some("a description")
    );
}

#[test]
fn show_response_omits_display_fields_when_absent() {
    let native = make_native_with_caps("publisher/model", "llm", false, false);
    let info = ModelInfo::from_native_data(&native);
    let show = info.to_show_response_verbose(false);
    assert!(show.get("display_name").is_none());
    assert!(show.get("description").is_none());
}

// ════════════════════════════════════════════════════════════════════════════
// ModelInfo::from_native_data — state and field mapping
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
    assert_eq!(info.max_context_length, 4096);
}

#[test]
fn from_native_context_length_falls_back_to_max_when_instance_has_no_config() {
    let mut n = native("foo");
    n.max_context_length = 4096;
    n.loaded_instances.push(NativeLoadedInstance {
        id: "x".into(),
        config: None,
    });
    let info = ModelInfo::from_native_data(&n);
    assert_eq!(info.context_length, 4096);
    assert_eq!(info.max_context_length, 4096);
}

// ════════════════════════════════════════════════════════════════════════════
// ModelInfo::to_ollama_tags_model
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn tags_model_has_all_spec_keys() {
    let info = ModelInfo::from_native_data(&native("publisher/model"));
    let v = info.to_ollama_tags_model();
    for key in [
        "name",
        "model",
        "modified_at",
        "size",
        "digest",
        "context_length",
        "max_context_length",
        "details",
    ] {
        assert!(v.get(key).is_some(), "missing key {key} in {v}");
    }
    let details = &v["details"];
    for key in [
        "format",
        "family",
        "families",
        "parameter_size",
        "quantization_level",
        "context_length",
        "max_context_length",
    ] {
        assert!(details.get(key).is_some(), "missing details.{key}");
    }
    assert!(
        details.get("parent_model").is_none(),
        "tags details must not include parent_model"
    );
    assert_eq!(v["context_length"], json!(4096));
    assert_eq!(v["max_context_length"], json!(4096));
    assert_eq!(details["context_length"], json!(4096));
    assert_eq!(details["max_context_length"], json!(4096));
    assert!(
        v["size"].is_u64(),
        "size must serialize as integer, got {}",
        v["size"]
    );
}

#[test]
fn tags_model_digest_is_sha256_shaped() {
    let info = ModelInfo::from_native_data(&native("publisher/model"));
    let v = info.to_ollama_tags_model();
    let digest = v["digest"].as_str().expect("digest must be a string");
    assert_eq!(
        digest.len(),
        64,
        "digest must be 64-char SHA-256 hex, got {digest:?}"
    );
    assert!(
        digest
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()),
        "digest must be lowercase hex, got {digest:?}"
    );
}

#[test]
fn ps_model_digest_is_sha256_shaped() {
    let info = ModelInfo::from_native_data(&native("publisher/model"));
    let v = info.to_ollama_ps_model();
    let digest = v["digest"].as_str().expect("digest must be a string");
    assert_eq!(
        digest.len(),
        64,
        "ps digest must be 64-char SHA-256 hex, got {digest:?}"
    );
    assert!(
        digest
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()),
        "ps digest must be lowercase hex, got {digest:?}"
    );
}

#[test]
fn digest_is_deterministic_for_same_model() {
    let info = ModelInfo::from_native_data(&native("publisher/model"));
    let v1 = info.to_ollama_tags_model();
    let v2 = info.to_ollama_tags_model();
    assert_eq!(
        v1["digest"], v2["digest"],
        "digest must be deterministic across calls"
    );
}

#[test]
fn digest_differs_for_distinct_models() {
    let a = ModelInfo::from_native_data(&native("publisher/model-a"));
    let b = ModelInfo::from_native_data(&native("publisher/model-b"));
    let da = a.to_ollama_tags_model()["digest"].clone();
    let db = b.to_ollama_tags_model()["digest"].clone();
    assert_ne!(da, db, "distinct models must produce distinct digests");
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

#[test]
fn tags_model_modified_at_uses_epoch_fallback() {
    let info = ModelInfo::from_native_data(&native("publisher/model"));
    let v = info.to_ollama_tags_model();
    assert_eq!(v["modified_at"], json!("1970-01-01T00:00:00Z"));
}

// ════════════════════════════════════════════════════════════════════════════
// ModelInfo::to_ollama_ps_model
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
        "context_length",
    ] {
        assert!(v.get(key).is_some(), "missing key {key}");
    }
    assert_eq!(v["size_vram"], json!(0));
    assert_eq!(v["context_length"], json!(info.context_length));
}

#[test]
fn ps_model_expires_at_is_rfc3339_in_future() {
    use chrono::DateTime;
    let info = ModelInfo::from_native_data(&native("publisher/model"));
    let v = info.to_ollama_ps_model();
    let s = v["expires_at"].as_str().expect("expires_at must be string");
    let ts = DateTime::parse_from_rfc3339(s).expect("must parse as RFC3339");
    assert!(ts > Utc::now(), "expires_at must be in the future, got {s}");
}

// ════════════════════════════════════════════════════════════════════════════
// ModelInfo::to_show_response_verbose — additional coverage
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn show_response_has_all_top_level_keys() {
    let info = ModelInfo::from_native_data(&native("publisher/model"));
    let v = info.to_show_response_verbose(false);
    for key in [
        "parameters",
        "template",
        "details",
        "capabilities",
        "modified_at",
    ] {
        assert!(v.get(key).is_some(), "missing key {key} in show response");
    }
    // not in ShowResponse schema — must be absent
    assert!(
        v.get("digest").is_none(),
        "digest must not appear in show response"
    );
    assert!(
        v.get("size").is_none(),
        "size must not appear in show response"
    );
    // modelfile is not in the schema — must be absent
    assert!(
        v.get("modelfile").is_none(),
        "modelfile must not appear in non-verbose show response"
    );
    // model_info is verbose-only
    assert!(
        v.get("model_info").is_none(),
        "model_info must not appear in non-verbose show response"
    );
}

#[test]
fn show_response_parameters_is_multiline_string() {
    let info = ModelInfo::from_native_data(&native("publisher/model"));
    let v = info.to_show_response_verbose(false);
    let params = v["parameters"].as_str().expect("parameters must be string");
    assert!(
        params.contains('\n'),
        "parameters must be multi-line, got {params:?}"
    );
    assert!(params.contains("temperature"));
    assert!(params.contains("top_p"));
    assert!(params.contains("num_ctx 4096"));
}

#[test]
fn show_response_modelfile_absent() {
    // modelfile is not listed in ShowResponse schema — must never appear
    let info = ModelInfo::from_native_data(&native("publisher/model"));
    let v_default = info.to_show_response_verbose(false);
    let v_verbose = info.to_show_response_verbose(true);
    assert!(
        v_default.get("modelfile").is_none(),
        "modelfile must not appear in default show"
    );
    assert!(
        v_verbose.get("modelfile").is_none(),
        "modelfile must not appear in verbose show"
    );
}

#[test]
fn show_response_model_info_has_general_keys_and_arch_context_length() {
    let info = ModelInfo::from_native_data(&native("publisher/model"));
    // model_info is verbose-only
    let v = info.to_show_response_verbose(true);
    let mi = &v["model_info"];
    assert_eq!(mi["general.architecture"], json!("llama"));
    assert_eq!(mi["general.file_type"], json!(2));
    assert_eq!(mi["general.quantization_version"], json!(2));
    assert_eq!(mi["llama.context_length"], json!(4096));
}

#[test]
fn show_response_model_info_has_parameter_count_when_params_known() {
    let mut n = native("foo");
    n.params_string = Some("7B".to_string());
    let info = ModelInfo::from_native_data(&n);
    // model_info is verbose-only
    let v = info.to_show_response_verbose(true);
    let mi = &v["model_info"];
    assert_eq!(mi["general.parameter_count"], json!(7_000_000_000_u64));
}

#[test]
fn show_response_model_info_omits_parameter_count_when_unknown() {
    let info = ModelInfo::from_native_data(&native("publisher/wholly-unknown-shape"));
    // model_info is verbose-only
    let v = info.to_show_response_verbose(true);
    assert!(v["model_info"].get("general.parameter_count").is_none());
}

#[test]
fn show_response_capabilities_reflects_flags() {
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
    let v = info.to_show_response_verbose(false);
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
// verbose vs non-verbose show response
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn show_response_verbose_includes_model_info() {
    let info = ModelInfo::from_native_data(&native("publisher/model"));
    let v = info.to_show_response_verbose(true);
    assert!(
        v.get("model_info").is_some(),
        "verbose must include model_info"
    );
    assert!(v.get("modelfile").is_none(), "modelfile must never appear");
}

#[test]
fn show_response_non_verbose_omits_model_info() {
    let info = ModelInfo::from_native_data(&native("publisher/model"));
    let v = info.to_show_response_verbose(false);
    assert!(
        v.get("model_info").is_none(),
        "non-verbose must omit model_info"
    );
    assert!(v.get("modelfile").is_none(), "modelfile must never appear");
}

#[test]
fn show_response_license_absent_for_base_model() {
    // LM Studio exposes no license data — license must be omitted, not null
    let info = ModelInfo::from_native_data(&native("publisher/model"));
    let v = info.to_show_response_verbose(false);
    assert!(
        v.get("license").is_none(),
        "license must be absent when no license data available"
    );
}

#[test]
fn show_response_license_from_virtual_alias() {
    // virtual alias metadata.license is plumbed through in the handler;
    // confirm VirtualModelMetadata stores the license value correctly.
    use crate::storage::virtual_models::VirtualModelMetadata;
    let meta = VirtualModelMetadata {
        license: Some(json!("Apache 2.0")),
        ..Default::default()
    };
    assert_eq!(meta.license, Some(json!("Apache 2.0")));
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
    let n = native("llama-7b");
    let info = ModelInfo::from_native_data(&n);
    assert_eq!(info.calculate_estimated_size(), 3_850_000_000);
}

#[test]
fn calculate_size_scales_with_quant_for_70b_q8() {
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
    let mut n = native("llama-7b");
    n.quantization = Some(NativeQuantization {
        name: Some("F16".to_string()),
        bits_per_weight: Some(16.0),
    });
    let info = ModelInfo::from_native_data(&n);
    assert_eq!(info.calculate_estimated_size(), 14_000_000_000);
}

// ════════════════════════════════════════════════════════════════════════════
// determine_capabilities — branches beyond the inline tests already present
// ════════════════════════════════════════════════════════════════════════════

fn caps_of(info: &ModelInfo) -> Vec<String> {
    info.to_show_response_verbose(false)["capabilities"]
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
    let base = vec![ModelInfo::from_native_data(&native("a/one"))];
    let virtuals = vec![virt("dangling:latest", "ghost")];
    let out = ModelInfo::merge_with_virtuals(&base, &virtuals, |m| json!({ "n": m.ollama_name }));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0]["n"], "a/one:latest");
}
