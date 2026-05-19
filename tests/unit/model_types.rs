use super::*;

fn make_native(
    key: &str,
    size_bytes: Option<u64>,
    params_string: Option<String>,
) -> NativeModelData {
    NativeModelData {
        key: key.to_string(),
        model_type: "llm".to_string(),
        publisher: "test".to_string(),
        architecture: Some("llama".to_string()),
        format: Some("gguf".to_string()),
        quantization: Some(NativeQuantization {
            name: Some("Q4_K_M".to_string()),
            bits_per_weight: Some(4.0),
        }),
        max_context_length: 4096,
        loaded_instances: vec![],
        capabilities: None,
        size_bytes,
        params_string,
        display_name: None,
        description: None,
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
fn uses_real_size_bytes_when_present() {
    let native = make_native("mymodel", Some(4_200_000_000), None);
    let info = ModelInfo::from_native_data(&native);
    assert_eq!(info.calculate_estimated_size(), 4_200_000_000);
}

#[test]
fn falls_back_to_heuristic_when_size_bytes_absent() {
    let native = make_native("llama-7b", None, None);
    let info = ModelInfo::from_native_data(&native);
    assert!(info.calculate_estimated_size() > 0);
    assert_ne!(info.calculate_estimated_size(), 4_200_000_000);
}

#[test]
fn uses_real_params_string_when_present() {
    let native = make_native("somemodel", None, Some("13B".to_string()));
    let info = ModelInfo::from_native_data(&native);
    assert_eq!(info.parse_parameters().size_string, "13B");
}

#[test]
fn falls_back_to_inferred_params_when_absent() {
    let native = make_native("llama-7b-instruct", None, None);
    let info = ModelInfo::from_native_data(&native);
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
    let show = info.to_show_response();
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
    let show = info.to_show_response();
    assert!(show.get("display_name").is_none());
    assert!(show.get("description").is_none());
}
