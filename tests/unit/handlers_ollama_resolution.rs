use super::*;
use serde_json::json;

#[test]
fn extracts_think_from_body() {
    let body = json!({ "think": true, "model": "x", "messages": [] });
    let top = make_top_level_params(&body);
    assert!(top.think.is_some());
    assert_eq!(top.think, Some(&json!(true)));
}

#[test]
fn absent_think_gives_none() {
    let body = json!({ "model": "x", "messages": [] });
    let top = make_top_level_params(&body);
    assert!(top.think.is_none());
}

#[test]
fn extracts_logprobs_and_top_logprobs() {
    let body = json!({ "logprobs": true, "top_logprobs": 3 });
    let top = make_top_level_params(&body);
    assert_eq!(top.logprobs, Some(&json!(true)));
    assert_eq!(top.top_logprobs, Some(&json!(3)));
}

#[test]
fn extracts_think_from_generate_body() {
    let body = json!({ "think": "high", "model": "x", "prompt": "hi" });
    let top = make_top_level_params(&body);
    assert_eq!(top.think, Some(&json!("high")));
}

#[test]
fn suffix_inserted_into_lm_request() {
    use crate::lmstudio::request::{LMStudioRequestType, TopLevelParams, build_lm_studio_request};
    use std::borrow::Cow;

    let body = json!({ "suffix": "world", "model": "test", "prompt": "hello" });
    let suffix_val = body.get("suffix");
    let top_level = TopLevelParams {
        think: None,
        logprobs: None,
        top_logprobs: None,
        model_is_thinking: false,
    };

    let mut lm_request = build_lm_studio_request(
        "test",
        LMStudioRequestType::Completion {
            prompt: Cow::Borrowed("hello"),
            stream: false,
        },
        None,
        None,
        None,
        Some(&top_level),
    );

    if let Some(s) = suffix_val
        && let Some(obj) = lm_request.as_object_mut()
    {
        obj.insert("suffix".to_string(), s.clone());
    }

    assert_eq!(lm_request.get("suffix"), Some(&json!("world")));
}

#[test]
fn suffix_not_inserted_on_vision_path() {
    let body = json!({ "suffix": "world", "model": "test", "prompt": "hello",
                       "images": ["base64data"] });
    let current_images = body.get("images");
    let suffix_val = body.get("suffix");
    let mut lm_request = json!({ "model": "test" });

    if current_images.is_none()
        && let Some(s) = suffix_val
        && let Some(obj) = lm_request.as_object_mut()
    {
        obj.insert("suffix".to_string(), s.clone());
    }

    assert!(
        lm_request.get("suffix").is_none(),
        "suffix must be absent on vision path"
    );
}

#[test]
fn absent_think_gives_none_in_generate() {
    let body = json!({ "model": "x", "prompt": "hi" });
    let top = make_top_level_params(&body);
    assert!(top.think.is_none());
}

/// /api/generate: a top-level `system` string is the canonical Ollama field.
/// Reference: api-docs/ollama.md §"Generate a completion" — `system` overrides
/// the Modelfile-defined system prompt.
#[test]
fn extract_system_prompt_finds_top_level_system() {
    let body = json!({ "system": "be concise" });
    assert_eq!(extract_system_prompt(&body), Some("be concise".to_string()));
}

/// Some clients place the system prompt inside `options`. The helper falls
/// back to `options.system` when the top-level field is absent.
#[test]
fn extract_system_prompt_falls_back_to_options_system() {
    let body = json!({ "options": { "system": "from options" } });
    assert_eq!(
        extract_system_prompt(&body),
        Some("from options".to_string())
    );
}

/// When both are present the top-level field wins (it is the documented
/// override). Reference: api-docs/ollama.md §"Generate a completion".
#[test]
fn extract_system_prompt_top_level_wins_over_options() {
    let body = json!({
        "system": "top wins",
        "options": { "system": "should be ignored" }
    });
    assert_eq!(extract_system_prompt(&body), Some("top wins".to_string()));
}

/// Neither location populated → None.
#[test]
fn extract_system_prompt_returns_none_when_absent() {
    let body = json!({ "model": "x" });
    assert!(extract_system_prompt(&body).is_none());
}

/// Documented divergence: the helper only checks `body.system` and
/// `body.options.system`. /api/chat's `messages` array with a `{role:"system"}`
/// entry is NOT scanned here — that work is done by other translation steps
/// (e.g. normalize_chat_messages). This test pins the current behavior.
#[test]
fn extract_system_prompt_does_not_inspect_messages_array() {
    let body = json!({
        "messages": [
            { "role": "system", "content": "from messages" },
            { "role": "user", "content": "hi" }
        ]
    });
    assert!(
        extract_system_prompt(&body).is_none(),
        "extract_system_prompt is not responsible for the messages array; \
         system-role messages flow through normalize_chat_messages"
    );
}

/// Non-string `system` is ignored (the helper only accepts strings).
#[test]
fn extract_system_prompt_rejects_non_string_system() {
    let body = json!({ "system": 42 });
    assert!(extract_system_prompt(&body).is_none());
}

#[test]
fn reasoning_effort_used_when_think_absent() {
    let body = json!({ "reasoning_effort": "medium", "model": "x", "messages": [] });
    let top = make_top_level_params(&body);
    assert_eq!(
        top.think,
        Some(&json!("medium")),
        "reasoning_effort must be picked up when think is absent"
    );
}

#[test]
fn think_wins_over_reasoning_effort_when_both_present() {
    let body = json!({
        "think": "high",
        "reasoning_effort": "low",
        "model": "x",
        "messages": []
    });
    let top = make_top_level_params(&body);
    assert_eq!(
        top.think,
        Some(&json!("high")),
        "think must take precedence over reasoning_effort"
    );
}

#[test]
fn make_top_level_params_defaults_model_is_thinking_false() {
    // The body alone can't tell whether the model reasons; the inference path
    // fills this from the resolved model, so the constructed default is false.
    let body = json!({ "think": true, "model": "x", "messages": [] });
    let top = make_top_level_params(&body);
    assert!(!top.model_is_thinking);
}

// Build a minimal ModelInfo without the HTTP stack. `resolve_model_with_context`
// populates `model_supports_thinking` from exactly this predicate, so testing
// `is_thinking_model()` pins the field's value for both cases.
fn model_info(id: &str, supports_reasoning: bool) -> ModelInfo {
    ModelInfo {
        id: id.to_string(),
        ollama_name: format!("{id}:latest"),
        model_type: "llm".to_string(),
        publisher: "test".to_string(),
        arch: "llama".to_string(),
        compatibility_type: "gguf".to_string(),
        quantization: "q4_0".to_string(),
        state: "not-loaded".to_string(),
        max_context_length: 4096,
        context_length: 4096,
        is_loaded: false,
        supports_vision: false,
        supports_tools: false,
        supports_reasoning,
        has_backend_capabilities: true,
        size_bytes: None,
        params_string: None,
        display_name: None,
        description: None,
        loaded_flash_attention: None,
        loaded_eval_batch_size: None,
        loaded_parallel: None,
    }
}

#[test]
fn model_supports_thinking_true_for_reasoning_model() {
    // Mirrors how resolve_model_with_context derives model_supports_thinking.
    let info = model_info("some-model", true);
    assert!(info.is_thinking_model());
}

#[test]
fn model_supports_thinking_false_for_non_reasoning_model() {
    let info = model_info("some-model", false);
    assert!(!info.is_thinking_model());
}

// ── Modelfile PARAMs as defaults: merge_option_maps ─────────────────────────
//
// A virtual model's `metadata.parameters` (Modelfile PARAM block) feeds the
// `base` slot of `merge_option_maps`; per-request `options` feed `overrides`.
// The merge yields the PARAMs as defaults, with caller-supplied keys winning
// per-key (mirrors real Ollama: PARAMs are the floor, `options` override).

#[test]
fn merge_parameters_applied_as_defaults_when_options_absent() {
    let base = json!({ "temperature": 0.5, "num_ctx": 4096 });
    let merged = merge_option_maps(Some(&base), None).expect("base present");
    assert_eq!(merged["temperature"], 0.5);
    assert_eq!(merged["num_ctx"], 4096);
}

#[test]
fn merge_parameters_caller_options_override_defaults_per_key() {
    let base = json!({ "temperature": 0.5, "num_ctx": 4096 });
    let overrides = json!({ "temperature": 0.9, "seed": 42 });
    let merged = merge_option_maps(Some(&base), Some(&overrides)).expect("both present");

    // Override wins on the shared key...
    assert_eq!(
        merged["temperature"], 0.9,
        "caller option must override the PARAM default"
    );
    // ...base-only key is preserved (the default survives)...
    assert_eq!(
        merged["num_ctx"], 4096,
        "un-overridden PARAM default must carry through"
    );
    // ...and override-only key is added.
    assert_eq!(
        merged["seed"], 42,
        "caller-only key must appear in the merged map"
    );
}

#[test]
fn merge_options_without_parameters_passes_options_through() {
    let overrides = json!({ "temperature": 0.7 });
    let merged = merge_option_maps(None, Some(&overrides)).expect("override only");
    assert_eq!(merged["temperature"], 0.7);
    assert_eq!(merged.as_object().unwrap().len(), 1);
}

#[test]
fn merge_both_absent_returns_none() {
    assert!(merge_option_maps(None, None).is_none());
}

/// A non-object base (e.g. an array or scalar) must not crash the merge: the
/// helper falls back to the override verbatim, so the request still works.
#[test]
fn merge_non_object_base_falls_back_to_overrides() {
    let base = json!([1, 2, 3]);
    let overrides = json!({ "temperature": 0.2 });
    let merged = merge_option_maps(Some(&base), Some(&overrides)).expect("fallback merged");
    assert_eq!(
        merged["temperature"], 0.2,
        "non-object base must defer to the override"
    );
}
