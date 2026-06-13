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
