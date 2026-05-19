use super::*;
use serde_json::json;

/// /api/generate: a top-level `system` string is the canonical Ollama field.
/// Reference: api_docs/ollama.md §"Generate a completion" — `system` overrides
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
/// override). Reference: api_docs/ollama.md §"Generate a completion".
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
