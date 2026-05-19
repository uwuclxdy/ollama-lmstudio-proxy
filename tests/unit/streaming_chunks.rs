use super::*;
use serde_json::json;

fn choice_with_delta(content: Option<&str>, reasoning: Option<&str>) -> serde_json::Value {
    let mut delta = json!({});
    if let Some(c) = content {
        delta
            .as_object_mut()
            .unwrap()
            .insert("content".to_string(), json!(c));
    }
    if let Some(r) = reasoning {
        delta
            .as_object_mut()
            .unwrap()
            .insert("reasoning".to_string(), json!(r));
    }
    json!({ "delta": delta })
}

#[test]
fn reasoning_goes_to_thinking_not_content() {
    let choice = choice_with_delta(Some("answer"), Some("my thinking"));
    let mut state = ChunkProcessingState::default();
    let payload = process_choice_delta(&choice, &mut state).unwrap();
    assert_eq!(payload.content, "answer");
    assert_eq!(payload.thinking, "my thinking");
}

#[test]
fn reasoning_only_chunk_is_not_dropped() {
    let choice = choice_with_delta(None, Some("reasoning only"));
    let mut state = ChunkProcessingState::default();
    let payload = process_choice_delta(&choice, &mut state);
    assert!(payload.is_some());
    let p = payload.unwrap();
    assert_eq!(p.content, "");
    assert_eq!(p.thinking, "reasoning only");
}

#[test]
fn chat_chunk_thinking_in_message() {
    let chunk = create_ollama_streaming_chunk("m", "hi", true, false, None, "my thought");
    let msg = chunk.get("message").unwrap();
    assert_eq!(
        msg.get("thinking").and_then(|v| v.as_str()),
        Some("my thought")
    );
    assert_eq!(msg.get("content").and_then(|v| v.as_str()), Some("hi"));
}

#[test]
fn chat_chunk_no_thinking_field_when_empty() {
    let chunk = create_ollama_streaming_chunk("m", "hi", true, false, None, "");
    let msg = chunk.get("message").unwrap();
    assert!(msg.get("thinking").is_none());
}

#[test]
fn generate_chunk_thinking_top_level() {
    let chunk = create_ollama_streaming_chunk("m", "response", false, false, None, "thought");
    assert_eq!(
        chunk.get("thinking").and_then(|v| v.as_str()),
        Some("thought")
    );
    assert!(chunk.get("message").is_none());
}
