//! Streaming chunk construction and per-choice delta processing for the
//! Ollama-compatible NDJSON stream produced by the proxy.
//!
//! References:
//!   - api_docs/ollama.md /api/chat streaming response shape:
//!       {"model","created_at","message":{"role","content"},"done":false}
//!     final chunk adds done:true, done_reason, and the six timing fields.
//!   - api_docs/ollama.md /api/generate streaming response shape:
//!       {"model","created_at","response","done":false}; final chunk same pattern.
//!   - api_docs/lmstudio/1_developer/3_openai-compat/chat-completions.md
//!     streaming format: SSE `data:` lines with choices[0].delta.{content,
//!     reasoning, tool_calls} and finish_reason.

#[path = "../src/constants.rs"]
#[allow(dead_code)]
mod constants;

#[path = "../src/handlers/transform.rs"]
#[allow(dead_code)]
pub mod transform_inner;

mod handlers {
    pub use super::transform_inner as transform;
}

#[path = "../src/streaming/chunks.rs"]
#[allow(dead_code)]
mod chunks;

use std::time::Duration;

use chunks::{
    ChunkProcessingState, FinalChunkParams, create_cancellation_chunk, create_error_chunk,
    create_final_chunk, create_ollama_streaming_chunk, extract_first_choice, process_choice_delta,
};
use serde_json::{Value, json};

use crate::constants::ERROR_CANCELLED;

// ---------------------------------------------------------------------------
// process_choice_delta / ChunkProcessingState
// ---------------------------------------------------------------------------

/// lmstudio chat-completions.md — `choices[0].delta` may be entirely empty on
/// the role-only opener chunk. Such chunks must be filtered out by returning
/// None so the proxy does not emit an Ollama chunk with an empty message.
#[test]
fn empty_delta_returns_none() {
    let choice = json!({ "delta": {} });
    let mut state = ChunkProcessingState::default();
    assert!(process_choice_delta(&choice, &mut state).is_none());
}

/// lmstudio chat-completions.md — `delta.content` arrives as a String for
/// the common text-only streaming case.
#[test]
fn delta_string_content_captured() {
    let choice = json!({ "delta": { "content": "hello" } });
    let mut state = ChunkProcessingState::default();
    let p = process_choice_delta(&choice, &mut state).unwrap();
    assert_eq!(p.content, "hello");
    assert!(p.thinking.is_empty());
    assert!(p.tool_calls_delta.is_none());
}

/// OpenAI multimodal content parts (lmstudio passes them through): an array
/// of `{type:"text", text:"..."}` fragments must be concatenated in order
/// into the Ollama content buffer.
#[test]
fn delta_array_of_text_parts_is_concatenated() {
    let choice = json!({
        "delta": {
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world"},
            ]
        }
    });
    let mut state = ChunkProcessingState::default();
    let p = process_choice_delta(&choice, &mut state).unwrap();
    assert_eq!(p.content, "Hello world");
}

/// lmstudio chat-completions.md — content arrays may mix `text` and
/// `output_text` part types (Responses-style). Both should be captured.
#[test]
fn delta_array_mixed_text_and_output_text_types() {
    let choice = json!({
        "delta": {
            "content": [
                {"type": "text", "text": "alpha "},
                {"type": "output_text", "text": "beta"},
            ]
        }
    });
    let mut state = ChunkProcessingState::default();
    let p = process_choice_delta(&choice, &mut state).unwrap();
    assert_eq!(p.content, "alpha beta");
}

/// Object content with a nested `content` field — the recursive branch in
/// `append_stream_content` must unwrap one level of nesting.
#[test]
fn delta_object_content_recurses_into_nested_content() {
    let choice = json!({
        "delta": {
            "content": {
                "content": [
                    {"type": "text", "text": "nested"}
                ]
            }
        }
    });
    let mut state = ChunkProcessingState::default();
    let p = process_choice_delta(&choice, &mut state).unwrap();
    assert_eq!(p.content, "nested");
}

/// lmstudio chat-completions.md — `delta.tool_calls` arrives as an OpenAI
/// array; the proxy must convert it to the Ollama tool-call shape (id/type
/// stripped, arguments JSON parsed).
#[test]
fn delta_tool_calls_converted_to_ollama_shape() {
    let choice = json!({
        "delta": {
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_time", "arguments": "{\"tz\":\"UTC\"}"}
            }]
        }
    });
    let mut state = ChunkProcessingState::default();
    let p = process_choice_delta(&choice, &mut state).unwrap();
    let tc = p.tool_calls_delta.expect("tool_calls_delta must be Some");
    let arr = tc.as_array().unwrap();
    assert_eq!(arr.len(), 1);
    let entry = &arr[0];
    assert!(entry.get("id").is_none(), "openai id must be stripped");
    assert!(entry.get("type").is_none(), "openai type must be stripped");
    let args = entry.get("function").unwrap().get("arguments").unwrap();
    assert!(args.is_object(), "arguments must be parsed into object");
    assert_eq!(args.get("tz").and_then(|v| v.as_str()), Some("UTC"));
}

/// lmstudio /v1/completions streaming uses `choices[0].text` instead of
/// `delta.content`. When the delta has no content, the proxy must fall back
/// to the top-level `text` field on the choice.
#[test]
fn completion_choice_text_fallback_used_when_delta_empty() {
    let choice = json!({
        "delta": {},
        "text": "completion piece"
    });
    let mut state = ChunkProcessingState::default();
    let p = process_choice_delta(&choice, &mut state).unwrap();
    assert_eq!(p.content, "completion piece");
}

/// lmstudio chat-completions.md — `finish_reason` may be null until the very
/// last chunk. The proxy must track the latest non-null reason across a
/// stream.
#[test]
fn finish_reason_persists_across_multiple_updates() {
    let mut state = ChunkProcessingState::default();

    let first = json!({ "delta": { "content": "a" }, "finish_reason": null });
    let _ = process_choice_delta(&first, &mut state);
    assert!(state.finish_reason().is_none());

    let second = json!({ "delta": { "content": "b" }, "finish_reason": "stop" });
    let _ = process_choice_delta(&second, &mut state);
    assert_eq!(state.finish_reason(), Some("stop"));
}

/// When several chunks each carry a non-null finish_reason, the LAST one wins.
/// This matches the LM Studio streaming contract: a follow-up tool_calls chunk
/// can supersede an earlier "stop".
#[test]
fn finish_reason_keeps_last_seen_non_null() {
    let mut state = ChunkProcessingState::default();
    let one = json!({ "delta": { "content": "x" }, "finish_reason": "stop" });
    let _ = process_choice_delta(&one, &mut state);
    let two = json!({ "delta": { "content": "y" }, "finish_reason": "tool_calls" });
    let _ = process_choice_delta(&two, &mut state);
    assert_eq!(state.finish_reason(), Some("tool_calls"));
}

/// extract_first_choice returns None when `choices` is missing, empty, or not
/// an array — basic guard.
#[test]
fn extract_first_choice_handles_missing_or_empty() {
    assert!(extract_first_choice(&json!({})).is_none());
    assert!(extract_first_choice(&json!({"choices": []})).is_none());
    let chunk = json!({"choices": [{"delta": {"content": "ok"}}]});
    assert!(extract_first_choice(&chunk).is_some());
}

// ---------------------------------------------------------------------------
// create_ollama_streaming_chunk — chat (is_chat_endpoint = true)
// ---------------------------------------------------------------------------

/// ollama.md §/api/chat streaming response — in-progress chunk shape is
/// {"model","created_at","message":{"role,"content"},"done":false}.
#[test]
fn chat_streaming_chunk_basic_shape() {
    let c = create_ollama_streaming_chunk("llama3", "hi", true, false, None, "");
    assert_eq!(c.get("model").and_then(|v| v.as_str()), Some("llama3"));
    assert_eq!(c.get("done").and_then(|v| v.as_bool()), Some(false));
    let created = c.get("created_at").and_then(|v| v.as_str()).unwrap();
    // RFC3339: chrono::Utc::now().to_rfc3339() — must parse back.
    chrono::DateTime::parse_from_rfc3339(created).expect("created_at must be RFC3339");
    let msg = c.get("message").and_then(|v| v.as_object()).unwrap();
    assert_eq!(msg.get("role").and_then(|v| v.as_str()), Some("assistant"));
    assert_eq!(msg.get("content").and_then(|v| v.as_str()), Some("hi"));
    assert!(c.get("response").is_none(), "chat must not emit response");
}

/// ollama.md §/api/chat — tool_calls live inside `message`, not at the top
/// of the chunk.
#[test]
fn chat_streaming_chunk_tool_calls_in_message() {
    let tc = json!([{
        "function": {"name": "x", "arguments": {"k": "v"}}
    }]);
    let c = create_ollama_streaming_chunk("m", "", true, false, Some(&tc), "");
    let msg = c.get("message").unwrap();
    let calls = msg.get("tool_calls").unwrap().as_array().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].get("function").unwrap().get("name").unwrap(), "x");
}

/// ollama.md §/api/chat — `thinking` is a member of `message` when non-empty;
/// when the proxy has no reasoning to expose, the key must be entirely absent
/// (not an empty string).
#[test]
fn chat_streaming_chunk_thinking_only_when_non_empty() {
    let with = create_ollama_streaming_chunk("m", "hi", true, false, None, "musing");
    assert_eq!(
        with.get("message")
            .unwrap()
            .get("thinking")
            .and_then(|v| v.as_str()),
        Some("musing")
    );
    let without = create_ollama_streaming_chunk("m", "hi", true, false, None, "");
    assert!(
        without.get("message").unwrap().get("thinking").is_none(),
        "empty thinking must not be serialized"
    );
}

// ---------------------------------------------------------------------------
// create_ollama_streaming_chunk — generate (is_chat_endpoint = false)
// ---------------------------------------------------------------------------

/// ollama.md §/api/generate streaming response — in-progress chunk shape is
/// {"model","created_at","response","done":false}.
#[test]
fn generate_streaming_chunk_basic_shape() {
    let c = create_ollama_streaming_chunk("llama3", "tok", false, false, None, "");
    assert_eq!(c.get("model").and_then(|v| v.as_str()), Some("llama3"));
    assert_eq!(c.get("response").and_then(|v| v.as_str()), Some("tok"));
    assert_eq!(c.get("done").and_then(|v| v.as_bool()), Some(false));
    assert!(c.get("message").is_none(), "generate must not emit message");
    let created = c.get("created_at").and_then(|v| v.as_str()).unwrap();
    chrono::DateTime::parse_from_rfc3339(created).expect("created_at must be RFC3339");
}

/// ollama.md §/api/generate — `thinking` lives at the TOP LEVEL of the chunk
/// (no nested message object) and is omitted entirely when empty.
#[test]
fn generate_streaming_chunk_thinking_top_level_only_when_non_empty() {
    let with = create_ollama_streaming_chunk("m", "resp", false, false, None, "musing");
    assert_eq!(
        with.get("thinking").and_then(|v| v.as_str()),
        Some("musing")
    );
    assert!(with.get("message").is_none());

    let without = create_ollama_streaming_chunk("m", "resp", false, false, None, "");
    assert!(without.get("thinking").is_none());
}

// ---------------------------------------------------------------------------
// create_error_chunk
// ---------------------------------------------------------------------------

/// ollama.md §/api/chat — terminal error chunk must set done:true, carry an
/// `error` key, and keep `message.content` as an empty string so clients can
/// still parse `message` without crashing.
#[test]
fn error_chunk_chat_is_terminal_with_error_and_empty_content() {
    let c = create_error_chunk("m", "boom", true);
    assert_eq!(c.get("done").and_then(|v| v.as_bool()), Some(true));
    assert_eq!(c.get("error").and_then(|v| v.as_str()), Some("boom"));
    let msg = c.get("message").unwrap();
    assert_eq!(msg.get("content").and_then(|v| v.as_str()), Some(""));
    assert_eq!(msg.get("role").and_then(|v| v.as_str()), Some("assistant"));
}

/// ollama.md §/api/generate — terminal error chunk must set done:true and
/// carry an `error` key. There is no message object on this endpoint.
#[test]
fn error_chunk_generate_is_terminal_with_error() {
    let c = create_error_chunk("m", "boom", false);
    assert_eq!(c.get("done").and_then(|v| v.as_bool()), Some(true));
    assert_eq!(c.get("error").and_then(|v| v.as_str()), Some("boom"));
    assert!(c.get("message").is_none());
    // response is initialised to empty string by create_ollama_streaming_chunk
    assert_eq!(c.get("response").and_then(|v| v.as_str()), Some(""));
}

// ---------------------------------------------------------------------------
// create_cancellation_chunk
// ---------------------------------------------------------------------------

fn assert_six_timings(chunk: &Value) {
    for key in [
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
    ] {
        assert!(
            chunk.get(key).is_some(),
            "missing timing field `{}` in chunk {}",
            key,
            chunk
        );
    }
}

/// ollama.md §/api/chat — cancelled stream emits a final chunk where
/// `message.content` carries the cancellation notice, done_reason is
/// "cancelled", and all six timing fields are present.
#[test]
fn cancellation_chunk_chat_shape_with_tokens() {
    let c = create_cancellation_chunk("m", Duration::from_millis(50), 7, true);
    assert_eq!(c.get("done").and_then(|v| v.as_bool()), Some(true));
    assert_eq!(
        c.get("done_reason").and_then(|v| v.as_str()),
        Some("cancelled")
    );
    let content = c
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|v| v.as_str())
        .unwrap();
    assert!(
        content.contains("7"),
        "content must include the token estimate; got {:?}",
        content
    );
    assert!(content.to_lowercase().contains("cancel"));
    assert_six_timings(&c);
}

/// ollama.md §/api/generate — same as the chat case but the cancellation
/// notice lives in the top-level `response` field.
#[test]
fn cancellation_chunk_generate_shape_with_tokens() {
    let c = create_cancellation_chunk("m", Duration::from_millis(50), 3, false);
    assert_eq!(c.get("done").and_then(|v| v.as_bool()), Some(true));
    assert_eq!(
        c.get("done_reason").and_then(|v| v.as_str()),
        Some("cancelled")
    );
    let response = c.get("response").and_then(|v| v.as_str()).unwrap();
    assert!(response.contains("3"));
    assert!(response.to_lowercase().contains("cancel"));
    assert_six_timings(&c);
}

/// When no tokens were generated before cancellation, the proxy emits the
/// generic ERROR_CANCELLED string rather than the "after N tokens" template.
#[test]
fn cancellation_chunk_zero_tokens_uses_generic_message() {
    let chat = create_cancellation_chunk("m", Duration::from_millis(10), 0, true);
    let chat_content = chat
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|v| v.as_str())
        .unwrap();
    assert_eq!(chat_content, ERROR_CANCELLED);

    let generate = create_cancellation_chunk("m", Duration::from_millis(10), 0, false);
    let generate_response = generate.get("response").and_then(|v| v.as_str()).unwrap();
    assert_eq!(generate_response, ERROR_CANCELLED);
}

// ---------------------------------------------------------------------------
// create_final_chunk
// ---------------------------------------------------------------------------

/// ollama.md §/api/chat — final streaming chunk: done:true, done_reason
/// defaults to "stop" when the caller passes None, and all six timing fields
/// are present.
#[test]
fn final_chunk_chat_default_done_reason_stop_with_timings() {
    let c = create_final_chunk(FinalChunkParams {
        model_name: "m",
        duration: Duration::from_millis(120),
        chunk_count: 4,
        is_chat: true,
        done_reason: None,
    });
    assert_eq!(c.get("done").and_then(|v| v.as_bool()), Some(true));
    assert_eq!(c.get("done_reason").and_then(|v| v.as_str()), Some("stop"));
    assert!(c.get("message").is_some());
    assert_six_timings(&c);
    // /api/chat never carried `context`; guard against accidental leakage on
    // the streaming final-chunk path too.
    assert!(c.get("context").is_none(), "chat must not emit context");
}

/// Caller-supplied done_reason is propagated verbatim (e.g. "length" when
/// the model hit the max-tokens cap).
#[test]
fn final_chunk_chat_propagates_done_reason_length() {
    let c = create_final_chunk(FinalChunkParams {
        model_name: "m",
        duration: Duration::from_millis(10),
        chunk_count: 1,
        is_chat: true,
        done_reason: Some("length"),
    });
    assert_eq!(
        c.get("done_reason").and_then(|v| v.as_str()),
        Some("length")
    );
}

/// ollama.md §/api/generate — final streaming chunk omits `context` (the
/// deprecated token-ID conversation encoding). Verified for non-streaming in
/// translation_generate_context.rs; re-asserted here for the streaming path.
#[test]
fn final_chunk_generate_omits_context_and_emits_timings() {
    let c = create_final_chunk(FinalChunkParams {
        model_name: "m",
        duration: Duration::from_millis(80),
        chunk_count: 6,
        is_chat: false,
        done_reason: None,
    });
    assert_eq!(c.get("done").and_then(|v| v.as_bool()), Some(true));
    assert_eq!(c.get("done_reason").and_then(|v| v.as_str()), Some("stop"));
    assert!(c.get("message").is_none());
    assert!(
        c.get("context").is_none(),
        "generate streaming final chunk must not emit context"
    );
    assert_six_timings(&c);
}

/// Generate path also honours an explicit done_reason override.
#[test]
fn final_chunk_generate_propagates_done_reason_length() {
    let c = create_final_chunk(FinalChunkParams {
        model_name: "m",
        duration: Duration::from_millis(10),
        chunk_count: 1,
        is_chat: false,
        done_reason: Some("length"),
    });
    assert_eq!(
        c.get("done_reason").and_then(|v| v.as_str()),
        Some("length")
    );
}
