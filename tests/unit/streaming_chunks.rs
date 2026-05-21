use std::time::Duration;

use serde_json::{Value, json};

use super::*;

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

// ════════════════════════════════════════════════════════════════════════════
// process_choice_delta / ChunkProcessingState — additional coverage
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn empty_delta_returns_none() {
    let choice = json!({ "delta": {} });
    let mut state = ChunkProcessingState::default();
    assert!(process_choice_delta(&choice, &mut state).is_none());
}

#[test]
fn delta_string_content_captured() {
    let choice = json!({ "delta": { "content": "hello" } });
    let mut state = ChunkProcessingState::default();
    let p = process_choice_delta(&choice, &mut state).unwrap();
    assert_eq!(p.content, "hello");
    assert!(p.thinking.is_empty());
    assert!(p.tool_calls_delta.is_none());
}

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

#[test]
fn delta_tool_calls_accumulated_into_state() {
    // A delta with only tool_calls emits an intermediate payload carrying the
    // partial tool_calls AND accumulates them into state for the final chunk.
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
    let payload = process_choice_delta(&choice, &mut state)
        .expect("tool_calls-only delta must emit an intermediate payload");
    assert!(payload.content.is_empty());
    assert!(payload.thinking.is_empty());
    assert!(
        payload.tool_calls_delta.is_some(),
        "intermediate payload must carry the partial tool_calls"
    );

    // Accumulated tool_calls should still be available for the final done chunk.
    let tc = state
        .take_tool_calls()
        .expect("state must hold accumulated tool_calls");
    let arr = tc.as_array().unwrap();
    assert_eq!(arr.len(), 1);
    let entry = &arr[0];
    // Ollama shape: id/type wrapper stripped, arguments is an object.
    assert!(entry.get("id").is_none(), "openai id must be stripped");
    assert!(entry.get("type").is_none(), "openai type must be stripped");
    let args = entry.get("function").unwrap().get("arguments").unwrap();
    assert!(args.is_object(), "arguments must be parsed into object");
    assert_eq!(args.get("tz").and_then(|v| v.as_str()), Some("UTC"));
}

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

#[test]
fn finish_reason_keeps_last_seen_non_null() {
    let mut state = ChunkProcessingState::default();
    let one = json!({ "delta": { "content": "x" }, "finish_reason": "stop" });
    let _ = process_choice_delta(&one, &mut state);
    let two = json!({ "delta": { "content": "y" }, "finish_reason": "tool_calls" });
    let _ = process_choice_delta(&two, &mut state);
    assert_eq!(state.finish_reason(), Some("tool_calls"));
}

#[test]
fn extract_first_choice_handles_missing_or_empty() {
    assert!(extract_first_choice(&json!({})).is_none());
    assert!(extract_first_choice(&json!({"choices": []})).is_none());
    let chunk = json!({"choices": [{"delta": {"content": "ok"}}]});
    assert!(extract_first_choice(&chunk).is_some());
}

// ════════════════════════════════════════════════════════════════════════════
// create_ollama_streaming_chunk — chat
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn chat_streaming_chunk_basic_shape() {
    let c = create_ollama_streaming_chunk("llama3", "hi", true, false, None, "");
    assert_eq!(c.get("model").and_then(|v| v.as_str()), Some("llama3"));
    assert_eq!(c.get("done").and_then(|v| v.as_bool()), Some(false));
    let created = c.get("created_at").and_then(|v| v.as_str()).unwrap();
    chrono::DateTime::parse_from_rfc3339(created).expect("created_at must be RFC3339");
    let msg = c.get("message").and_then(|v| v.as_object()).unwrap();
    assert_eq!(msg.get("role").and_then(|v| v.as_str()), Some("assistant"));
    assert_eq!(msg.get("content").and_then(|v| v.as_str()), Some("hi"));
    assert!(c.get("response").is_none(), "chat must not emit response");
}

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

// ════════════════════════════════════════════════════════════════════════════
// create_ollama_streaming_chunk — generate
// ════════════════════════════════════════════════════════════════════════════

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

// ════════════════════════════════════════════════════════════════════════════
// create_error_chunk
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn error_chunk_chat_is_bare_error_object() {
    let c = create_error_chunk("boom");
    assert_eq!(c.get("error").and_then(|v| v.as_str()), Some("boom"));
    // bare {"error":"…"} — no extra fields
    assert!(c.get("done").is_none());
    assert!(c.get("message").is_none());
    assert!(c.get("model").is_none());
}

#[test]
fn error_chunk_generate_is_bare_error_object() {
    let c = create_error_chunk("boom");
    assert_eq!(c.get("error").and_then(|v| v.as_str()), Some("boom"));
    // bare {"error":"…"} — no extra fields
    assert!(c.get("done").is_none());
    assert!(c.get("response").is_none());
    assert!(c.get("model").is_none());
}

// ════════════════════════════════════════════════════════════════════════════
// create_cancellation_chunk
// ════════════════════════════════════════════════════════════════════════════

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

#[test]
fn cancellation_chunk_chat_has_empty_content() {
    let c = create_cancellation_chunk("m", Duration::from_millis(50), 7, None, true);
    assert_eq!(c.get("done").and_then(|v| v.as_bool()), Some(true));
    assert!(
        c.get("done_reason").is_none(),
        "done_reason must be omitted on cancellation; ollama spec uses stop/length only"
    );
    let content = c
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|v| v.as_str())
        .unwrap();
    assert_eq!(
        content, "",
        "cancellation chunk must leave content empty per ollama spec; got {:?}",
        content
    );
    assert_six_timings(&c);
}

#[test]
fn cancellation_chunk_generate_has_empty_response() {
    let c = create_cancellation_chunk("m", Duration::from_millis(50), 3, None, false);
    assert_eq!(c.get("done").and_then(|v| v.as_bool()), Some(true));
    assert!(
        c.get("done_reason").is_none(),
        "done_reason must be omitted on cancellation; ollama spec uses stop/length only"
    );
    let response = c.get("response").and_then(|v| v.as_str()).unwrap();
    assert_eq!(
        response, "",
        "cancellation chunk must leave response empty per ollama spec; got {:?}",
        response
    );
    assert_six_timings(&c);
}

#[test]
fn cancellation_chunk_zero_tokens_still_empty_content() {
    let chat = create_cancellation_chunk("m", Duration::from_millis(10), 0, None, true);
    let chat_content = chat
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|v| v.as_str())
        .unwrap();
    assert_eq!(chat_content, "");
    assert!(chat.get("done_reason").is_none());

    let generate = create_cancellation_chunk("m", Duration::from_millis(10), 0, None, false);
    let generate_response = generate.get("response").and_then(|v| v.as_str()).unwrap();
    assert_eq!(generate_response, "");
    assert!(generate.get("done_reason").is_none());
}

#[test]
fn cancellation_chunk_chat_embeds_buffered_tool_calls() {
    // Mirror the success path: an interrupted stream that already buffered
    // tool_calls must surface them on the final done chunk per the chat spec.
    let mut state = ChunkProcessingState::default();
    let delta = json!({
        "delta": {
            "tool_calls": [{
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_temp", "arguments": "{\"city\":\"NYC\"}"}
            }]
        }
    });
    let _ = process_choice_delta(&delta, &mut state);
    let buffered = state
        .take_tool_calls()
        .expect("delta must accumulate tool_calls");

    let c = create_cancellation_chunk(
        "m",
        Duration::from_millis(50),
        7,
        Some(buffered.clone()),
        true,
    );
    let msg = c
        .get("message")
        .expect("chat cancellation must carry a message");
    let calls = msg
        .get("tool_calls")
        .and_then(|v| v.as_array())
        .expect("buffered tool_calls must surface on cancellation");
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[0].get("function").unwrap().get("name").unwrap(),
        "get_temp"
    );
    assert_six_timings(&c);
}

#[test]
fn cancellation_chunk_chat_omits_tool_calls_when_none() {
    let c = create_cancellation_chunk("m", Duration::from_millis(50), 7, None, true);
    let msg = c.get("message").unwrap();
    assert!(
        msg.get("tool_calls").is_none(),
        "no buffered tool_calls means no tool_calls field on the message"
    );
}

#[test]
fn cancellation_chunk_generate_never_has_tool_calls() {
    // Generate has no tool_calls semantically — defensively drop them even if
    // a caller passes Some(..).
    let tc = json!([{
        "function": {"name": "x", "arguments": {"k": "v"}}
    }]);
    let c = create_cancellation_chunk("m", Duration::from_millis(50), 3, Some(tc), false);
    assert!(c.get("message").is_none(), "generate has no message");
    assert!(
        c.get("tool_calls").is_none(),
        "generate chunks must never carry tool_calls"
    );
}

// ════════════════════════════════════════════════════════════════════════════
// create_final_chunk
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn final_chunk_chat_no_done_reason_omits_field_with_timings() {
    let c = create_final_chunk(FinalChunkParams {
        model_name: "m",
        duration: Duration::from_millis(120),
        chunk_count: 4,
        is_chat: true,
        done_reason: None,
        tool_calls: None,
    });
    assert_eq!(c.get("done").and_then(|v| v.as_bool()), Some(true));
    assert!(
        c.get("done_reason").is_none(),
        "done_reason must be omitted when upstream gave no finish_reason"
    );
    assert!(c.get("message").is_some());
    assert_six_timings(&c);
    assert!(c.get("context").is_none(), "chat must not emit context");
}

#[test]
fn final_chunk_chat_propagates_done_reason_length() {
    let c = create_final_chunk(FinalChunkParams {
        model_name: "m",
        duration: Duration::from_millis(10),
        chunk_count: 1,
        is_chat: true,
        done_reason: Some("length"),
        tool_calls: None,
    });
    assert_eq!(
        c.get("done_reason").and_then(|v| v.as_str()),
        Some("length")
    );
}

#[test]
fn final_chunk_generate_omits_context_and_emits_timings() {
    let c = create_final_chunk(FinalChunkParams {
        model_name: "m",
        duration: Duration::from_millis(80),
        chunk_count: 6,
        is_chat: false,
        done_reason: None,
        tool_calls: None,
    });
    assert_eq!(c.get("done").and_then(|v| v.as_bool()), Some(true));
    assert!(
        c.get("done_reason").is_none(),
        "done_reason must be omitted when upstream gave no finish_reason"
    );
    assert!(c.get("message").is_none());
    assert!(
        c.get("context").is_none(),
        "generate streaming final chunk must not emit context"
    );
    assert_six_timings(&c);
}

#[test]
fn final_chunk_generate_propagates_done_reason_length() {
    let c = create_final_chunk(FinalChunkParams {
        model_name: "m",
        duration: Duration::from_millis(10),
        chunk_count: 1,
        is_chat: false,
        done_reason: Some("length"),
        tool_calls: None,
    });
    assert_eq!(
        c.get("done_reason").and_then(|v| v.as_str()),
        Some("length")
    );
}

// ════════════════════════════════════════════════════════════════════════════
// GAP C — streaming tool_calls accumulation
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn tool_calls_accumulated_across_three_deltas() {
    let delta_name = json!({
        "delta": {
            "tool_calls": [{
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_temperature", "arguments": ""}
            }]
        }
    });
    let delta_args_part1 = json!({
        "delta": {
            "tool_calls": [{
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"arguments": "{\"city\""}
            }]
        }
    });
    let delta_args_part2 = json!({
        "delta": {
            "tool_calls": [{
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"arguments": ":\"NYC\"}"}
            }]
        }
    });

    let mut state = ChunkProcessingState::default();

    // Each tool_calls-only delta now produces an intermediate payload AND
    // accumulates into state. The intermediate carries the per-delta fragment;
    // the accumulator merges them for the final chunk.
    assert!(process_choice_delta(&delta_name, &mut state).is_some());
    assert!(process_choice_delta(&delta_args_part1, &mut state).is_some());
    assert!(process_choice_delta(&delta_args_part2, &mut state).is_some());

    let tool_calls = state
        .take_tool_calls()
        .expect("accumulated tool_calls must be present");
    let arr = tool_calls.as_array().unwrap();
    assert_eq!(arr.len(), 1);
    let function = arr[0].get("function").unwrap();
    assert_eq!(function.get("index"), Some(&json!(0)));
    assert_eq!(function.get("name"), Some(&json!("get_temperature")));
    assert_eq!(function.get("arguments"), Some(&json!({"city": "NYC"})));

    assert!(
        state.take_tool_calls().is_none(),
        "take must consume the accumulated calls"
    );
}

#[test]
fn concurrent_tool_call_fragments_are_ordered_by_index() {
    let first_delta = json!({
        "delta": {
            "tool_calls": [
                {
                    "index": 1,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_conditions", "arguments": "{\"city\""}
                },
                {
                    "index": 0,
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "get_temperature", "arguments": "{\"city\""}
                }
            ]
        }
    });
    let second_delta = json!({
        "delta": {
            "tool_calls": [
                {
                    "index": 1,
                    "id": "call_1",
                    "type": "function",
                    "function": {"arguments": ":\"London\"}"}
                },
                {
                    "index": 0,
                    "id": "call_0",
                    "type": "function",
                    "function": {"arguments": ":\"New York\"}"}
                }
            ]
        }
    });

    let mut state = ChunkProcessingState::default();
    assert!(process_choice_delta(&first_delta, &mut state).is_some());
    assert!(process_choice_delta(&second_delta, &mut state).is_some());

    let tool_calls = state
        .take_tool_calls()
        .expect("accumulated tool_calls must be present");
    let calls = tool_calls.as_array().unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0]["function"]["index"], json!(0));
    assert_eq!(calls[0]["function"]["name"], json!("get_temperature"));
    assert_eq!(
        calls[0]["function"]["arguments"],
        json!({"city": "New York"})
    );
    assert_eq!(calls[1]["function"]["index"], json!(1));
    assert_eq!(calls[1]["function"]["name"], json!("get_conditions"));
    assert_eq!(calls[1]["function"]["arguments"], json!({"city": "London"}));
}

#[test]
fn tool_calls_in_final_chunk_when_provided() {
    let tc = json!([{
        "function": {"name": "get_temp", "arguments": {"city": "NYC"}}
    }]);
    let c = create_final_chunk(FinalChunkParams {
        model_name: "m",
        duration: Duration::from_millis(50),
        chunk_count: 3,
        is_chat: true,
        done_reason: Some("tool_calls"),
        tool_calls: Some(tc),
    });
    assert_eq!(c.get("done").and_then(|v| v.as_bool()), Some(true));
    let msg = c
        .get("message")
        .expect("message must be present in chat chunk");
    let calls = msg
        .get("tool_calls")
        .and_then(|v| v.as_array())
        .expect("tool_calls must be in the final chunk message");
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[0].get("function").unwrap().get("name").unwrap(),
        "get_temp"
    );
}

#[test]
fn final_chunk_without_tool_calls_has_no_tool_calls_field() {
    let c = create_final_chunk(FinalChunkParams {
        model_name: "m",
        duration: Duration::from_millis(50),
        chunk_count: 2,
        is_chat: true,
        done_reason: None,
        tool_calls: None,
    });
    let msg = c.get("message").unwrap();
    assert!(
        msg.get("tool_calls").is_none(),
        "tool_calls must be absent when none were accumulated"
    );
}

#[test]
fn content_and_thinking_deltas_still_stream_mid_message() {
    // Ensure content and thinking chunks still emit immediately (per-delta),
    // only tool_calls are deferred to the final chunk.
    let choice_with_content = json!({ "delta": { "content": "some text" } });
    let choice_with_thinking = json!({ "delta": { "reasoning": "my thought" } });
    let mut state = ChunkProcessingState::default();

    let p1 = process_choice_delta(&choice_with_content, &mut state);
    assert!(p1.is_some(), "content delta must produce a payload");
    assert_eq!(p1.unwrap().content, "some text");

    let p2 = process_choice_delta(&choice_with_thinking, &mut state);
    assert!(p2.is_some(), "reasoning delta must produce a payload");
    assert_eq!(p2.unwrap().thinking, "my thought");

    // No tool_calls were accumulated.
    assert!(state.take_tool_calls().is_none());
}

// ════════════════════════════════════════════════════════════════════════════
// T7 — tool-call-only / finish-only / no-op delta handling
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn process_choice_delta_emits_intermediate_chunk_for_tool_calls_only_delta() {
    let choice = json!({
        "delta": {
            "tool_calls": [{
                "index": 0,
                "function": {"name": "f"}
            }]
        }
    });
    let mut state = ChunkProcessingState::default();
    let payload = process_choice_delta(&choice, &mut state)
        .expect("tool_calls-only delta must produce an intermediate payload");
    assert!(payload.content.is_empty());
    assert!(payload.thinking.is_empty());
    let tc = payload
        .tool_calls_delta
        .expect("tool_calls_delta must be set");
    let arr = tc.as_array().expect("tool_calls must be an array");
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["function"]["name"], json!("f"));
}

#[test]
fn process_choice_delta_returns_none_for_finish_reason_only_delta() {
    let choice = json!({ "delta": {}, "finish_reason": "stop" });
    let mut state = ChunkProcessingState::default();
    assert!(
        process_choice_delta(&choice, &mut state).is_none(),
        "finish_reason-only delta produces no intermediate chunk"
    );
    assert_eq!(state.finish_reason(), Some("stop"));
}

#[test]
fn process_choice_delta_returns_none_for_empty_delta() {
    let choice = json!({ "delta": {} });
    let mut state = ChunkProcessingState::default();
    assert!(process_choice_delta(&choice, &mut state).is_none());
}
