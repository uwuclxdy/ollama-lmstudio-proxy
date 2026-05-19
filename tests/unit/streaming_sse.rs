use std::sync::atomic::Ordering;

use serde_json::json;

use crate::constants::{SSE_DATA_PREFIX, SSE_DONE_MESSAGE, SSE_MESSAGE_BOUNDARY};
use crate::streaming::chunks::{ChunkProcessingState, extract_first_choice, process_choice_delta};
use crate::streaming::recovery::recover_json_from_chunk;

// ════════════════════════════════════════════════════════════════════════════
// SSE buffer parsing logic — replicated from the inline stream task
//
// The actual parse loop lives in an async closure inside handle_streaming_response
// and cannot be called directly. These tests replicate the same logic using the
// same constants so any constant-level regression is caught.
// ════════════════════════════════════════════════════════════════════════════

/// Simulates one pass of the SSE buffer parser used in the stream task.
/// Returns the parsed data payloads and whether [DONE] was seen.
fn parse_sse_buffer(input: &str) -> (Vec<String>, bool) {
    let mut buffer = input.to_string();
    let mut payloads = Vec::new();
    let mut done_seen = false;

    let mut cursor = 0;
    while let Some(rel_pos) = buffer[cursor..].find(SSE_MESSAGE_BOUNDARY) {
        let boundary_pos = cursor + rel_pos;
        let message_text = &buffer[cursor..boundary_pos];
        cursor = boundary_pos + SSE_MESSAGE_BOUNDARY.len();

        if message_text.bytes().all(|b| b.is_ascii_whitespace()) {
            continue;
        }

        if let Some(data_content) = message_text.strip_prefix(SSE_DATA_PREFIX) {
            if data_content.trim() == SSE_DONE_MESSAGE {
                done_seen = true;
                break;
            }
            payloads.push(data_content.to_string());
        }
    }

    if cursor > 0 {
        buffer.drain(..cursor);
    }

    (payloads, done_seen)
}

#[test]
fn single_data_event_parsed() {
    let input = "data: {\"id\":\"1\"}\n\n";
    let (payloads, done) = parse_sse_buffer(input);
    assert_eq!(payloads.len(), 1);
    assert!(payloads[0].contains("\"id\""));
    assert!(!done);
}

#[test]
fn done_message_terminates_parsing() {
    let input = "data: {\"a\":1}\n\ndata: [DONE]\n\n";
    let (payloads, done) = parse_sse_buffer(input);
    assert_eq!(payloads.len(), 1, "payload before [DONE] must be collected");
    assert!(done, "[DONE] must set done flag");
}

#[test]
fn done_before_data_terminates_immediately() {
    let input = "data: [DONE]\n\ndata: {\"a\":1}\n\n";
    let (payloads, done) = parse_sse_buffer(input);
    assert_eq!(payloads.len(), 0, "data after [DONE] must be ignored");
    assert!(done);
}

#[test]
fn blank_line_only_events_are_skipped() {
    let input = "\n\n\n\n";
    let (payloads, done) = parse_sse_buffer(input);
    assert!(payloads.is_empty());
    assert!(!done);
}

#[test]
fn multiple_events_all_collected() {
    let chunks: Vec<String> = (0..5)
        .map(|i| format!("data: {{\"idx\":{}}}\n\n", i))
        .collect();
    let input = chunks.join("");
    let (payloads, done) = parse_sse_buffer(&input);
    assert_eq!(payloads.len(), 5);
    assert!(!done);
    for (i, payload) in payloads.iter().enumerate() {
        assert!(
            payload.contains(&i.to_string()),
            "payload {i} must contain index {i}"
        );
    }
}

#[test]
fn partial_buffer_without_boundary_yields_no_payloads() {
    // If the stream is split mid-event (no \n\n yet), nothing must be emitted
    let input = "data: {\"incomplete\"";
    let (payloads, done) = parse_sse_buffer(input);
    assert!(
        payloads.is_empty(),
        "incomplete event must not emit a payload"
    );
    assert!(!done);
}

#[test]
fn event_without_data_prefix_is_not_collected() {
    // Lines without "data: " prefix are skipped (logged as non-standard)
    let input = "event: something\n\ndata: {\"ok\":true}\n\n";
    let (payloads, done) = parse_sse_buffer(input);
    // Only the "data: " line must be in payloads
    assert_eq!(payloads.len(), 1);
    assert!(payloads[0].contains("\"ok\""));
    assert!(!done);
}

#[test]
fn done_with_surrounding_whitespace_recognized() {
    // data_content.trim() == SSE_DONE_MESSAGE  — leading/trailing spaces count
    let input = "data:  [DONE] \n\n";
    // strip_prefix(SSE_DATA_PREFIX) gives " [DONE] ", trim gives "[DONE]"
    let (payloads, done) = parse_sse_buffer(input);
    assert_eq!(payloads.len(), 0);
    assert!(done, "trimmed [DONE] must terminate stream");
}

#[test]
fn large_single_chunk_with_many_events_processed() {
    let mut big = String::new();
    for i in 0..100 {
        big.push_str(&format!("data: {{\"n\":{}}}\n\n", i));
    }
    big.push_str("data: [DONE]\n\n");
    let (payloads, done) = parse_sse_buffer(&big);
    assert_eq!(payloads.len(), 100);
    assert!(done);
}

#[test]
fn whitespace_only_message_between_events_is_skipped() {
    let input = "data: {\"a\":1}\n\n   \n\ndata: {\"b\":2}\n\n";
    let (payloads, _) = parse_sse_buffer(input);
    assert_eq!(payloads.len(), 2);
}

// ════════════════════════════════════════════════════════════════════════════
// SSE constants correctness
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn sse_data_prefix_is_data_colon_space() {
    assert_eq!(SSE_DATA_PREFIX, "data: ");
}

#[test]
fn sse_done_message_is_bracket_done_bracket() {
    assert_eq!(SSE_DONE_MESSAGE, "[DONE]");
}

#[test]
fn sse_message_boundary_is_double_newline() {
    assert_eq!(SSE_MESSAGE_BOUNDARY, "\n\n");
}

// ════════════════════════════════════════════════════════════════════════════
// STREAM_COUNTER — monotonic increment
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn stream_counter_increments_each_read() {
    let a = super::STREAM_COUNTER.fetch_add(1, Ordering::Relaxed);
    let b = super::STREAM_COUNTER.fetch_add(1, Ordering::Relaxed);
    assert!(b > a, "STREAM_COUNTER must monotonically increase");
}

#[test]
fn stream_id_modulo_stays_below_million() {
    // The stream task does: fetch_add(1) % 1_000_000
    // Verify the modulo arithmetic keeps the id < 1_000_000
    for raw in [0u64, 999_999, 1_000_000, 2_000_001, u64::MAX] {
        let stream_id = raw % 1_000_000;
        assert!(
            stream_id < 1_000_000,
            "stream_id {stream_id} must be < 1_000_000"
        );
    }
}

// ════════════════════════════════════════════════════════════════════════════
// STREAM_START_LOADING_THRESHOLD_MS — internal constant
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn loading_threshold_is_500ms() {
    assert_eq!(super::STREAM_START_LOADING_THRESHOLD_MS, 500);
}

// ════════════════════════════════════════════════════════════════════════════
// Integration: parse → extract_first_choice → process_choice_delta pipeline
//
// These replicate the core processing path inside the stream task without
// spinning up an actual HTTP server.
// ════════════════════════════════════════════════════════════════════════════

fn make_lm_delta_chunk(content: &str) -> String {
    let payload = json!({
        "choices": [{
            "delta": { "content": content },
            "finish_reason": null
        }]
    });
    format!("data: {}\n\n", payload)
}

fn make_lm_done_chunk(finish_reason: &str) -> String {
    let payload = json!({
        "choices": [{
            "delta": {},
            "finish_reason": finish_reason
        }]
    });
    format!("data: {}\n\n", payload)
}

#[test]
fn pipeline_extracts_content_from_sse_chunk() {
    let raw = make_lm_delta_chunk("Hello");
    let (payloads, _) = parse_sse_buffer(&raw);
    assert_eq!(payloads.len(), 1);

    let parsed: serde_json::Value = serde_json::from_str(&payloads[0]).unwrap();
    let choice = extract_first_choice(&parsed).unwrap();
    let mut state = ChunkProcessingState::default();
    let delta = process_choice_delta(choice, &mut state).unwrap();
    assert_eq!(delta.content, "Hello");
}

#[test]
fn pipeline_finish_reason_propagated_through_state() {
    let mut state = ChunkProcessingState::default();

    let chunk1 = make_lm_delta_chunk("token");
    let (payloads1, _) = parse_sse_buffer(&chunk1);
    let p1: serde_json::Value = serde_json::from_str(&payloads1[0]).unwrap();
    let _ = process_choice_delta(extract_first_choice(&p1).unwrap(), &mut state);
    assert!(state.finish_reason().is_none());

    let chunk2 = make_lm_done_chunk("stop");
    let (payloads2, _) = parse_sse_buffer(&chunk2);
    let p2: serde_json::Value = serde_json::from_str(&payloads2[0]).unwrap();
    let _ = process_choice_delta(extract_first_choice(&p2).unwrap(), &mut state);
    assert_eq!(state.finish_reason(), Some("stop"));
}

#[test]
fn pipeline_with_recovery_on_truncated_json() {
    // Simulates the recovery path: data content is not valid top-level JSON
    // but recover_json_from_chunk can salvage the choices array.
    let malformed = r#"data: NOISE{"choices":[{"delta":{"content":"recovered"},"finish_reason":null}]}NOISE\n\n"#;
    let data = malformed.strip_prefix(SSE_DATA_PREFIX).unwrap_or(malformed);

    // Direct parse fails
    assert!(serde_json::from_str::<serde_json::Value>(data).is_err());

    // Recovery path
    let recovered = recover_json_from_chunk(data);
    assert!(
        recovered.is_some(),
        "recovery must salvage the choices object"
    );
    let v = recovered.unwrap();
    let choice = extract_first_choice(&v);
    assert!(
        choice.is_some(),
        "choices must be extractable after recovery"
    );
}

#[test]
fn pipeline_tool_calls_survive_full_pass() {
    let payload = json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "fn", "arguments": "{\"k\":\"v\"}"}
                }]
            },
            "finish_reason": null
        }]
    });
    let raw = format!("data: {}\n\n", payload);
    let (payloads, _) = parse_sse_buffer(&raw);
    let parsed: serde_json::Value = serde_json::from_str(&payloads[0]).unwrap();
    let choice = extract_first_choice(&parsed).unwrap();
    let mut state = ChunkProcessingState::default();
    let delta = process_choice_delta(choice, &mut state).unwrap();
    assert!(
        delta.tool_calls_delta.is_some(),
        "tool_calls must survive the parse → process pipeline"
    );
}

#[test]
fn pipeline_thinking_field_routed_separately() {
    let payload = json!({
        "choices": [{
            "delta": {
                "content": "answer",
                "reasoning": "thought"
            },
            "finish_reason": null
        }]
    });
    let raw = format!("data: {}\n\n", payload);
    let (payloads, _) = parse_sse_buffer(&raw);
    let parsed: serde_json::Value = serde_json::from_str(&payloads[0]).unwrap();
    let choice = extract_first_choice(&parsed).unwrap();
    let mut state = ChunkProcessingState::default();
    let delta = process_choice_delta(choice, &mut state).unwrap();
    assert_eq!(delta.content, "answer");
    assert_eq!(delta.thinking, "thought");
}

#[test]
fn pipeline_stream_split_across_two_chunks_reassembled() {
    // First chunk ends mid-event; second chunk completes it.
    let full_payload = json!({"choices":[{"delta":{"content":"x"},"finish_reason":null}]});
    let full_event = format!("data: {}\n\n", full_payload);

    // Split at arbitrary byte position
    let split_at = full_event.len() / 2;
    let first_half = &full_event[..split_at];
    let second_half = &full_event[split_at..];

    let mut buffer = String::new();
    buffer.push_str(first_half);

    // First half has no complete \n\n boundary yet
    let (p1, d1) = parse_sse_buffer(&buffer);

    // Simulate the drain: keep what was not consumed
    // (parse_sse_buffer leaves unconsumed tail in place; we simulate similarly)
    // If no complete event found, the buffer should yield nothing
    if p1.is_empty() && !d1 {
        buffer.push_str(second_half);
        let (p2, _) = parse_sse_buffer(&buffer);
        assert!(
            !p2.is_empty(),
            "after second chunk, complete event must be parsed"
        );
    }
    // If the split happened after the \n\n then p1 would have one entry — also fine
}
