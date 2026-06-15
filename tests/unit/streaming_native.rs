use serde_json::{Value, json};

use crate::streaming::chunks::ChunkProcessingState;
use crate::streaming::native::{
    NativeEvent, map_native_event, parse_chat_end, parse_native_sse_message,
};

fn map(event_type: &str, data: Value) -> (NativeEvent, ChunkProcessingState) {
    let mut state = ChunkProcessingState::default();
    let event = map_native_event(event_type, &data, &mut state);
    (event, state)
}

#[test]
fn reasoning_delta_maps_to_thinking() {
    let (event, _) = map(
        "reasoning.delta",
        json!({ "type": "reasoning.delta", "content": "Need to" }),
    );
    match event {
        NativeEvent::Delta(p) => {
            assert_eq!(p.thinking, "Need to");
            assert!(p.content.is_empty());
            assert!(p.tool_calls_delta.is_none());
        }
        _ => panic!("expected Delta"),
    }
}

#[test]
fn message_delta_maps_to_content() {
    let (event, _) = map(
        "message.delta",
        json!({ "type": "message.delta", "content": "The current" }),
    );
    match event {
        NativeEvent::Delta(p) => {
            assert_eq!(p.content, "The current");
            assert!(p.thinking.is_empty());
        }
        _ => panic!("expected Delta"),
    }
}

#[test]
fn tool_call_arguments_accumulates_and_surfaces_delta() {
    let data = json!({
        "type": "tool_call.arguments",
        "tool": "model_search",
        "arguments": { "sort": "trendingScore", "limit": 1 },
        "provider_info": { "type": "ephemeral_mcp", "server_label": "huggingface" }
    });
    let mut state = ChunkProcessingState::default();
    let event = map_native_event("tool_call.arguments", &data, &mut state);

    match event {
        NativeEvent::Delta(p) => {
            let delta = p.tool_calls_delta.expect("tool_calls_delta present");
            let arr = delta.as_array().expect("array");
            assert_eq!(arr[0]["function"]["name"], json!("model_search"));
            assert_eq!(
                arr[0]["function"]["arguments"],
                json!({ "sort": "trendingScore", "limit": 1 })
            );
        }
        _ => panic!("expected Delta"),
    }

    // The accumulator holds the merged call for the final chunk.
    let final_calls = state.take_tool_calls().expect("accumulated tool calls");
    assert_eq!(final_calls[0]["function"]["name"], json!("model_search"));
}

#[test]
fn tool_call_success_also_accumulates() {
    let data = json!({
        "type": "tool_call.success",
        "tool": "model_search",
        "arguments": { "sort": "trendingScore", "limit": 1 },
        "output": "[{\"type\":\"text\",\"text\":\"...\"}]",
        "provider_info": { "type": "ephemeral_mcp", "server_label": "huggingface" }
    });
    let (event, _) = map("tool_call.success", data);
    assert!(matches!(event, NativeEvent::Delta(_)));
}

#[test]
fn error_event_surfaces_payload() {
    let data = json!({
        "type": "error",
        "error": {
            "type": "invalid_request",
            "message": "\"model\" is required",
            "code": "missing_required_parameter",
            "param": "model"
        }
    });
    let (event, _) = map("error", data);
    match event {
        NativeEvent::Error(e) => {
            assert_eq!(e.error_type, "invalid_request");
            assert_eq!(e.message, "\"model\" is required");
            assert_eq!(e.code.as_deref(), Some("missing_required_parameter"));
            assert_eq!(e.param.as_deref(), Some("model"));
            let line = e.to_message();
            assert!(line.contains("invalid_request"));
            assert!(line.contains("missing_required_parameter"));
            assert!(line.contains("model"));
        }
        _ => panic!("expected Error"),
    }
}

#[test]
fn chat_end_extracts_stats() {
    let data = json!({
        "type": "chat.end",
        "result": {
            "model_instance_id": "openai/gpt-oss-20b",
            "output": [{ "type": "message", "content": "done" }],
            "stats": {
                "input_tokens": 329,
                "total_output_tokens": 268,
                "reasoning_output_tokens": 5,
                "tokens_per_second": 43.73,
                "time_to_first_token_seconds": 0.781
            }
        }
    });
    let (event, _) = map("chat.end", data);
    match event {
        NativeEvent::End(end) => {
            assert_eq!(end.done_reason, "stop");
            let stats = end.stats.expect("stats present");
            assert_eq!(stats["input_tokens"], json!(329));
            assert_eq!(stats["total_output_tokens"], json!(268));
            // result is the full aggregate, usable by the non-streaming converter.
            assert_eq!(end.result["output"][0]["content"], json!("done"));
        }
        _ => panic!("expected End"),
    }
}

#[test]
fn chat_end_extracts_stats_without_response_id() {
    let end = parse_chat_end(&json!({
        "result": { "output": [], "stats": {} }
    }));
    assert!(end.stats.is_some());
}

#[test]
fn boundary_events_are_ignored() {
    for ty in [
        "chat.start",
        "model_load.start",
        "model_load.progress",
        "model_load.end",
        "prompt_processing.start",
        "prompt_processing.progress",
        "prompt_processing.end",
        "reasoning.start",
        "reasoning.end",
        "message.start",
        "message.end",
        "tool_call.start",
        "tool_call.failure",
    ] {
        let (event, _) = map(ty, json!({ "type": ty }));
        assert!(
            matches!(event, NativeEvent::Ignore),
            "expected {ty} to be ignored"
        );
    }
}

#[test]
fn parse_sse_message_splits_event_and_data() {
    let block = "event: message.delta\ndata: {\"type\":\"message.delta\",\"content\":\"hi\"}";
    let (event_type, data) = parse_native_sse_message(block).expect("parsed");
    assert_eq!(event_type, "message.delta");
    assert_eq!(data["content"], json!("hi"));
}

#[test]
fn parse_sse_message_tolerates_crlf() {
    let block = "event: chat.start\r\ndata: {\"type\":\"chat.start\"}\r\n";
    let (event_type, data) = parse_native_sse_message(block).expect("parsed");
    assert_eq!(event_type, "chat.start");
    assert_eq!(data["type"], json!("chat.start"));
}

#[test]
fn parse_sse_message_concatenates_multiline_data() {
    let block =
        "event: message.delta\ndata: {\"type\":\"message.delta\",\ndata: \"content\":\"x\"}";
    let (_, data) = parse_native_sse_message(block).expect("parsed");
    assert_eq!(data["content"], json!("x"));
}

#[test]
fn parse_sse_message_rejects_missing_event_or_data() {
    assert!(parse_native_sse_message("data: {}").is_none());
    assert!(parse_native_sse_message("event: message.delta").is_none());
    assert!(parse_native_sse_message("event: message.delta\ndata: not-json").is_none());
}
