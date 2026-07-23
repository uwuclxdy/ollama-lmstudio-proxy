use std::time::Duration;

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
fn pipeline_tool_calls_accumulated_into_state() {
    // tool_calls fragments accumulate into ChunkProcessingState without
    // surfacing an intermediate payload; the stream driver emits the merged
    // calls once, right before the final chunk.
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
    assert!(
        process_choice_delta(choice, &mut state).is_none(),
        "tool_calls-only delta must not surface an intermediate payload"
    );
    // Tool call is in state, ready for the pre-final emission.
    let tc = state
        .take_tool_calls()
        .expect("tool_calls must be in state after processing");
    let arr = tc.as_array().unwrap();
    assert_eq!(arr.len(), 1, "one tool call must be accumulated");
    assert_eq!(arr[0]["function"]["name"], "fn");
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

// ════════════════════════════════════════════════════════════════════════════
// PassthroughProtocol — injected cancel/timeout/error frame shaping
// ════════════════════════════════════════════════════════════════════════════

use crate::streaming::sse::PassthroughProtocol;

#[test]
fn passthrough_protocol_maps_endpoints() {
    use PassthroughProtocol::*;
    assert_eq!(
        PassthroughProtocol::from_endpoint("/v1/messages"),
        Anthropic
    );
    assert_eq!(
        PassthroughProtocol::from_endpoint("/v1/messages/count_tokens"),
        Anthropic
    );
    assert_eq!(
        PassthroughProtocol::from_endpoint("/v1/responses"),
        Responses
    );
    assert_eq!(
        PassthroughProtocol::from_endpoint("/v1/chat/completions"),
        OpenAi
    );
    assert_eq!(
        PassthroughProtocol::from_endpoint("/v1/completions"),
        OpenAi
    );
    assert_eq!(PassthroughProtocol::from_endpoint("/v1/embeddings"), OpenAi);
    // /api/v0 is LM Studio's OpenAI-compat surface; /api/v1+ is native events.
    assert_eq!(
        PassthroughProtocol::from_endpoint("/api/v0/chat/completions"),
        OpenAi
    );
    assert_eq!(PassthroughProtocol::from_endpoint("/api/v1/chat"), NativeV1);
}

/// Split a framed SSE block into (event-name, parsed data payload).
fn parse_frame(frame: &str) -> (Option<String>, serde_json::Value) {
    assert!(
        frame.ends_with("\n\n"),
        "frame must be a complete SSE block"
    );
    let mut event = None;
    let mut data = String::new();
    for line in frame.lines() {
        if let Some(rest) = line.strip_prefix("event: ") {
            event = Some(rest.to_string());
        } else if let Some(rest) = line.strip_prefix("data: ") {
            data.push_str(rest);
        }
    }
    (
        event,
        serde_json::from_str(&data).expect("frame data is JSON"),
    )
}

#[test]
fn anthropic_frame_is_named_error_event_with_typed_body() {
    let (event, data) = parse_frame(&PassthroughProtocol::Anthropic.frame_error("boom"));
    assert_eq!(event.as_deref(), Some("error"));
    assert_eq!(data["type"], json!("error"));
    assert_eq!(data["error"]["type"], json!("api_error"));
    assert_eq!(data["error"]["message"], json!("boom"));
}

#[test]
fn openai_frame_is_bare_data_with_typed_error_object() {
    let (event, data) = parse_frame(&PassthroughProtocol::OpenAi.frame_error("boom"));
    assert_eq!(event, None, "OpenAI SSE uses bare data: lines");
    assert_eq!(data["error"]["message"], json!("boom"));
    assert_eq!(data["error"]["type"], json!("server_error"));
}

#[test]
fn responses_frame_is_response_failed_event() {
    let (event, data) = parse_frame(&PassthroughProtocol::Responses.frame_error("boom"));
    assert_eq!(event.as_deref(), Some("response.failed"));
    assert_eq!(data["type"], json!("response.failed"));
    assert_eq!(data["response"]["status"], json!("failed"));
    assert_eq!(data["response"]["error"]["message"], json!("boom"));
}

#[test]
fn native_frame_is_named_error_event_with_native_type() {
    let (event, data) = parse_frame(&PassthroughProtocol::NativeV1.frame_error("boom"));
    assert_eq!(event.as_deref(), Some("error"));
    assert_eq!(data["type"], json!("error"));
    assert_eq!(data["error"]["type"], json!("internal_error"));
    assert_eq!(data["error"]["message"], json!("boom"));
}

// ════════════════════════════════════════════════════════════════════════════
// handle_passthrough_streaming_response — wiring at each frame_error call site
//
// `frame_error`/`from_endpoint` above are pure functions already covered; these
// drive the real function end-to-end (a real `reqwest::Response` from a raw
// one-shot TCP server) so a call site reverted to a bare ollama-shaped string
// instead of `protocol.frame_error(...)` would be caught.
// ════════════════════════════════════════════════════════════════════════════

use crate::constants::{ERROR_CANCELLED, ERROR_TIMEOUT};
use crate::streaming::sse::handle_passthrough_streaming_response;
use tokio_util::sync::CancellationToken;

/// One-shot raw TCP responder: accepts a single connection, drains the
/// request, then hands the socket to `write_response` — which controls
/// exactly what bytes (and when) the client sees. Used to manufacture a real
/// `reqwest::Response` backed by a stalled or truncated body, which no
/// wiremock `ResponseTemplate` can produce.
async fn spawn_raw_response<F, Fut>(write_response: F) -> reqwest::Response
where
    F: FnOnce(tokio::net::TcpStream) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = ()> + Send,
{
    use tokio::io::AsyncReadExt;
    use tokio::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local_addr");

    tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.expect("accept");
        let mut buf = vec![0u8; 4096];
        let _ = stream.read(&mut buf).await; // drain the request
        write_response(stream).await;
    });

    reqwest::Client::new()
        .get(format!("http://{}/", addr))
        .send()
        .await
        .expect("GET raw one-shot server")
}

async fn write_minimal_200(mut stream: tokio::net::TcpStream) {
    use tokio::io::AsyncWriteExt;
    let _ = stream
        .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\n{}")
        .await;
}

/// Declares a body but sends none of it before closing — hyper detects the
/// socket closing before content-length is satisfied and surfaces an
/// incomplete-message error on the body stream. Sending zero body bytes (not
/// just fewer than declared) keeps this a pure error with no raw passthrough
/// chunk ahead of it to pollute the frame the test parses.
async fn write_truncated_body(mut stream: tokio::net::TcpStream) {
    use tokio::io::AsyncWriteExt;
    let _ = stream
        .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 10000\r\n\r\n")
        .await;
    let _ = stream.shutdown().await;
}

/// Sends only headers, then stalls well past the caller's `stream_timeout_seconds`.
async fn write_then_stall(mut stream: tokio::net::TcpStream) {
    use tokio::io::AsyncWriteExt;
    let _ = stream
        .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 5\r\n\r\n")
        .await;
    tokio::time::sleep(Duration::from_secs(2)).await;
}

/// Collect a streamed axum response body and parse it as a single SSE frame
/// via `parse_frame` above.
async fn collect_frame(response: axum::response::Response) -> (Option<String>, serde_json::Value) {
    use http_body_util::BodyExt;
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("collect body")
        .to_bytes();
    let text = String::from_utf8(bytes.to_vec()).expect("utf8 frame");
    parse_frame(&text)
}

#[tokio::test]
async fn passthrough_cancel_wires_real_frame_error() {
    // Pre-cancel: `tokio::select! { biased; ... }` polls the cancellation arm
    // first on every iteration, so an already-cancelled token deterministically
    // wins the very first loop iteration regardless of the mocked response.
    let token = CancellationToken::new();
    token.cancel();

    let response = spawn_raw_response(write_minimal_200).await;
    let result =
        handle_passthrough_streaming_response(response, PassthroughProtocol::Anthropic, token, 60)
            .await
            .expect("cancelled passthrough stream must still build a response");

    let (event, data) = collect_frame(result).await;
    assert_eq!(event.as_deref(), Some("error"));
    assert_eq!(data["error"]["type"], json!("api_error"));
    assert_eq!(data["error"]["message"], json!(ERROR_CANCELLED));
}

#[tokio::test]
async fn passthrough_midstream_error_wires_real_frame_error() {
    let response = spawn_raw_response(write_truncated_body).await;
    let result = handle_passthrough_streaming_response(
        response,
        PassthroughProtocol::NativeV1,
        CancellationToken::new(),
        60,
    )
    .await
    .expect("mid-stream network error must still build a response");

    let (event, data) = collect_frame(result).await;
    assert_eq!(event.as_deref(), Some("error"));
    assert_eq!(data["error"]["type"], json!("internal_error"));
    let message = data["error"]["message"].as_str().expect("message string");
    assert!(
        message.starts_with("streaming error: "),
        "mid-stream network error must be wrapped as 'streaming error: ...'; got {message}"
    );
}

#[tokio::test]
async fn passthrough_timeout_wires_real_frame_error() {
    let response = spawn_raw_response(write_then_stall).await;
    let result = handle_passthrough_streaming_response(
        response,
        PassthroughProtocol::OpenAi,
        CancellationToken::new(),
        1, // stream_timeout_seconds — well under the 2s stall above
    )
    .await
    .expect("timed-out passthrough stream must still build a response");

    let (event, data) = collect_frame(result).await;
    assert_eq!(event, None, "OpenAI SSE uses bare data: lines");
    assert_eq!(data["error"]["message"], json!(ERROR_TIMEOUT));
    assert_eq!(data["error"]["type"], json!("server_error"));
}
