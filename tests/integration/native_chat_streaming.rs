// Integration tests for the `--native-chat-streaming` auto mode.
//
// With the flag on, the chat handler routes based on the stream flag:
//   - `stream:true`  -> native `/api/v1/chat` (richer reasoning events, stats)
//   - `stream:false` -> the default OpenAI-compat `/api/v0/chat/completions`
//
// `use_native = use_native_chat || (native_chat_streaming && stream)` (see
// `src/api/ollama/chat.rs`). These tests pin BOTH halves of that expression:
// streaming lands on `/api/v1/chat`, and non-streaming stays on
// `/api/v0/chat/completions` instead of being pulled onto the native path.

use serde_json::{Value, json};
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, ResponseTemplate};

use crate::common::spawn_proxy_with_native_streaming;

/// GET /api/v1/models stub returning a single loaded model for resolution.
async fn mount_model_catalog(proxy: &crate::common::TestProxy, model_key: &str) {
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [{
                "key": model_key,
                "type": "llm",
                "publisher": "meta",
                "architecture": "llama",
                "format": "gguf",
                "quantization": { "name": "Q4_K_M", "bits_per_weight": 4.5 },
                "max_context_length": 8192,
                "loaded_instances": [
                    { "id": "inst-0", "config": { "context_length": 4096 } }
                ],
                "capabilities": { "vision": false, "trained_for_tool_use": true }
            }]
        })))
        .mount(&proxy.mock)
        .await;
}

/// Parse NDJSON body into a `Vec<Value>`, skipping blank lines.
fn parse_ndjson(text: &str) -> Vec<Value> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("valid JSON line"))
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// stream:true -> native /api/v1/chat
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn streaming_routes_to_native_v1_chat() {
    let p = spawn_proxy_with_native_streaming().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    // Pin the route: the streaming request must land on /api/v1/chat with
    // stream:true in the body — NOT on the OpenAI-compat v0 path.
    Mock::given(method("POST"))
        .and(path("/api/v1/chat"))
        .and(body_partial_json(json!({ "stream": true })))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(
                concat!(
                    "event: chat.start\ndata: {\"type\":\"chat.start\",\"model_instance_id\":\"llama3.1-8b-instruct\"}\n\n",
                    "event: message.delta\ndata: {\"type\":\"message.delta\",\"content\":\"Hello\"}\n\n",
                    "event: chat.end\ndata: {\"type\":\"chat.end\",\"result\":{\"model_instance_id\":\"llama3.1-8b-instruct\",\"output\":[{\"type\":\"message\",\"content\":\"Hello\"}],\"stats\":{\"input_tokens\":5,\"total_output_tokens\":1,\"tokens_per_second\":40.0,\"time_to_first_token_seconds\":0.1}}}\n\n"
                )
                .as_bytes(),
                "text/event-stream",
            ),
        )
        .expect(1)
        .mount(&p.mock)
        .await;

    // If routing misfires onto the v0 path, this mock would go unmatched and
    // the request would 404/500 — making the bug obvious.
    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"_unexpected_v0": true})))
        .expect(0)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "messages": [{ "role": "user", "content": "Hi" }],
            "stream": true
        }))
        .send()
        .await
        .expect("POST /api/chat streaming");

    assert_eq!(resp.status(), 200);
    let text = resp.text().await.expect("body text");
    let chunks = parse_ndjson(&text);
    assert!(!chunks.is_empty(), "stream must produce NDJSON chunks");

    let final_chunk = chunks.last().expect("terminal chunk");
    assert_eq!(final_chunk["done"], true, "final chunk must be done:true");
    assert!(
        final_chunk["eval_count"].is_number(),
        "terminal chunk must carry stats"
    );

    p.mock.verify().await;
}

// ═══════════════════════════════════════════════════════════════════════════
// upstream non-2xx -> surfaces the error, never a silent done:true
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn native_streaming_upstream_error_surfaced() {
    let p = spawn_proxy_with_native_streaming().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    // LM Studio returns 400 with a JSON error body — NOT SSE events.
    Mock::given(method("POST"))
        .and(path("/api/v1/chat"))
        .respond_with(
            ResponseTemplate::new(400)
                .set_body_json(json!({ "error": "invalid request: model not loaded" })),
        )
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "messages": [{ "role": "user", "content": "Hi" }],
            "stream": true
        }))
        .send()
        .await
        .expect("POST /api/chat streaming");

    // Must NOT be a silent 200 with done:true content.
    assert_ne!(
        resp.status().as_u16(),
        200,
        "proxy must not swallow a 400 upstream error as a silent 200"
    );

    p.mock.verify().await;
}

// ═══════════════════════════════════════════════════════════════════════════
// stream:false -> stays on the default /api/v0/chat/completions
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn non_streaming_stays_on_v0_chat_completions() {
    let p = spawn_proxy_with_native_streaming().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    // Non-streaming must NOT hit the native path — expect(0) makes a misroute fail.
    Mock::given(method("POST"))
        .and(path("/api/v1/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"_unexpected_native": true})))
        .expect(0)
        .mount(&p.mock)
        .await;

    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "OK" },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3 }
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "messages": [{ "role": "user", "content": "Hi" }],
            "stream": false
        }))
        .send()
        .await
        .expect("POST /api/chat non-streaming");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("JSON body");
    assert_eq!(body["done"], true);
    assert_eq!(body["message"]["content"], "OK");

    p.mock.verify().await;
}

// ═══════════════════════════════════════════════════════════════════════════
// error event is recoverable: error line surfaced, chat.end stats still land
// ═══════════════════════════════════════════════════════════════════════════
//
// Per api-docs streaming-events.md: "The final payload will still be sent in
// `chat.end` with whatever was generated." The proxy must keep reading.

#[tokio::test]
async fn native_stream_error_event_recovers_through_chat_end() {
    let p = spawn_proxy_with_native_streaming().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/api/v1/chat"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(
                concat!(
                    "event: chat.start\ndata: {\"type\":\"chat.start\",\"model_instance_id\":\"m\"}\n\n",
                    "event: message.delta\ndata: {\"type\":\"message.delta\",\"content\":\"partial\"}\n\n",
                    "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"mcp_connection_error\",\"message\":\"tool server dropped\"}}\n\n",
                    "event: message.delta\ndata: {\"type\":\"message.delta\",\"content\":\" tail\"}\n\n",
                    "event: chat.end\ndata: {\"type\":\"chat.end\",\"result\":{\"output\":[{\"type\":\"message\",\"content\":\"partial tail\"}],\"stats\":{\"input_tokens\":5,\"total_output_tokens\":2,\"tokens_per_second\":40.0,\"time_to_first_token_seconds\":0.1}}}\n\n"
                )
                .as_bytes(),
                "text/event-stream",
            ),
        )
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "messages": [{ "role": "user", "content": "Hi" }],
            "stream": true
        }))
        .send()
        .await
        .expect("POST /api/chat streaming");

    assert_eq!(resp.status(), 200);
    let text = resp.text().await.expect("body text");
    let chunks = parse_ndjson(&text);

    // The Ollama `{"error"}` line is a terminal sentinel — a recoverable native
    // error must never surface as one, or a compliant client aborts a request
    // that actually completed.
    assert!(
        chunks.iter().all(|c| c.get("error").is_none()),
        "no streamed chunk may carry a terminal error key: {chunks:?}"
    );
    // Content generated AFTER the error still streams as a normal done:false chunk.
    assert!(
        chunks
            .iter()
            .any(|c| c["message"]["content"] == " tail" && c["done"] == false),
        "content generated after a recoverable error must still stream"
    );

    let last = chunks.last().expect("final chunk");
    assert_eq!(
        last["done"], true,
        "chat.end must still produce a final chunk"
    );
    assert_eq!(
        last["prompt_eval_count"], 5,
        "final chunk carries real stats"
    );
    assert_eq!(last["eval_count"], 2);
    // The recovered error is folded verbatim into the final chunk's warning.
    assert!(
        last["warning"]
            .as_str()
            .is_some_and(|w| w.contains("tool server dropped")),
        "recovered error must fold into the final warning: {last}"
    );

    p.mock.verify().await;
}

// ═══════════════════════════════════════════════════════════════════════════
// error event with NO chat.end: now genuinely terminal, surfaces as an error line
// ═══════════════════════════════════════════════════════════════════════════
//
// A recoverable error is buffered, but if the stream then dies without a
// chat.end the buffered error is the real failure — it must end the stream as a
// terminal `{"error"}` line, never a fabricated done:true.

#[tokio::test]
async fn native_stream_error_without_chat_end_is_terminal() {
    let p = spawn_proxy_with_native_streaming().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/api/v1/chat"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(
                concat!(
                    "event: chat.start\ndata: {\"type\":\"chat.start\",\"model_instance_id\":\"m\"}\n\n",
                    "event: message.delta\ndata: {\"type\":\"message.delta\",\"content\":\"partial\"}\n\n",
                    "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"mcp_connection_error\",\"message\":\"tool server dropped\"}}\n\n"
                )
                .as_bytes(),
                "text/event-stream",
            ),
        )
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "messages": [{ "role": "user", "content": "Hi" }],
            "stream": true
        }))
        .send()
        .await
        .expect("POST /api/chat streaming");

    assert_eq!(resp.status(), 200);
    let text = resp.text().await.expect("body text");
    let chunks = parse_ndjson(&text);

    // The pre-error delta still streamed as a normal done:false chunk.
    assert!(
        chunks
            .iter()
            .any(|c| c["message"]["content"] == "partial" && c["done"] == false),
        "content before the terminal error must still stream: {chunks:?}"
    );
    // The stream ends on the terminal error sentinel carrying the native message.
    let last = chunks.last().expect("at least one chunk");
    assert!(
        last["error"]
            .as_str()
            .is_some_and(|e| e.contains("tool server dropped")),
        "last line must be the terminal error sentinel: {last}"
    );
    // No fabricated done:true anywhere — the request did not complete.
    assert!(
        chunks.iter().all(|c| c["done"] != true),
        "a stream that died post-error must not fabricate done:true: {chunks:?}"
    );

    p.mock.verify().await;
}

// ═══════════════════════════════════════════════════════════════════════════
// two sequential tool calls both surface (no last-wins collapse)
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn native_stream_two_tool_calls_both_surface() {
    let p = spawn_proxy_with_native_streaming().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/api/v1/chat"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(
                concat!(
                    "event: chat.start\ndata: {\"type\":\"chat.start\",\"model_instance_id\":\"m\"}\n\n",
                    "event: tool_call.start\ndata: {\"type\":\"tool_call.start\",\"tool\":\"get_weather\"}\n\n",
                    "event: tool_call.arguments\ndata: {\"type\":\"tool_call.arguments\",\"tool\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}\n\n",
                    "event: tool_call.success\ndata: {\"type\":\"tool_call.success\",\"tool\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"},\"output\":\"{}\"}\n\n",
                    "event: tool_call.start\ndata: {\"type\":\"tool_call.start\",\"tool\":\"get_time\"}\n\n",
                    "event: tool_call.arguments\ndata: {\"type\":\"tool_call.arguments\",\"tool\":\"get_time\",\"arguments\":{\"city\":\"Tokyo\"}}\n\n",
                    "event: tool_call.success\ndata: {\"type\":\"tool_call.success\",\"tool\":\"get_time\",\"arguments\":{\"city\":\"Tokyo\"},\"output\":\"{}\"}\n\n",
                    "event: chat.end\ndata: {\"type\":\"chat.end\",\"result\":{\"output\":[],\"stats\":{\"input_tokens\":5,\"total_output_tokens\":10,\"tokens_per_second\":40.0,\"time_to_first_token_seconds\":0.1}}}\n\n"
                )
                .as_bytes(),
                "text/event-stream",
            ),
        )
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "messages": [{ "role": "user", "content": "both tools please" }],
            "stream": true
        }))
        .send()
        .await
        .expect("POST /api/chat streaming");

    assert_eq!(resp.status(), 200);
    let text = resp.text().await.expect("body text");
    let chunks = parse_ndjson(&text);

    let tool_chunk = chunks
        .iter()
        .find(|c| c["message"].get("tool_calls").is_some())
        .expect("one pre-final chunk must carry the assembled tool calls");
    let calls = tool_chunk["message"]["tool_calls"]
        .as_array()
        .expect("tool_calls array");
    assert_eq!(calls.len(), 2, "both calls must survive: {calls:?}");
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    assert_eq!(
        calls[0]["function"]["arguments"],
        json!({ "city": "Paris" })
    );
    assert_eq!(calls[1]["function"]["name"], "get_time");
    assert_eq!(
        calls[1]["function"]["arguments"],
        json!({ "city": "Tokyo" })
    );

    p.mock.verify().await;
}

// ═══════════════════════════════════════════════════════════════════════════
// model_load.end fills load_duration when chat.end stats omit theirs
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn native_stream_model_load_end_fills_load_duration() {
    let p = spawn_proxy_with_native_streaming().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    // stats: ttft 0.1s, 4 tokens at 40 tps -> 0.1s generation; load 2.5s from
    // the model_load.end event only.
    Mock::given(method("POST"))
        .and(path("/api/v1/chat"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(
                concat!(
                    "event: chat.start\ndata: {\"type\":\"chat.start\",\"model_instance_id\":\"m\"}\n\n",
                    "event: model_load.end\ndata: {\"type\":\"model_load.end\",\"model_instance_id\":\"m\",\"load_time_seconds\":2.5}\n\n",
                    "event: message.delta\ndata: {\"type\":\"message.delta\",\"content\":\"hi\"}\n\n",
                    "event: chat.end\ndata: {\"type\":\"chat.end\",\"result\":{\"output\":[{\"type\":\"message\",\"content\":\"hi\"}],\"stats\":{\"input_tokens\":5,\"total_output_tokens\":4,\"tokens_per_second\":40.0,\"time_to_first_token_seconds\":0.1}}}\n\n"
                )
                .as_bytes(),
                "text/event-stream",
            ),
        )
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "messages": [{ "role": "user", "content": "Hi" }],
            "stream": true
        }))
        .send()
        .await
        .expect("POST /api/chat streaming");

    assert_eq!(resp.status(), 200);
    let text = resp.text().await.expect("body text");
    let chunks = parse_ndjson(&text);
    let last = chunks.last().expect("final chunk");

    assert_eq!(
        last["load_duration"], 2_500_000_000u64,
        "model_load.end load time must land in load_duration"
    );
    assert_eq!(
        last["total_duration"], 2_700_000_000u64,
        "total must cover load + ttft + generation"
    );

    p.mock.verify().await;
}

// ═══════════════════════════════════════════════════════════════════════════
// done_reason "length" heuristic at the cap + dropped-field warning
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn native_stream_length_done_reason_and_drop_warning() {
    let p = spawn_proxy_with_native_streaming().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/api/v1/chat"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(
                concat!(
                    "event: chat.start\ndata: {\"type\":\"chat.start\",\"model_instance_id\":\"m\"}\n\n",
                    "event: message.delta\ndata: {\"type\":\"message.delta\",\"content\":\"one two\"}\n\n",
                    "event: chat.end\ndata: {\"type\":\"chat.end\",\"result\":{\"output\":[{\"type\":\"message\",\"content\":\"one two\"}],\"stats\":{\"input_tokens\":5,\"total_output_tokens\":4,\"tokens_per_second\":40.0,\"time_to_first_token_seconds\":0.1}}}\n\n"
                )
                .as_bytes(),
                "text/event-stream",
            ),
        )
        .expect(1)
        .mount(&p.mock)
        .await;

    // tools has no native slot -> warned; output hit num_predict:4 -> length.
    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "messages": [{ "role": "user", "content": "Hi" }],
            "tools": [{ "type": "function", "function": { "name": "noop" } }],
            "options": { "num_predict": 4 },
            "stream": true
        }))
        .send()
        .await
        .expect("POST /api/chat streaming");

    assert_eq!(resp.status(), 200);
    let text = resp.text().await.expect("body text");
    let chunks = parse_ndjson(&text);
    let last = chunks.last().expect("final chunk");

    assert_eq!(
        last["done_reason"], "length",
        "output at the num_predict cap must report length (proxy heuristic)"
    );
    assert!(
        last["warning"]
            .as_str()
            .is_some_and(|w| w.contains("tools")),
        "dropped tools must be warned on the final chunk: {last}"
    );

    p.mock.verify().await;
}

// ═══════════════════════════════════════════════════════════════════════════
// MCP integration: server-executed tool call surfaces; its output stays server-side
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn native_stream_mcp_tool_call_surfaces_without_server_output() {
    let p = spawn_proxy_with_native_streaming().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    // An MCP-sourced tool call: provider_info marks it ephemeral_mcp and the
    // success event carries the server-executed `output`.
    Mock::given(method("POST"))
        .and(path("/api/v1/chat"))
        .and(body_partial_json(json!({ "integrations": ["huggingface"] })))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(
                concat!(
                    "event: chat.start\ndata: {\"type\":\"chat.start\",\"model_instance_id\":\"m\"}\n\n",
                    "event: tool_call.start\ndata: {\"type\":\"tool_call.start\",\"tool\":\"hf_search\",\"provider_info\":{\"type\":\"ephemeral_mcp\",\"server_label\":\"huggingface\"}}\n\n",
                    "event: tool_call.arguments\ndata: {\"type\":\"tool_call.arguments\",\"tool\":\"hf_search\",\"arguments\":{\"query\":\"rust\"},\"provider_info\":{\"type\":\"ephemeral_mcp\",\"server_label\":\"huggingface\"}}\n\n",
                    "event: tool_call.success\ndata: {\"type\":\"tool_call.success\",\"tool\":\"hf_search\",\"arguments\":{\"query\":\"rust\"},\"provider_info\":{\"type\":\"ephemeral_mcp\",\"server_label\":\"huggingface\"},\"output\":\"SERVER_ONLY_OUTPUT_marker\"}\n\n",
                    "event: chat.end\ndata: {\"type\":\"chat.end\",\"result\":{\"output\":[],\"stats\":{\"input_tokens\":5,\"total_output_tokens\":3,\"tokens_per_second\":40.0,\"time_to_first_token_seconds\":0.1}}}\n\n"
                )
                .as_bytes(),
                "text/event-stream",
            ),
        )
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "messages": [{ "role": "user", "content": "search please" }],
            "integrations": ["huggingface"],
            "stream": true
        }))
        .send()
        .await
        .expect("POST /api/chat streaming");

    assert_eq!(resp.status(), 200);
    let text = resp.text().await.expect("body text");
    let chunks = parse_ndjson(&text);

    // The assembled tool call surfaces exactly once, in a pre-final done:false chunk.
    let tool_chunks: Vec<&Value> = chunks
        .iter()
        .filter(|c| c["message"].get("tool_calls").is_some())
        .collect();
    assert_eq!(
        tool_chunks.len(),
        1,
        "assembled tool call must surface exactly once: {chunks:?}"
    );
    let tool_chunk = tool_chunks[0];
    assert_eq!(tool_chunk["done"], false, "tool-call chunk is pre-final");
    let calls = tool_chunk["message"]["tool_calls"]
        .as_array()
        .expect("tool_calls array");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["function"]["name"], "hf_search");
    assert_eq!(
        calls[0]["function"]["arguments"],
        json!({ "query": "rust" })
    );

    // The server-executed `output` is intentionally NOT injected as message
    // content — Ollama has no slot for server-run tool output, so it stays
    // server-side. This is by design, not a lost field.
    assert!(
        chunks.iter().all(|c| c["message"]["content"]
            .as_str()
            .is_none_or(|s| !s.contains("SERVER_ONLY_OUTPUT_marker"))),
        "server-executed tool output must not leak into client content: {chunks:?}"
    );

    p.mock.verify().await;
}

// ═══════════════════════════════════════════════════════════════════════════
// MCP integration: recover from mcp_connection_error, warning folds the message
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn native_stream_mcp_connection_error_recovers() {
    let p = spawn_proxy_with_native_streaming().await;
    mount_model_catalog(&p, "llama3.1-8b-instruct").await;

    Mock::given(method("POST"))
        .and(path("/api/v1/chat"))
        .and(body_partial_json(json!({ "integrations": ["huggingface"] })))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(
                concat!(
                    "event: chat.start\ndata: {\"type\":\"chat.start\",\"model_instance_id\":\"m\"}\n\n",
                    "event: tool_call.start\ndata: {\"type\":\"tool_call.start\",\"tool\":\"hf_search\"}\n\n",
                    "event: tool_call.arguments\ndata: {\"type\":\"tool_call.arguments\",\"tool\":\"hf_search\",\"arguments\":{\"query\":\"rust\"}}\n\n",
                    "event: tool_call.success\ndata: {\"type\":\"tool_call.success\",\"tool\":\"hf_search\",\"arguments\":{\"query\":\"rust\"},\"output\":\"{}\"}\n\n",
                    "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"mcp_connection_error\",\"message\":\"mcp server unreachable\"}}\n\n",
                    "event: message.delta\ndata: {\"type\":\"message.delta\",\"content\":\"after error\"}\n\n",
                    "event: chat.end\ndata: {\"type\":\"chat.end\",\"result\":{\"output\":[],\"stats\":{\"input_tokens\":5,\"total_output_tokens\":4,\"tokens_per_second\":40.0,\"time_to_first_token_seconds\":0.1}}}\n\n"
                )
                .as_bytes(),
                "text/event-stream",
            ),
        )
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3.1:8b",
            "messages": [{ "role": "user", "content": "search please" }],
            "integrations": ["huggingface"],
            "stream": true
        }))
        .send()
        .await
        .expect("POST /api/chat streaming");

    assert_eq!(resp.status(), 200);
    let text = resp.text().await.expect("body text");
    let chunks = parse_ndjson(&text);

    // No inline terminal error chunk.
    assert!(
        chunks.iter().all(|c| c.get("error").is_none()),
        "recoverable mcp error must not surface as a terminal line: {chunks:?}"
    );

    let last = chunks.last().expect("final chunk");
    assert_eq!(last["done"], true, "chat.end still produces a final chunk");
    // The mcp error message is folded into the final chunk's warning.
    assert!(
        last["warning"]
            .as_str()
            .is_some_and(|w| w.contains("mcp server unreachable")),
        "recovered mcp error must fold into the final warning: {last}"
    );

    p.mock.verify().await;
}
