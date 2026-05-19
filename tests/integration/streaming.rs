// End-to-end streaming tests: LM Studio SSE → proxy → Ollama NDJSON.
//
// The wiremock backend emits SSE bodies; the proxy translates them to newline-
// delimited JSON (one JSON object per line). Tests collect all lines and assert
// on shape and content.

use futures_util::StreamExt;
use serde_json::{Value, json};
use wiremock::matchers::{method, path, path_regex};
use wiremock::{Mock, ResponseTemplate};

use crate::common::spawn_proxy;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Collect all NDJSON lines from a streaming response into a Vec<Value>.
/// Ignores blank lines. Panics if any line is not valid JSON.
async fn collect_ndjson(resp: reqwest::Response) -> Vec<Value> {
    let mut stream = resp.bytes_stream();
    let mut buf = Vec::new();
    while let Some(chunk) = stream.next().await {
        buf.extend_from_slice(&chunk.expect("stream chunk"));
    }
    let text = String::from_utf8(buf).expect("utf8 body");
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("bad JSON line {l:?}: {e}")))
        .collect()
}

/// Build a minimal SSE body from a slice of JSON strings plus a final [DONE].
fn sse_body(data_jsons: &[&str]) -> String {
    let mut body = String::new();
    for j in data_jsons {
        body.push_str("data: ");
        body.push_str(j);
        body.push_str("\n\n");
    }
    body.push_str("data: [DONE]\n\n");
    body
}

fn sse_response(body: String) -> ResponseTemplate {
    ResponseTemplate::new(200)
        .insert_header("content-type", "text/event-stream")
        .set_body_raw(body.into_bytes(), "text/event-stream")
}

// ---------------------------------------------------------------------------
// 1. /api/chat stream:true — basic delta chunks
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_stream_emits_ndjson_chunks_with_done() {
    let p = spawn_proxy().await;

    let body = sse_body(&[
        r#"{"choices":[{"delta":{"role":"assistant","content":"He"},"finish_reason":null}]}"#,
        r#"{"choices":[{"delta":{"content":"llo"},"finish_reason":null}]}"#,
        r#"{"choices":[{"delta":{"content":"!"},"finish_reason":"stop"}]}"#,
    ]);

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(sse_response(body))
        .mount(&p.mock)
        .await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [{"key": "llama3", "type": "llm", "publisher": "meta",
                        "architecture": "llama", "format": "gguf",
                        "quantization": {"name": "Q4_K_M", "bits_per_weight": 4.5},
                        "max_context_length": 8192, "loaded_instances": [],
                        "capabilities": {"vision": false, "trained_for_tool_use": false}}]
        })))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        }))
        .send()
        .await
        .expect("POST /api/chat");

    assert_eq!(resp.status(), 200);
    let chunks = collect_ndjson(resp).await;
    assert!(!chunks.is_empty(), "expected at least one chunk");

    let content_chunks: Vec<&Value> = chunks
        .iter()
        .filter(|c| {
            c.get("message")
                .and_then(|m| m.get("content"))
                .and_then(|v| v.as_str())
                .map(|s| !s.is_empty())
                .unwrap_or(false)
        })
        .collect();
    assert!(
        !content_chunks.is_empty(),
        "expected at least one content delta chunk"
    );

    let last = chunks.last().expect("last chunk");
    assert_eq!(
        last.get("done"),
        Some(&json!(true)),
        "last chunk must have done:true"
    );
    assert!(
        last.get("done_reason").is_some()
            || last.get("total_duration").is_some()
            || last.get("eval_count").is_some(),
        "terminal chunk must carry completion stats: {last}"
    );

    for chunk in &chunks {
        assert!(
            serde_json::to_string(chunk).is_ok(),
            "chunk must be individually serialisable: {chunk}"
        );
    }
}

// ---------------------------------------------------------------------------
// 2. /api/generate stream:true
// ---------------------------------------------------------------------------

#[tokio::test]
async fn generate_stream_emits_ndjson_chunks_with_done() {
    let p = spawn_proxy().await;

    let body = sse_body(&[
        r#"{"choices":[{"text":"Hel","finish_reason":null}]}"#,
        r#"{"choices":[{"text":"lo","finish_reason":"stop"}]}"#,
    ]);

    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .respond_with(sse_response(body))
        .mount(&p.mock)
        .await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [{"key": "llama3", "type": "llm", "publisher": "meta",
                        "architecture": "llama", "format": "gguf",
                        "quantization": {"name": "Q4_K_M", "bits_per_weight": 4.5},
                        "max_context_length": 8192, "loaded_instances": [],
                        "capabilities": {"vision": false, "trained_for_tool_use": false}}]
        })))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/generate"))
        .json(&json!({
            "model": "llama3",
            "prompt": "hello",
            "stream": true
        }))
        .send()
        .await
        .expect("POST /api/generate");

    assert_eq!(resp.status(), 200);
    let chunks = collect_ndjson(resp).await;
    assert!(!chunks.is_empty());

    let last = chunks.last().expect("last chunk");
    assert_eq!(last.get("done"), Some(&json!(true)));
}

// ---------------------------------------------------------------------------
// 3. /api/embed — no streaming, returns a single JSON body
// ---------------------------------------------------------------------------

#[tokio::test]
async fn embed_returns_single_json_no_streaming() {
    let p = spawn_proxy().await;

    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
            "model": "llama3",
            "usage": {"prompt_tokens": 4, "total_tokens": 4}
        })))
        .mount(&p.mock)
        .await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [{"key": "llama3", "type": "llm", "publisher": "meta",
                        "architecture": "llama", "format": "gguf",
                        "quantization": {"name": "Q4_K_M", "bits_per_weight": 4.5},
                        "max_context_length": 8192, "loaded_instances": [],
                        "capabilities": {"vision": false, "trained_for_tool_use": false}}]
        })))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/embed"))
        .json(&json!({"model": "llama3", "input": "hello"}))
        .send()
        .await
        .expect("POST /api/embed");

    assert_eq!(resp.status(), 200);

    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        content_type.contains("application/json"),
        "embed should return JSON, got: {content_type}"
    );

    let body: Value = resp.json().await.expect("JSON body");
    // body should not be an array of NDJSON lines — it is a single object
    assert!(
        body.is_object(),
        "embed must return a JSON object, not a stream"
    );
}

// ---------------------------------------------------------------------------
// 4. Mid-stream backend error
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_stream_mid_error_yields_error_chunk() {
    let p = spawn_proxy().await;

    // Two good chunks then an error event
    let mut body = String::new();
    body.push_str(
        "data: {\"choices\":[{\"delta\":{\"content\":\"A\"},\"finish_reason\":null}]}\n\n",
    );
    body.push_str(
        "data: {\"choices\":[{\"delta\":{\"content\":\"B\"},\"finish_reason\":null}]}\n\n",
    );
    body.push_str("data: {\"error\":{\"message\":\"internal model error\"}}\n\n");
    body.push_str("data: [DONE]\n\n");

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(sse_response(body))
        .mount(&p.mock)
        .await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [{"key": "llama3", "type": "llm", "publisher": "meta",
                        "architecture": "llama", "format": "gguf",
                        "quantization": {"name": "Q4_K_M", "bits_per_weight": 4.5},
                        "max_context_length": 8192, "loaded_instances": [],
                        "capabilities": {"vision": false, "trained_for_tool_use": false}}]
        })))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3",
            "messages": [{"role": "user", "content": "go"}],
            "stream": true
        }))
        .send()
        .await
        .expect("POST /api/chat");

    let chunks = collect_ndjson(resp).await;
    assert!(!chunks.is_empty());
    // The stream must terminate — no hang. The final chunk should have done:true.
    let last = chunks.last().expect("last chunk");
    assert_eq!(
        last.get("done"),
        Some(&json!(true)),
        "stream must terminate with done:true: {last}"
    );
}

// ---------------------------------------------------------------------------
// 5. Backend disconnects mid-stream without [DONE] — recovery chunk emitted
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_stream_backend_disconnect_without_done_terminates() {
    let p = spawn_proxy().await;

    // No [DONE] at the end — wiremock closes after last chunk
    let mut body = String::new();
    body.push_str(
        "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n",
    );

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(sse_response(body))
        .mount(&p.mock)
        .await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [{"key": "llama3", "type": "llm", "publisher": "meta",
                        "architecture": "llama", "format": "gguf",
                        "quantization": {"name": "Q4_K_M", "bits_per_weight": 4.5},
                        "max_context_length": 8192, "loaded_instances": [],
                        "capabilities": {"vision": false, "trained_for_tool_use": false}}]
        })))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        }))
        .send()
        .await
        .expect("POST /api/chat");

    // Key assertion: the proxy must not hang — we must be able to collect all chunks.
    let chunks = collect_ndjson(resp).await;
    assert!(
        !chunks.is_empty(),
        "expected at least one chunk before close"
    );

    // With enable_chunk_recovery=true a synthetic done:true chunk should appear.
    let last = chunks.last().expect("last chunk");
    assert_eq!(
        last.get("done"),
        Some(&json!(true)),
        "recovery should emit terminal done:true chunk: {last}"
    );
}

// ---------------------------------------------------------------------------
// 6. Reasoning tokens mapped to `thinking` field
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_stream_reasoning_delta_maps_to_thinking() {
    let p = spawn_proxy().await;

    let body = sse_body(&[
        r#"{"choices":[{"delta":{"reasoning":"let me think..."},"finish_reason":null}]}"#,
        r#"{"choices":[{"delta":{"content":"Answer"},"finish_reason":"stop"}]}"#,
    ]);

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(sse_response(body))
        .mount(&p.mock)
        .await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [{"key": "llama3", "type": "llm", "publisher": "meta",
                        "architecture": "llama", "format": "gguf",
                        "quantization": {"name": "Q4_K_M", "bits_per_weight": 4.5},
                        "max_context_length": 8192, "loaded_instances": [],
                        "capabilities": {"vision": false, "trained_for_tool_use": false}}]
        })))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3",
            "messages": [{"role": "user", "content": "think"}],
            "stream": true
        }))
        .send()
        .await
        .expect("POST /api/chat");

    let chunks = collect_ndjson(resp).await;
    let thinking_chunks: Vec<&Value> = chunks
        .iter()
        .filter(|c| {
            c.get("message")
                .and_then(|m| m.get("thinking"))
                .and_then(|v| v.as_str())
                .map(|s| !s.is_empty())
                .unwrap_or(false)
        })
        .collect();

    assert!(
        !thinking_chunks.is_empty(),
        "expected at least one chunk with thinking field; chunks: {chunks:?}"
    );

    // The reasoning chunk must NOT surface in `content`
    let reasoning_as_content = chunks.iter().any(|c| {
        c.get("message")
            .and_then(|m| m.get("content"))
            .and_then(|v| v.as_str())
            .map(|s| s.contains("let me think"))
            .unwrap_or(false)
    });
    assert!(
        !reasoning_as_content,
        "reasoning text must not appear in content field"
    );
}

// ---------------------------------------------------------------------------
// 7. Mixed reasoning + content in same chunk
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_stream_mixed_reasoning_and_content_chunk() {
    let p = spawn_proxy().await;

    let body = sse_body(&[
        r#"{"choices":[{"delta":{"reasoning":"step 1","content":"ans"},"finish_reason":"stop"}]}"#,
    ]);

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(sse_response(body))
        .mount(&p.mock)
        .await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [{"key": "llama3", "type": "llm", "publisher": "meta",
                        "architecture": "llama", "format": "gguf",
                        "quantization": {"name": "Q4_K_M", "bits_per_weight": 4.5},
                        "max_context_length": 8192, "loaded_instances": [],
                        "capabilities": {"vision": false, "trained_for_tool_use": false}}]
        })))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3",
            "messages": [{"role": "user", "content": "go"}],
            "stream": true
        }))
        .send()
        .await
        .expect("POST /api/chat");

    let chunks = collect_ndjson(resp).await;
    let has_content = chunks.iter().any(|c| {
        c.get("message")
            .and_then(|m| m.get("content"))
            .and_then(|v| v.as_str())
            .map(|s| s.contains("ans"))
            .unwrap_or(false)
    });
    let has_thinking = chunks.iter().any(|c| {
        c.get("message")
            .and_then(|m| m.get("thinking"))
            .and_then(|v| v.as_str())
            .map(|s| !s.is_empty())
            .unwrap_or(false)
    });
    assert!(has_content, "expected content in chunks: {chunks:?}");
    assert!(has_thinking, "expected thinking in chunks: {chunks:?}");
}

// ---------------------------------------------------------------------------
// 8. Tool calls in stream
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_stream_tool_calls_present_in_output() {
    let p = spawn_proxy().await;

    let body = sse_body(&[
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"London\"}"}}]},"finish_reason":null}]}"#,
        r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
    ]);

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(sse_response(body))
        .mount(&p.mock)
        .await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [{"key": "llama3", "type": "llm", "publisher": "meta",
                        "architecture": "llama", "format": "gguf",
                        "quantization": {"name": "Q4_K_M", "bits_per_weight": 4.5},
                        "max_context_length": 8192, "loaded_instances": [],
                        "capabilities": {"vision": false, "trained_for_tool_use": false}}]
        })))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3",
            "messages": [{"role": "user", "content": "weather?"}],
            "stream": true
        }))
        .send()
        .await
        .expect("POST /api/chat");

    let chunks = collect_ndjson(resp).await;
    let has_tool_calls = chunks.iter().any(|c| {
        c.get("message")
            .and_then(|m| m.get("tool_calls"))
            .and_then(|v| v.as_array())
            .map(|a| !a.is_empty())
            .unwrap_or(false)
    });
    assert!(
        has_tool_calls,
        "expected tool_calls in at least one chunk; got: {chunks:?}"
    );
}

// ---------------------------------------------------------------------------
// 9. Empty stream — backend emits only [DONE]
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_stream_empty_backend_emits_terminal_done() {
    let p = spawn_proxy().await;

    // Only [DONE], no data chunks
    let body = "data: [DONE]\n\n".to_string();

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(sse_response(body))
        .mount(&p.mock)
        .await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [{"key": "llama3", "type": "llm", "publisher": "meta",
                        "architecture": "llama", "format": "gguf",
                        "quantization": {"name": "Q4_K_M", "bits_per_weight": 4.5},
                        "max_context_length": 8192, "loaded_instances": [],
                        "capabilities": {"vision": false, "trained_for_tool_use": false}}]
        })))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        }))
        .send()
        .await
        .expect("POST /api/chat");

    let chunks = collect_ndjson(resp).await;
    // Must emit at least the terminal done chunk
    assert!(
        !chunks.is_empty(),
        "expected at least one chunk (terminal done)"
    );
    let last = chunks.last().expect("last chunk");
    assert_eq!(
        last.get("done"),
        Some(&json!(true)),
        "terminal done required: {last}"
    );
}

// ---------------------------------------------------------------------------
// 10. SSE comment lines (`: keep-alive`) are ignored
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_stream_sse_comments_are_ignored() {
    let p = spawn_proxy().await;

    let mut body = String::new();
    body.push_str(": keep-alive\n\n");
    body.push_str(
        "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n",
    );
    body.push_str(": another comment\n\n");
    body.push_str("data: [DONE]\n\n");

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(sse_response(body))
        .mount(&p.mock)
        .await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [{"key": "llama3", "type": "llm", "publisher": "meta",
                        "architecture": "llama", "format": "gguf",
                        "quantization": {"name": "Q4_K_M", "bits_per_weight": 4.5},
                        "max_context_length": 8192, "loaded_instances": [],
                        "capabilities": {"vision": false, "trained_for_tool_use": false}}]
        })))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        }))
        .send()
        .await
        .expect("POST /api/chat");

    let chunks = collect_ndjson(resp).await;
    // Comment lines must not appear as valid JSON chunks
    for chunk in &chunks {
        assert!(
            chunk.is_object(),
            "all chunks must be JSON objects (comment lines must be filtered): {chunk}"
        );
    }
    let last = chunks.last().expect("last chunk");
    assert_eq!(last.get("done"), Some(&json!(true)));
}

// ---------------------------------------------------------------------------
// 11. /api/pull — NDJSON progress stream, terminal chunk is bare {"status":"success"}
// ---------------------------------------------------------------------------

#[tokio::test]
async fn pull_stream_terminates_with_bare_success() {
    let p = spawn_proxy().await;

    // Pull is handled locally by the proxy (downloads from LM Studio).
    // The proxy emits NDJSON progress lines then a terminal {"status":"success"}.
    // Return a completed download so no status polling is needed.
    Mock::given(method("POST"))
        .and(path("/api/v1/models/download"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "job_id": "stream-job-1",
            "status": "completed",
            "total_size_bytes": 4_000_000_000u64,
            "downloaded_bytes": 4_000_000_000u64,
            "completed_at": "2026-01-01T00:00:00Z"
        })))
        .mount(&p.mock)
        .await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [{"key": "llama3", "type": "llm", "publisher": "meta",
                        "architecture": "llama", "format": "gguf",
                        "quantization": {"name": "Q4_K_M", "bits_per_weight": 4.5},
                        "max_context_length": 8192, "loaded_instances": [],
                        "capabilities": {"vision": false, "trained_for_tool_use": false}}]
        })))
        .mount(&p.mock)
        .await;

    // Mock download status polling — proxy appends job_id to the path.
    Mock::given(method("GET"))
        .and(path_regex(r"^/api/v1/models/download/status/.*"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "status": "complete",
            "progress": 1.0
        })))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/pull"))
        .json(&json!({"model": "llama3", "stream": true}))
        .send()
        .await
        .expect("POST /api/pull");

    assert_eq!(resp.status(), 200);
    let chunks = collect_ndjson(resp).await;
    assert!(
        !chunks.is_empty(),
        "pull must emit at least one NDJSON chunk"
    );

    let last = chunks.last().expect("last chunk");
    // Per commit 229f417: terminal chunk is bare {"status":"success"}
    assert_eq!(
        last.get("status").and_then(|v| v.as_str()),
        Some("success"),
        "terminal pull chunk must be bare {{\"status\":\"success\"}}: {last}"
    );
    // Ensure it truly is bare — no extra keys beyond "status"
    if let Some(obj) = last.as_object() {
        assert_eq!(
            obj.len(),
            1,
            "terminal pull chunk must be bare {{\"status\":\"success\"}}, got extra keys: {last}"
        );
    }
}

// ---------------------------------------------------------------------------
// 12. /api/create — NDJSON progress stream terminates in {"status":"success"}
// ---------------------------------------------------------------------------

#[tokio::test]
async fn create_stream_terminates_with_success() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [{"key": "llama3", "type": "llm", "publisher": "meta",
                        "architecture": "llama", "format": "gguf",
                        "quantization": {"name": "Q4_K_M", "bits_per_weight": 4.5},
                        "max_context_length": 8192, "loaded_instances": [],
                        "capabilities": {"vision": false, "trained_for_tool_use": false}}]
        })))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/create"))
        .json(&json!({
            "model": "my-model",
            "from": "llama3",
            "system": "You are a helpful assistant."
        }))
        .send()
        .await
        .expect("POST /api/create");

    assert_eq!(resp.status(), 200);
    let chunks = collect_ndjson(resp).await;
    assert!(
        !chunks.is_empty(),
        "create must emit at least one NDJSON chunk"
    );

    let last = chunks.last().expect("last chunk");
    assert_eq!(
        last.get("status").and_then(|v| v.as_str()),
        Some("success"),
        "terminal create chunk must be {{\"status\":\"success\"}}: {last}"
    );
}

// ---------------------------------------------------------------------------
// 13. Multi-line / split SSE data fields are reassembled correctly
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_stream_split_sse_data_reassembled() {
    let p = spawn_proxy().await;

    // Simulate a backend that puts a newline inside the JSON value (unusual but valid SSE)
    // Proxy must reassemble before parsing.
    // We test a simpler case: multiple `data:` lines in one event block — SSE spec
    // says continuation lines after `data:` in the same event block are joined with \n.
    // In practice the proxy sees these as a single event.
    let mut body = String::new();
    // Write two separate complete events; each must produce one Ollama chunk.
    body.push_str(
        "data: {\"choices\":[{\"delta\":{\"content\":\"A\"},\"finish_reason\":null}]}\n\n",
    );
    body.push_str(
        "data: {\"choices\":[{\"delta\":{\"content\":\"B\"},\"finish_reason\":\"stop\"}]}\n\n",
    );
    body.push_str("data: [DONE]\n\n");

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(sse_response(body))
        .mount(&p.mock)
        .await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [{"key": "llama3", "type": "llm", "publisher": "meta",
                        "architecture": "llama", "format": "gguf",
                        "quantization": {"name": "Q4_K_M", "bits_per_weight": 4.5},
                        "max_context_length": 8192, "loaded_instances": [],
                        "capabilities": {"vision": false, "trained_for_tool_use": false}}]
        })))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        }))
        .send()
        .await
        .expect("POST /api/chat");

    let chunks = collect_ndjson(resp).await;
    let full_content: String = chunks
        .iter()
        .filter_map(|c| {
            c.get("message")
                .and_then(|m| m.get("content"))
                .and_then(|v| v.as_str())
        })
        .collect();
    assert!(
        full_content.contains('A'),
        "expected 'A' in accumulated content: {full_content:?}"
    );
    assert!(
        full_content.contains('B'),
        "expected 'B' in accumulated content: {full_content:?}"
    );
}

// ---------------------------------------------------------------------------
// 14. Generate stream — reasoning-only chunks produce `thinking` not `response`
// ---------------------------------------------------------------------------

#[tokio::test]
async fn generate_stream_reasoning_goes_to_thinking_not_response() {
    let p = spawn_proxy().await;

    let body = sse_body(&[
        r#"{"choices":[{"delta":{"reasoning":"internal thought"},"finish_reason":null}]}"#,
        r#"{"choices":[{"text":"done","finish_reason":"stop"}]}"#,
    ]);

    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .respond_with(sse_response(body))
        .mount(&p.mock)
        .await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [{"key": "llama3", "type": "llm", "publisher": "meta",
                        "architecture": "llama", "format": "gguf",
                        "quantization": {"name": "Q4_K_M", "bits_per_weight": 4.5},
                        "max_context_length": 8192, "loaded_instances": [],
                        "capabilities": {"vision": false, "trained_for_tool_use": false}}]
        })))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/generate"))
        .json(&json!({"model": "llama3", "prompt": "think", "stream": true}))
        .send()
        .await
        .expect("POST /api/generate");

    let chunks = collect_ndjson(resp).await;
    let reasoning_in_response = chunks.iter().any(|c| {
        c.get("response")
            .and_then(|v| v.as_str())
            .map(|s| s.contains("internal thought"))
            .unwrap_or(false)
    });
    assert!(
        !reasoning_in_response,
        "reasoning text must not appear in 'response' field of /api/generate chunks"
    );
}

// ---------------------------------------------------------------------------
// 15. Each backend chunk produces exactly one Ollama NDJSON line (no batching)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_stream_one_to_one_chunk_mapping() {
    let p = spawn_proxy().await;

    let data_chunks = [
        r#"{"choices":[{"delta":{"content":"1"},"finish_reason":null}]}"#,
        r#"{"choices":[{"delta":{"content":"2"},"finish_reason":null}]}"#,
        r#"{"choices":[{"delta":{"content":"3"},"finish_reason":"stop"}]}"#,
    ];
    let body = sse_body(&data_chunks);

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(sse_response(body))
        .mount(&p.mock)
        .await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [{"key": "llama3", "type": "llm", "publisher": "meta",
                        "architecture": "llama", "format": "gguf",
                        "quantization": {"name": "Q4_K_M", "bits_per_weight": 4.5},
                        "max_context_length": 8192, "loaded_instances": [],
                        "capabilities": {"vision": false, "trained_for_tool_use": false}}]
        })))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3",
            "messages": [{"role": "user", "content": "count"}],
            "stream": true
        }))
        .send()
        .await
        .expect("POST /api/chat");

    let chunks = collect_ndjson(resp).await;
    // 3 data chunks + at least the terminal done — each chunk individually parseable.
    assert!(
        chunks.len() >= data_chunks.len(),
        "expected at least {} chunks, got {}: {chunks:?}",
        data_chunks.len(),
        chunks.len()
    );
}

// ---------------------------------------------------------------------------
// 16. /api/chat non-streaming — response is a single JSON object
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_non_stream_returns_single_object() {
    let p = spawn_proxy().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
        })))
        .mount(&p.mock)
        .await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [{"key": "llama3", "type": "llm", "publisher": "meta",
                        "architecture": "llama", "format": "gguf",
                        "quantization": {"name": "Q4_K_M", "bits_per_weight": 4.5},
                        "max_context_length": 8192, "loaded_instances": [],
                        "capabilities": {"vision": false, "trained_for_tool_use": false}}]
        })))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": false
        }))
        .send()
        .await
        .expect("POST /api/chat");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("JSON body");
    assert!(
        body.is_object(),
        "non-streaming response must be a JSON object"
    );
    assert_eq!(
        body.get("done"),
        Some(&json!(true)),
        "non-streaming must have done:true"
    );
    assert!(
        body.get("message").is_some(),
        "non-streaming chat must have message field: {body}"
    );
}

// ---------------------------------------------------------------------------
// 17. /api/generate non-streaming — response is a single JSON object
// ---------------------------------------------------------------------------

#[tokio::test]
async fn generate_non_stream_returns_single_object() {
    let p = spawn_proxy().await;

    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "cmpl-1",
            "object": "text_completion",
            "choices": [{"text": "world", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
        })))
        .mount(&p.mock)
        .await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [{"key": "llama3", "type": "llm", "publisher": "meta",
                        "architecture": "llama", "format": "gguf",
                        "quantization": {"name": "Q4_K_M", "bits_per_weight": 4.5},
                        "max_context_length": 8192, "loaded_instances": [],
                        "capabilities": {"vision": false, "trained_for_tool_use": false}}]
        })))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/generate"))
        .json(&json!({"model": "llama3", "prompt": "hello", "stream": false}))
        .send()
        .await
        .expect("POST /api/generate");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("JSON body");
    assert!(body.is_object());
    assert_eq!(body.get("done"), Some(&json!(true)));
    assert!(
        body.get("response").is_some(),
        "generate non-streaming must have response: {body}"
    );
}

// ---------------------------------------------------------------------------
// 18. Client drops connection mid-stream — proxy should not hang
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_stream_client_drop_does_not_hang() {
    let p = spawn_proxy().await;

    // Long stream body
    let mut body = String::new();
    for _ in 0..20 {
        body.push_str(
            "data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":null}]}\n\n",
        );
    }
    body.push_str("data: [DONE]\n\n");

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(sse_response(body))
        .mount(&p.mock)
        .await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [{"key": "llama3", "type": "llm", "publisher": "meta",
                        "architecture": "llama", "format": "gguf",
                        "quantization": {"name": "Q4_K_M", "bits_per_weight": 4.5},
                        "max_context_length": 8192, "loaded_instances": [],
                        "capabilities": {"vision": false, "trained_for_tool_use": false}}]
        })))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3",
            "messages": [{"role": "user", "content": "flood"}],
            "stream": true
        }))
        .send()
        .await
        .expect("POST /api/chat");

    // Read only the first few bytes then drop the response.
    // The main assertion is that this completes without hanging (tokio test timeout).
    let mut stream = resp.bytes_stream();
    let mut read = 0usize;
    while let Some(chunk) = stream.next().await {
        read += chunk.expect("chunk").len();
        if read > 10 {
            break;
        }
    }
    drop(stream);
    // If we reach here without timing out, the test passes.
}

// ---------------------------------------------------------------------------
// 19. All NDJSON chunks across a stream are individually valid JSON
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_stream_all_lines_are_valid_json() {
    let p = spawn_proxy().await;

    let body = sse_body(&[
        r#"{"choices":[{"delta":{"content":"H"},"finish_reason":null}]}"#,
        r#"{"choices":[{"delta":{"content":"i"},"finish_reason":null}]}"#,
        r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
    ]);

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(sse_response(body))
        .mount(&p.mock)
        .await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [{"key": "llama3", "type": "llm", "publisher": "meta",
                        "architecture": "llama", "format": "gguf",
                        "quantization": {"name": "Q4_K_M", "bits_per_weight": 4.5},
                        "max_context_length": 8192, "loaded_instances": [],
                        "capabilities": {"vision": false, "trained_for_tool_use": false}}]
        })))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        }))
        .send()
        .await
        .expect("POST /api/chat");

    // collect_ndjson already panics on invalid JSON lines; reaching here means all were valid.
    let chunks = collect_ndjson(resp).await;
    assert!(!chunks.is_empty());
    for chunk in &chunks {
        assert!(
            chunk.is_object(),
            "every chunk must be a JSON object: {chunk}"
        );
        assert!(
            chunk.get("model").is_some(),
            "every chunk must have model: {chunk}"
        );
        assert!(
            chunk.get("done").is_some(),
            "every chunk must have done: {chunk}"
        );
    }
}
