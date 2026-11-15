// keep_alive:0 teardown must follow the response stream's own end (completion
// or client disconnect), never a guessed delay. These tests pace a real SSE
// backend over raw TCP because wiremock buffers whole bodies and cannot hold a
// stream open mid-body.
//
// The backend records whether *any* unload request arrived before it finished
// writing the SSE body, so a premature teardown is caught on every attempt at
// the source, not by sampling timestamps from the client side.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use crate::common::spawn_proxy_against_backend;

/// Pacing for the raw chat-completions SSE response.
#[derive(Clone, Copy)]
struct SsePlan {
    /// Number of content chunks before the `finish_reason` event and `[DONE]`.
    chunks: usize,
    /// Gap between consecutive SSE events.
    gap: Duration,
    /// Delay between the response headers and the first SSE event.
    initial_delay: Duration,
}

struct RawBackend {
    base_url: String,
    state: Arc<BackendState>,
}

async fn spawn_raw_backend(plan: SsePlan) -> RawBackend {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind raw backend");
    let addr = listener.local_addr().expect("raw backend addr");
    let state = Arc::new(BackendState::default());
    let task_state = Arc::clone(&state);
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                break;
            };
            let state = Arc::clone(&task_state);
            tokio::spawn(async move {
                if let Err(e) = serve_connection(&mut socket, plan, state).await {
                    log::debug!("backend connection error: {e}");
                }
            });
        }
    });
    RawBackend {
        base_url: format!("http://{addr}"),
        state,
    }
}

#[derive(Default)]
struct BackendState {
    unload_seen: AtomicBool,
    premature_unload: AtomicBool,
    stream_finished: AtomicBool,
}

/// Minimal HTTP/1.1 responder: one request per connection, close-delimited
/// bodies so SSE events dribble out at the plan's pace.
async fn serve_connection(
    socket: &mut TcpStream,
    plan: SsePlan,
    state: Arc<BackendState>,
) -> std::io::Result<()> {
    let mut buf = vec![0u8; 8192];
    let read = socket.read(&mut buf).await?;
    if read == 0 {
        return Ok(());
    }
    let request = String::from_utf8_lossy(&buf[..read]);
    let request_line = request.lines().next().unwrap_or_default();
    if request_line.starts_with("GET /api/v1/models ") {
        let body = json!({ "models": [loaded_model_entry()] }).to_string();
        write_json(socket, &body).await
    } else if request_line.starts_with("POST /api/v0/chat/completions ") {
        write_sse(socket, plan, &state).await
    } else if request_line.starts_with("POST /api/v1/models/unload ") {
        if !state.stream_finished.load(Ordering::SeqCst) {
            state.premature_unload.store(true, Ordering::SeqCst);
        }
        state.unload_seen.store(true, Ordering::SeqCst);
        write_json(socket, "{}").await
    } else {
        socket
            .write_all(b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n")
            .await?;
        socket.shutdown().await
    }
}

fn loaded_model_entry() -> serde_json::Value {
    json!({
        "key": "test-model",
        "type": "llm",
        "publisher": "test",
        "architecture": "llama",
        "format": "gguf",
        "quantization": { "name": "Q4_K_M", "bits_per_weight": 4.5 },
        "max_context_length": 8192,
        "loaded_instances": [{ "id": "inst-0", "config": { "context_length": 4096 } }],
        "capabilities": { "vision": false, "trained_for_tool_use": false }
    })
}

async fn write_json(socket: &mut TcpStream, body: &str) -> std::io::Result<()> {
    let head = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
        body.len()
    );
    socket.write_all(head.as_bytes()).await?;
    socket.write_all(body.as_bytes()).await?;
    socket.shutdown().await
}

/// Close-delimited body: no content-length, so the client sees each event as
/// it lands and end-of-body at close.
async fn write_sse(
    socket: &mut TcpStream,
    plan: SsePlan,
    state: &BackendState,
) -> std::io::Result<()> {
    socket
        .write_all(
            b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n",
        )
        .await?;
    tokio::time::sleep(plan.initial_delay).await;
    let mut events = Vec::with_capacity(plan.chunks + 2);
    events.push(
        r#"{"choices":[{"delta":{"role":"assistant","content":"He"},"finish_reason":null}]}"#
            .to_string(),
    );
    for _ in 1..plan.chunks {
        events.push(r#"{"choices":[{"delta":{"content":"y"},"finish_reason":null}]}"#.to_string());
    }
    events.push(r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#.to_string());
    events.push("[DONE]".to_string());
    for (index, event) in events.iter().enumerate() {
        let frame = format!("data: {event}\n\n");
        socket.write_all(frame.as_bytes()).await?;
        if index + 1 < events.len() {
            tokio::time::sleep(plan.gap).await;
        }
    }
    // Every byte is on the wire before the body can be considered finished;
    // a keep_alive:0 unload may only land from here on.
    state.stream_finished.store(true, Ordering::SeqCst);
    socket.shutdown().await
}

/// Next non-empty NDJSON line, or `None` at body end.
async fn next_line<S>(stream: &mut S, buf: &mut Vec<u8>) -> Option<String>
where
    S: Stream<Item = reqwest::Result<Bytes>> + Unpin,
{
    loop {
        if let Some(position) = buf.iter().position(|byte| *byte == b'\n') {
            let line: Vec<u8> = buf.drain(..=position).collect();
            let text = String::from_utf8_lossy(&line).trim().to_string();
            if !text.is_empty() {
                return Some(text);
            }
            continue;
        }
        match stream.next().await {
            Some(Ok(chunk)) => buf.extend_from_slice(&chunk),
            _ => {
                let text = String::from_utf8_lossy(buf).trim().to_string();
                buf.clear();
                return if text.is_empty() { None } else { Some(text) };
            }
        }
    }
}

async fn wait_for_unload(backend: &RawBackend) -> bool {
    for _ in 0..150 {
        if backend.state.unload_seen.load(Ordering::SeqCst) {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    false
}

#[tokio::test]
async fn keep_alive_zero_omitted_stream_unloads_after_stream_end() {
    // `stream` omitted: Ollama's chat default is streaming, while the old
    // is_streaming_request() default of false turned this into an immediate
    // teardown.
    let backend = spawn_raw_backend(SsePlan {
        chunks: 2,
        gap: Duration::from_millis(900),
        initial_delay: Duration::ZERO,
    })
    .await;
    let proxy = spawn_proxy_against_backend(&backend.base_url, 5, 60).await;

    let resp = proxy
        .client
        .post(format!("{}/api/chat", proxy.base_url))
        .json(&json!({
            "model": "test-model",
            "messages": [{ "role": "user", "content": "hi" }],
            "keep_alive": 0
        }))
        .send()
        .await
        .expect("POST /api/chat");
    assert_eq!(resp.status(), 200);

    let mut stream = resp.bytes_stream();
    let mut buf = Vec::new();
    let first = next_line(&mut stream, &mut buf)
        .await
        .expect("first NDJSON line");
    assert!(
        first.contains("\"message\""),
        "unexpected first line: {first}"
    );
    while next_line(&mut stream, &mut buf).await.is_some() {}

    assert!(
        wait_for_unload(&backend).await,
        "keep_alive:0 must unload after the stream ends"
    );
    assert!(
        !backend.state.premature_unload.load(Ordering::SeqCst),
        "keep_alive:0 unloaded the model before the response stream ended"
    );
}

#[tokio::test]
async fn keep_alive_zero_stream_outliving_flat_delay_unloads_after_stream_end() {
    // stream_timeout_seconds 1s while the backend keeps the stream open for
    // ~2.4s: a flat sleep(stream_timeout) teardown fires mid-stream; only
    // stream-end awareness keeps the model loaded until the last chunk. The
    // 100ms per-event pace leaves a 10x margin under the 1s inter-chunk bound.
    let backend = spawn_raw_backend(SsePlan {
        chunks: 24,
        gap: Duration::from_millis(100),
        initial_delay: Duration::ZERO,
    })
    .await;
    let proxy = spawn_proxy_against_backend(&backend.base_url, 5, 1).await;

    let resp = proxy
        .client
        .post(format!("{}/api/chat", proxy.base_url))
        .json(&json!({
            "model": "test-model",
            "messages": [{ "role": "user", "content": "hi" }],
            "stream": true,
            "keep_alive": 0
        }))
        .send()
        .await
        .expect("POST /api/chat");
    assert_eq!(resp.status(), 200);

    let mut stream = resp.bytes_stream();
    let mut buf = Vec::new();
    let mut last = String::new();
    while let Some(line) = next_line(&mut stream, &mut buf).await {
        last = line;
    }
    assert!(
        last.contains("\"done\":true"),
        "expected the final done chunk, got: {last}"
    );

    assert!(
        wait_for_unload(&backend).await,
        "keep_alive:0 must unload after the stream ends"
    );
    assert!(
        !backend.state.premature_unload.load(Ordering::SeqCst),
        "keep_alive:0 unloaded the model while the stream still outlived the flat timeout"
    );
}
