// Pins the passthrough timeout wiring: `forward_passthrough` hands
// `load_timeout_seconds` and `stream_timeout_seconds` to
// `handle_lmstudio_passthrough` as two adjacent u64 positionals, so a swap
// there compiles silently. A proxy with stream_timeout 1 / load_timeout 5
// pointed at a backend that sends response headers and then stalls the body
// for ~3s must surface the `Stream timeout` error frame: only the stream
// loop's own 1s bound can fire mid-stall. Swapping the arguments gives the
// stream loop the 5s load bound, the stall completes normally, and no error
// frame ever arrives.
//
// The raw TCP backend mirrors tests/integration/keep_alive_streaming.rs
// because wiremock buffers whole bodies and cannot hold a stream open
// mid-body.

use std::time::Duration;

use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use crate::common::spawn_proxy_against_backend;
use ollama_lmstudio_proxy::constants::ERROR_TIMEOUT;

struct RawBackend {
    base_url: String,
}

async fn spawn_raw_backend() -> RawBackend {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind raw backend");
    let addr = listener.local_addr().expect("raw backend addr");
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                break;
            };
            tokio::spawn(async move {
                if let Err(e) = serve_connection(&mut socket).await {
                    log::debug!("backend connection error: {e}");
                }
            });
        }
    });
    RawBackend {
        base_url: format!("http://{addr}"),
    }
}

/// Minimal HTTP/1.1 responder: one request per connection, close-delimited
/// bodies so the stall is visible on the wire.
async fn serve_connection(socket: &mut TcpStream) -> std::io::Result<()> {
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
    } else if request_line.starts_with("POST /v1/chat/completions ") {
        write_stalling_sse(socket).await
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

/// Headers immediately, then a ~3s stall with no body bytes, then a normal
/// OpenAI SSE pair. Only a stream timeout under 3s can surface the proxy's
/// error frame before the stall ends.
async fn write_stalling_sse(socket: &mut TcpStream) -> std::io::Result<()> {
    socket
        .write_all(
            b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n",
        )
        .await?;
    tokio::time::sleep(Duration::from_secs(3)).await;
    socket
        .write_all(
            b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n",
        )
        .await?;
    socket.write_all(b"data: [DONE]\n\n").await?;
    socket.shutdown().await
}

/// Next non-empty SSE line, or `None` at body end.
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

#[tokio::test]
async fn passthrough_stream_timeout_uses_stream_bound_not_load_bound() {
    let backend = spawn_raw_backend().await;
    let proxy = spawn_proxy_against_backend(&backend.base_url, 5, 1).await;

    let resp = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .json(&json!({
            "model": "test-model",
            "messages": [{ "role": "user", "content": "hi" }],
            "stream": true
        }))
        .send()
        .await
        .expect("POST /v1/chat/completions");
    assert_eq!(resp.status(), 200);

    let mut stream = resp.bytes_stream();
    let mut buf = Vec::new();
    let mut lines = Vec::new();
    let mut saw_timeout = false;
    while let Some(line) = next_line(&mut stream, &mut buf).await {
        if line.contains(ERROR_TIMEOUT) {
            saw_timeout = true;
            break;
        }
        lines.push(line);
    }

    assert!(
        saw_timeout,
        "expected the '{ERROR_TIMEOUT}' error frame while the backend stalled, got: {lines:?}"
    );
}
