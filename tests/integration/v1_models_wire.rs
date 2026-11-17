// Wire-level regression for the rebuilt `GET /v1/models` envelope.
//
// The alias merge re-serialises the listing and hands hyper an exact-size body.
// Copying the backend's hop-by-hop framing header (`Transfer-Encoding`) onto
// that body makes hyper 1.x abort the client connection with zero bytes written
// — the client sees no status line at all. A reverse proxy in front of LM
// Studio may well answer chunked, so the rebuilt envelope must drop it and let
// hyper frame the body itself.
//
// The raw TCP backend mirrors tests/integration/passthrough_stream_timeout.rs
// because wiremock always answers with `content-length` framing.

use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use crate::common::spawn_proxy_against_backend;

struct RawBackend {
    base_url: String,
}

async fn spawn_chunked_listing_backend() -> RawBackend {
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

async fn serve_connection(socket: &mut TcpStream) -> std::io::Result<()> {
    let mut buf = vec![0u8; 8192];
    let read = socket.read(&mut buf).await?;
    if read == 0 {
        return Ok(());
    }
    let request = String::from_utf8_lossy(&buf[..read]);
    let request_line = request.lines().next().unwrap_or_default();
    if request_line.starts_with("GET /api/v1/models ") {
        let body = json!({
            "models": [{
                "key": "wire-model",
                "type": "llm",
                "publisher": "test",
                "architecture": "llama",
                "format": "gguf",
                "quantization": {"name": "Q4_K_M"},
                "max_context_length": 8192,
                "loaded_instances": [],
                "capabilities": {"vision": false, "trained_for_tool_use": false}
            }]
        })
        .to_string();
        write_json(socket, &body).await
    } else if request_line.starts_with("GET /v1/models ") {
        // Chunked framing, as a reverse proxy in front of LM Studio may send it.
        socket
            .write_all(
                b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n",
            )
            .await?;
        let body = json!({
            "object": "list",
            "data": [{"id": "wire-model", "object": "model", "owned_by": "organization_owner"}]
        })
        .to_string();
        let framed = format!("{:x}\r\n{}\r\n0\r\n\r\n", body.len(), body);
        socket.write_all(framed.as_bytes()).await?;
        socket.shutdown().await
    } else {
        socket
            .write_all(b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n")
            .await?;
        socket.shutdown().await
    }
}

async fn write_json(socket: &mut TcpStream, body: &str) -> std::io::Result<()> {
    let response = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    socket.write_all(response.as_bytes()).await?;
    socket.shutdown().await
}

#[tokio::test]
async fn v1_models_listing_survives_a_chunked_backend() {
    let backend = spawn_chunked_listing_backend().await;
    let proxy = spawn_proxy_against_backend(&backend.base_url, 5, 60).await;

    // Seed an alias so the merge (and therefore the rebuilt envelope) runs.
    let copy = proxy
        .client
        .post(proxy.url("/api/copy"))
        .json(&json!({"source": "wire-model", "destination": "wire-alias"}))
        .send()
        .await
        .expect("POST /api/copy");
    assert_eq!(copy.status(), 200, "alias creation must succeed");

    let resp = proxy
        .client
        .get(proxy.url("/v1/models"))
        .send()
        .await
        .expect("GET /v1/models against a chunked backend");
    assert_eq!(resp.status(), 200);

    // A copied framing header makes hyper abort before any byte: this parse
    // fails with an incomplete message instead of a listing.
    let body: serde_json::Value = resp
        .json()
        .await
        .expect("the rebuilt listing must reach the client as readable JSON");
    let ids: Vec<&str> = body["data"]
        .as_array()
        .expect("data array")
        .iter()
        .filter_map(|entry| entry["id"].as_str())
        .collect();
    assert!(ids.contains(&"wire-model"), "backend entry present: {body}");
    assert!(ids.contains(&"wire-alias"), "alias merged: {body}");
}
