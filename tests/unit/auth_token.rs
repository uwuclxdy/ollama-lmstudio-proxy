// ── lmstudio bearer-token injection ─────────────────────────────────────────
//
// These tests verify that the reqwest Client is built with (or without) an
// Authorization default header depending on whether `lmstudio_token` is set.
// They do not exercise live HTTP; the header is inspected via a loopback
// mock server that echoes request headers back as JSON.

/// Build a reqwest Client with a default Authorization header, mirroring
/// the logic in ProxyServer::new_with_state_dir.
fn build_client_with_token(token: Option<&str>) -> reqwest::Client {
    let mut builder = reqwest::Client::builder();
    if let Some(t) = token {
        let mut default_headers = reqwest::header::HeaderMap::new();
        let value = reqwest::header::HeaderValue::from_str(&format!("Bearer {}", t)).unwrap();
        default_headers.insert(reqwest::header::AUTHORIZATION, value);
        builder = builder.default_headers(default_headers);
    }
    builder.build().unwrap()
}

#[tokio::test]
async fn client_sends_auth_header_when_token_configured() {
    // Spin up a tiny one-shot server that records the Authorization header it receives.
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let captured: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let captured_clone = captured.clone();

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut buf = vec![0u8; 4096];
        let n = stream.read(&mut buf).await.unwrap();
        let raw = String::from_utf8_lossy(&buf[..n]);

        // Extract Authorization header value from the raw HTTP request.
        let auth = raw.lines().find_map(|line| {
            let lower = line.to_ascii_lowercase();
            if lower.starts_with("authorization:") {
                Some(line["authorization:".len()..].trim().to_string())
            } else {
                None
            }
        });
        *captured_clone.lock().unwrap() = auth;

        // Minimal HTTP 200 response so the client doesn't error.
        stream
            .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\n{}")
            .await
            .unwrap();
    });

    let client = build_client_with_token(Some("secret-token"));
    let _ = client.get(format!("http://{}/", addr)).send().await;

    let auth_header = captured.lock().unwrap().clone();
    assert_eq!(
        auth_header.as_deref(),
        Some("Bearer secret-token"),
        "client must send Authorization: Bearer <token> when token is configured"
    );
}

#[tokio::test]
async fn client_sends_no_auth_header_when_token_unset() {
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let captured: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let captured_clone = captured.clone();

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut buf = vec![0u8; 4096];
        let n = stream.read(&mut buf).await.unwrap();
        let raw = String::from_utf8_lossy(&buf[..n]);

        let auth = raw.lines().find_map(|line| {
            let lower = line.to_ascii_lowercase();
            if lower.starts_with("authorization:") {
                Some(line["authorization:".len()..].trim().to_string())
            } else {
                None
            }
        });
        *captured_clone.lock().unwrap() = auth;

        stream
            .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\n{}")
            .await
            .unwrap();
    });

    let client = build_client_with_token(None);
    let _ = client.get(format!("http://{}/", addr)).send().await;

    let auth_header = captured.lock().unwrap().clone();
    assert!(
        auth_header.is_none(),
        "client must not send Authorization when no token is configured, got: {:?}",
        auth_header
    );
}

#[tokio::test]
async fn caller_auth_overrides_client_default_header() {
    // When the passthrough copies the caller's Authorization header as a
    // per-request header, it must win over the client default — reqwest's
    // documented merge behaviour: per-request headers take precedence.
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let captured: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let captured_clone = captured.clone();

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut buf = vec![0u8; 4096];
        let n = stream.read(&mut buf).await.unwrap();
        let raw = String::from_utf8_lossy(&buf[..n]);

        let auth = raw.lines().find_map(|line| {
            let lower = line.to_ascii_lowercase();
            if lower.starts_with("authorization:") {
                Some(line["authorization:".len()..].trim().to_string())
            } else {
                None
            }
        });
        *captured_clone.lock().unwrap() = auth;

        stream
            .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\n{}")
            .await
            .unwrap();
    });

    // Client has a default proxy token, but the request carries the caller's own token.
    let client = build_client_with_token(Some("proxy-token"));
    let _ = client
        .get(format!("http://{}/", addr))
        .header(
            reqwest::header::AUTHORIZATION,
            reqwest::header::HeaderValue::from_static("Bearer caller-token"),
        )
        .send()
        .await;

    let auth_header = captured.lock().unwrap().clone();
    assert_eq!(
        auth_header.as_deref(),
        Some("Bearer caller-token"),
        "caller-supplied Authorization must override the proxy default token"
    );
}
