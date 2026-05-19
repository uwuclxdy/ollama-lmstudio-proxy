use super::*;
use http_body_util::BodyExt;

#[tokio::test]
async fn root_returns_ollama_is_running() {
    let response = handle_ollama_root().await.unwrap();
    assert_eq!(response.status(), 200);
    let ct = response
        .headers()
        .get("Content-Type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        ct.contains("text/plain"),
        "expected text/plain, got: {}",
        ct
    );
    let body = response.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(body.as_ref(), b"Ollama is running");
}
