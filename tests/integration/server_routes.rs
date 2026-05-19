// Smoke tests for server route wiring. Wider coverage is added by the
// server-routes integration-test agent.

use serde_json::json;
use wiremock::{Mock, ResponseTemplate};
use wiremock::matchers::{method, path};

use crate::common::spawn_proxy;

#[tokio::test]
async fn root_returns_ollama_banner() {
    let p = spawn_proxy().await;
    let resp = p.client.get(p.url("/")).send().await.expect("GET /");
    assert_eq!(resp.status(), 200);
    let body = resp.text().await.expect("body");
    assert!(body.to_ascii_lowercase().contains("ollama"), "got: {body}");
}

#[tokio::test]
async fn unknown_route_returns_404() {
    let p = spawn_proxy().await;
    let resp = p
        .client
        .get(p.url("/api/does-not-exist"))
        .send()
        .await
        .expect("GET unknown");
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn health_check_reaches_lmstudio_backend() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": []
        })))
        .mount(&p.mock)
        .await;

    let resp = p.client.get(p.url("/health")).send().await.expect("GET /health");
    assert_eq!(resp.status(), 200);
}
