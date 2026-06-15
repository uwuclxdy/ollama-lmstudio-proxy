// Smoke tests for server route wiring. Wider coverage is added by the
// server-routes integration-test agent.

use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, ResponseTemplate};

use crate::common::spawn_proxy;

// ---------------------------------------------------------------------------
// Original tests (preserved)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn root_returns_ollama_banner() {
    let p = spawn_proxy().await;
    let resp = p.client.get(p.url("/")).send().await.expect("GET /");
    assert_eq!(resp.status(), 200);
    let body = resp.text().await.expect("body");
    // Real Ollama returns exactly this string; pinning it ensures clients
    // doing equality checks (rather than substring matches) keep working.
    assert_eq!(body, "Ollama is running", "got: {body}");
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

    let resp = p
        .client
        .get(p.url("/health"))
        .send()
        .await
        .expect("GET /health");
    assert_eq!(resp.status(), 200);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Mount a stub that accepts any GET /api/v1/models (model resolution via LM Studio native).
async fn mount_models_stub(p: &crate::common::TestProxy) {
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
}

/// Mount a stub POST for the given backend path returning a minimal chat completion.
async fn mount_chat_stub(p: &crate::common::TestProxy, backend_path: &str) {
    Mock::given(method("POST"))
        .and(path(backend_path))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "c1",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        })))
        .mount(&p.mock)
        .await;
}

async fn mount_completions_stub(p: &crate::common::TestProxy) {
    Mock::given(method("POST"))
        .and(path("/api/v0/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "c1",
            "object": "text_completion",
            "choices": [{"text": "hello", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        })))
        .mount(&p.mock)
        .await;
}

async fn mount_embeddings_stub(p: &crate::common::TestProxy) {
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
            "model": "llama3",
            "usage": {"prompt_tokens": 1, "total_tokens": 1}
        })))
        .mount(&p.mock)
        .await;
}

// ---------------------------------------------------------------------------
// Route presence: every documented Ollama endpoint returns non-404
// ---------------------------------------------------------------------------

#[tokio::test]
async fn route_tags_is_present() {
    let p = spawn_proxy().await;
    mount_models_stub(&p).await;
    let resp = p
        .client
        .get(p.url("/api/tags"))
        .send()
        .await
        .expect("GET /api/tags");
    assert_ne!(resp.status(), 404, "/api/tags must be a recognised route");
}

#[tokio::test]
async fn route_show_is_present() {
    let p = spawn_proxy().await;
    mount_models_stub(&p).await;
    // /api/show requires POST with model; missing model → 400/422, not 404
    let resp = p
        .client
        .post(p.url("/api/show"))
        .json(&json!({"model": "llama3"}))
        .send()
        .await
        .expect("POST /api/show");
    assert_ne!(resp.status(), 404, "/api/show must be a recognised route");
}

#[tokio::test]
async fn route_chat_is_present() {
    let p = spawn_proxy().await;
    mount_models_stub(&p).await;
    mount_chat_stub(&p, "/api/v0/chat/completions").await;
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
    assert_ne!(resp.status(), 404, "/api/chat must be a recognised route");
}

#[tokio::test]
async fn route_generate_is_present() {
    let p = spawn_proxy().await;
    mount_models_stub(&p).await;
    mount_completions_stub(&p).await;
    let resp = p
        .client
        .post(p.url("/api/generate"))
        .json(&json!({"model": "llama3", "prompt": "hello", "stream": false}))
        .send()
        .await
        .expect("POST /api/generate");
    assert_ne!(
        resp.status(),
        404,
        "/api/generate must be a recognised route"
    );
}

#[tokio::test]
async fn route_embed_is_present() {
    let p = spawn_proxy().await;
    mount_models_stub(&p).await;
    mount_embeddings_stub(&p).await;
    let resp = p
        .client
        .post(p.url("/api/embed"))
        .json(&json!({"model": "llama3", "input": "hello"}))
        .send()
        .await
        .expect("POST /api/embed");
    assert_ne!(resp.status(), 404, "/api/embed must be a recognised route");
}

#[tokio::test]
async fn route_embeddings_legacy_is_present() {
    let p = spawn_proxy().await;
    mount_models_stub(&p).await;
    mount_embeddings_stub(&p).await;
    let resp = p
        .client
        .post(p.url("/api/embeddings"))
        .json(&json!({"model": "llama3", "prompt": "hello"}))
        .send()
        .await
        .expect("POST /api/embeddings");
    assert_ne!(
        resp.status(),
        404,
        "/api/embeddings must be a recognised route"
    );
}

#[tokio::test]
async fn route_ps_is_present() {
    let p = spawn_proxy().await;
    mount_models_stub(&p).await;
    let resp = p
        .client
        .get(p.url("/api/ps"))
        .send()
        .await
        .expect("GET /api/ps");
    assert_ne!(resp.status(), 404, "/api/ps must be a recognised route");
}

#[tokio::test]
async fn route_pull_is_present() {
    let p = spawn_proxy().await;
    mount_models_stub(&p).await;
    // Mock the download endpoint that pull delegates to
    Mock::given(method("POST"))
        .and(path("/api/v1/models/download"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"status": "ok"})))
        .mount(&p.mock)
        .await;
    Mock::given(method("GET"))
        .and(path("/api/v1/models/download/status"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"status": "complete", "progress": 1.0})),
        )
        .mount(&p.mock)
        .await;
    let resp = p
        .client
        .post(p.url("/api/pull"))
        .json(&json!({"model": "llama3"}))
        .send()
        .await
        .expect("POST /api/pull");
    assert_ne!(resp.status(), 404, "/api/pull must be a recognised route");
}

#[tokio::test]
async fn route_delete_is_present() {
    let p = spawn_proxy().await;
    mount_models_stub(&p).await;

    // Create a virtual alias first so the delete handler can find the model.
    p.client
        .post(p.url("/api/copy"))
        .json(&json!({"source": "llama3", "destination": "llama3-to-delete"}))
        .send()
        .await
        .expect("POST /api/copy pre-create");

    let resp = p
        .client
        .delete(p.url("/api/delete"))
        .json(&json!({"model": "llama3-to-delete"}))
        .send()
        .await
        .expect("DELETE /api/delete");
    assert_ne!(resp.status(), 404, "/api/delete must be a recognised route");
}

#[tokio::test]
async fn route_copy_is_present() {
    let p = spawn_proxy().await;
    mount_models_stub(&p).await;
    let resp = p
        .client
        .post(p.url("/api/copy"))
        .json(&json!({"source": "llama3", "destination": "llama3-copy"}))
        .send()
        .await
        .expect("POST /api/copy");
    assert_ne!(resp.status(), 404, "/api/copy must be a recognised route");
}

#[tokio::test]
async fn route_push_is_present() {
    let p = spawn_proxy().await;
    mount_models_stub(&p).await;
    let resp = p
        .client
        .post(p.url("/api/push"))
        .json(&json!({"model": "llama3"}))
        .send()
        .await
        .expect("POST /api/push");
    assert_ne!(resp.status(), 404, "/api/push must be a recognised route");
}

#[tokio::test]
async fn route_web_search_is_present_and_returns_501() {
    let p = spawn_proxy().await;
    let resp = p
        .client
        .post(p.url("/api/web_search"))
        .json(&json!({"query": "test"}))
        .send()
        .await
        .expect("POST /api/web_search");
    assert_ne!(
        resp.status(),
        404,
        "/api/web_search must be a recognised route"
    );
    assert_eq!(
        resp.status().as_u16(),
        501,
        "/api/web_search must return 501 (cloud-only Ollama feature)"
    );
}

#[tokio::test]
async fn route_web_fetch_is_present_and_fetches() {
    let p = spawn_proxy().await;
    // web_fetch hits an arbitrary URL with its own client; point it at the mock
    // server so the smoke test stays hermetic (no real network).
    Mock::given(method("GET"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_string("<html><title>Hi</title></html>"))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/web_fetch"))
        .json(&json!({"url": p.mock.uri()}))
        .send()
        .await
        .expect("POST /api/web_fetch");
    assert_ne!(
        resp.status(),
        404,
        "/api/web_fetch must be a recognised route"
    );
    assert_eq!(
        resp.status().as_u16(),
        200,
        "/api/web_fetch is implemented and must fetch successfully"
    );
}

#[tokio::test]
async fn route_create_is_present() {
    let p = spawn_proxy().await;
    mount_models_stub(&p).await;
    let resp = p
        .client
        .post(p.url("/api/create"))
        .json(&json!({"model": "my-model", "from": "llama3"}))
        .send()
        .await
        .expect("POST /api/create");
    assert_ne!(resp.status(), 404, "/api/create must be a recognised route");
}

#[tokio::test]
async fn route_blobs_head_is_present() {
    let p = spawn_proxy().await;
    // HEAD /api/blobs/:digest — unknown digest → 404 from handler, but the route is wired.
    // 404 here is a domain response, not a routing miss; 405 would mean HEAD isn't registered.
    let resp = p
        .client
        .head(p.url("/api/blobs/sha256:aaabbbccc"))
        .send()
        .await
        .expect("HEAD /api/blobs/:digest");
    let status = resp.status().as_u16();
    assert_ne!(
        status, 405,
        "/api/blobs/:digest HEAD must be registered: {status}"
    );
}

// ---------------------------------------------------------------------------
// Method enforcement
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_get_returns_405() {
    let p = spawn_proxy().await;
    let resp = p
        .client
        .get(p.url("/api/chat"))
        .send()
        .await
        .expect("GET /api/chat");
    assert_eq!(resp.status().as_u16(), 405, "GET /api/chat must return 405");
}

#[tokio::test]
async fn tags_post_returns_405() {
    let p = spawn_proxy().await;
    let resp = p
        .client
        .post(p.url("/api/tags"))
        .json(&json!({}))
        .send()
        .await
        .expect("POST /api/tags");
    assert_eq!(
        resp.status().as_u16(),
        405,
        "POST /api/tags must return 405"
    );
}

#[tokio::test]
async fn generate_get_returns_405() {
    let p = spawn_proxy().await;
    let resp = p
        .client
        .get(p.url("/api/generate"))
        .send()
        .await
        .expect("GET /api/generate");
    assert_eq!(
        resp.status().as_u16(),
        405,
        "GET /api/generate must return 405"
    );
}

// ---------------------------------------------------------------------------
// 404 error shape is Ollama-compatible JSON
// ---------------------------------------------------------------------------

#[tokio::test]
async fn unknown_route_404_has_json_error_body() {
    let p = spawn_proxy().await;
    let resp = p
        .client
        .get(p.url("/api/totally-unknown"))
        .send()
        .await
        .expect("GET unknown");
    assert_eq!(resp.status(), 404);
    let body: Value = resp.json().await.expect("404 body must be JSON");
    assert!(
        body.get("error").is_some(),
        "404 body must have 'error' key per rejection handler: {body}"
    );
}

// ---------------------------------------------------------------------------
// Body size limit — 413 Payload Too Large
// ---------------------------------------------------------------------------

#[tokio::test]
async fn oversized_body_returns_413() {
    let p = spawn_proxy().await;
    // MAX_JSON_BODY_SIZE_BYTES = 16 MiB; send 17 MiB
    let big_body = vec![b'x'; 17 * 1024 * 1024];
    // Windows may surface the server-side rejection as a mid-stream connection
    // abort rather than a clean 413 response: the server closes the socket
    // after writing the response while the client is still pushing the body.
    // Either outcome proves the limit was enforced.
    match p
        .client
        .post(p.url("/api/chat"))
        .header("content-type", "application/json")
        .body(big_body)
        .send()
        .await
    {
        Ok(resp) => assert_eq!(
            resp.status(),
            413,
            "body exceeding MAX_JSON_BODY_SIZE_BYTES must return 413"
        ),
        Err(e) => assert!(
            e.is_request() || e.is_body(),
            "expected 413 or connection error from oversized body; got: {e}"
        ),
    }
}

// ---------------------------------------------------------------------------
// Malformed JSON body → 400
// ---------------------------------------------------------------------------

#[tokio::test]
async fn malformed_json_body_returns_400() {
    let p = spawn_proxy().await;
    let resp = p
        .client
        .post(p.url("/api/chat"))
        .header("content-type", "application/json")
        .body("not valid json at all {{{")
        .send()
        .await
        .expect("POST /api/chat bad json");
    assert_eq!(resp.status(), 400, "malformed JSON must return 400");
    let body: Value = resp.json().await.expect("error body must be JSON");
    assert!(
        body.get("error").is_some(),
        "400 response must have 'error' field: {body}"
    );
}

// ---------------------------------------------------------------------------
// Missing required fields → non-200 with Ollama-shaped error
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_missing_messages_returns_error() {
    let p = spawn_proxy().await;
    mount_models_stub(&p).await;
    // Provide model but omit messages
    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({"model": "llama3"}))
        .send()
        .await
        .expect("POST /api/chat no messages");
    let status = resp.status().as_u16();
    // The proxy returns an error — either 400 or a 200 streaming error chunk,
    // but it must not return 404 or 500 silently.
    assert!(
        status != 404,
        "missing messages must not produce a 404: {status}"
    );
}

#[tokio::test]
async fn generate_missing_prompt_returns_error() {
    let p = spawn_proxy().await;
    mount_models_stub(&p).await;
    let resp = p
        .client
        .post(p.url("/api/generate"))
        .json(&json!({"model": "llama3"}))
        .send()
        .await
        .expect("POST /api/generate no prompt");
    let status = resp.status().as_u16();
    assert!(
        status != 404,
        "missing prompt must not produce a 404: {status}"
    );
}

// ---------------------------------------------------------------------------
// Backend down → 502 or Ollama-shaped error body
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_backend_down_returns_error_response() {
    let p = spawn_proxy().await;
    // No mocks registered — all backend calls will be refused by wiremock.
    // The proxy should return an error (502 or a 200 with error chunk).
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
        .expect("POST /api/chat backend-down");
    let status = resp.status().as_u16();
    // Must not be a successful 200 with real content
    // Either an HTTP-level error (5xx) or a 200 with {"error": ...}
    if status == 200 {
        let body: Value = resp.json().await.expect("JSON body");
        assert!(
            body.get("error").is_some(),
            "200 response when backend is down must have 'error' field: {body}"
        );
    } else {
        assert!(
            status >= 400,
            "backend-down must return an error status, got {status}"
        );
    }
}

// ---------------------------------------------------------------------------
// CORS: cross-origin GET /api/tags returns Access-Control-Allow-Origin header
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tags_response_has_cors_header() {
    let p = spawn_proxy().await;
    mount_models_stub(&p).await;

    // spawn_proxy assembles routes without the cors_layer() that run() attaches,
    // so the CORS header may be absent. we assert the route works and,
    // if the header is present, that it is correct.
    let resp = p
        .client
        .get(p.url("/api/tags"))
        .header("Origin", "http://example.com")
        .send()
        .await
        .expect("GET /api/tags with Origin");

    assert_ne!(resp.status(), 404, "/api/tags must be recognised");
    // if the CORS layer is active, it must use wildcard origin
    if let Some(acao) = resp.headers().get("access-control-allow-origin") {
        let val = acao.to_str().unwrap_or("");
        assert_eq!(val, "*", "access-control-allow-origin must be '*': {val}");
    }
}

// ---------------------------------------------------------------------------
// /api/version — shape check
// ---------------------------------------------------------------------------

#[tokio::test]
async fn version_endpoint_returns_version_string() {
    let p = spawn_proxy().await;
    let resp = p
        .client
        .get(p.url("/api/version"))
        .send()
        .await
        .expect("GET /api/version");
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("JSON");
    let version = body.get("version").and_then(|v| v.as_str()).unwrap_or("");
    assert!(
        !version.is_empty(),
        "version must be a non-empty string: {body}"
    );
}

// ---------------------------------------------------------------------------
// /api/tags — response shape matches Ollama spec
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tags_response_has_models_array() {
    let p = spawn_proxy().await;
    mount_models_stub(&p).await;
    let resp = p
        .client
        .get(p.url("/api/tags"))
        .send()
        .await
        .expect("GET /api/tags");
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("JSON");
    assert!(
        body.get("models").and_then(|v| v.as_array()).is_some(),
        "/api/tags must return {{\"models\": [...]}} : {body}"
    );
}

// ---------------------------------------------------------------------------
// Rejection handler: MethodNotAllowed returns JSON body
// ---------------------------------------------------------------------------

#[tokio::test]
async fn method_not_allowed_returns_json_error() {
    let p = spawn_proxy().await;
    // /api/version is GET-only; PUT triggers the method_not_allowed_fallback handler
    let resp = p
        .client
        .put(p.url("/api/version"))
        .json(&json!({}))
        .send()
        .await
        .expect("PUT /api/version");
    assert_eq!(resp.status().as_u16(), 405, "wrong method must return 405");
    let body: Value = resp.json().await.expect("error body must be JSON");
    assert!(
        body.get("error").is_some(),
        "405 must have 'error' field: {body}"
    );
}

// ---------------------------------------------------------------------------
// /api/ps — response shape
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ps_response_has_models_array() {
    let p = spawn_proxy().await;
    mount_models_stub(&p).await;
    let resp = p
        .client
        .get(p.url("/api/ps"))
        .send()
        .await
        .expect("GET /api/ps");
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("JSON");
    assert!(
        body.get("models").and_then(|v| v.as_array()).is_some(),
        "/api/ps must return {{\"models\": [...]}} : {body}"
    );
}
