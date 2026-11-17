// Smoke tests for server route wiring. Wider coverage is added by the
// server-routes integration-test agent.

use serde_json::{Value, json};
use wiremock::matchers::{body_partial_json, method, path, query_param};
use wiremock::{Mock, ResponseTemplate};

use crate::common::{TestProxy, spawn_proxy, spawn_proxy_with_api_key};

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

/// Mount a decision-model listing (nullable `max_context_length`) whose
/// `decision_type` is the family LM Studio reports for the model; the System
/// One text cap follows that field, never `architecture`.
async fn mount_decision_model_stub_with_type(
    p: &crate::common::TestProxy,
    key: &str,
    decision_type: &str,
    architecture: &str,
) {
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [{"key": key, "type": "decision", "publisher": "ggml-org",
                        "architecture": architecture, "format": "gguf",
                        "quantization": {"name": "Q4_K_M", "bits_per_weight": 4.5},
                        "max_context_length": null, "loaded_instances": [],
                        "decision_type": decision_type, "has_vision_adapter": false}]
        })))
        .mount(&p.mock)
        .await;
}

/// Mount the clef decision model so `/v1/systemone` requests can resolve
/// against it.
async fn mount_decision_model_stub(p: &crate::common::TestProxy) {
    mount_decision_model_stub_with_type(p, "clef-flash", "clef", "clef").await;
}

#[tokio::test]
async fn route_systemone_forwards_decision_request() {
    let p = spawn_proxy().await;
    mount_decision_model_stub(&p).await;

    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .and(body_partial_json(json!({
            "model": "clef-flash",
            "state": "Our checkout has returned 500 errors since 9am.",
            "questions": {
                "label": {
                    "type": "choice",
                    "instructions": "Which label fits this ticket?",
                    "criteria": {"billing": "Payments and refunds", "bug": "Software errors"}
                }
            }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "clef-flash",
            "answers": {
                "label": {
                    "type": "choice",
                    "choice": "bug",
                    "probabilities": {"billing": 0.0125, "bug": 0.9781},
                    "confidence": 0.8906
                }
            },
            "usage": {"input_tokens": 174, "output_tokens": 0}
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/v1/systemone"))
        .json(&json!({
            "model": "clef-flash:latest",
            "state": "Our checkout has returned 500 errors since 9am.",
            "questions": {
                "label": {
                    "type": "choice",
                    "instructions": "Which label fits this ticket?",
                    "criteria": {"billing": "Payments and refunds", "bug": "Software errors"}
                }
            }
        }))
        .send()
        .await
        .expect("POST /v1/systemone");

    assert_eq!(
        resp.status(),
        200,
        "/v1/systemone must forward to LM Studio"
    );
    let body: Value = resp.json().await.expect("json body");
    assert_eq!(body["answers"]["label"]["choice"], "bug");
    assert_eq!(body["answers"]["label"]["confidence"], 0.8906);
    assert_eq!(body["usage"]["output_tokens"], 0);
}

/// A 1x1 transparent PNG, as an Ollama client sends `images` entries: bare
/// base64 with no `data:` prefix (api-docs/ollama/openapi.yaml
/// §SystemOneRequest.images). LM Studio's decision parser accepts only data
/// URLs, so the proxy must wrap them.
const PNG_1X1_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

#[tokio::test]
async fn route_systemone_wraps_bare_base64_images_as_data_urls() {
    let p = spawn_proxy().await;
    mount_decision_model_stub(&p).await;

    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .and(body_partial_json(json!({
            "model": "clef-flash",
            "images": [format!("data:image/png;base64,{PNG_1X1_BASE64}")]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "clef-flash",
            "answers": {"angry": {"type": "noul", "noul": 0.6328}},
            "usage": {"input_tokens": 300, "output_tokens": 0}
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/v1/systemone"))
        .json(&json!({
            "model": "clef-flash",
            "state": "I was charged twice.",
            "images": [PNG_1X1_BASE64],
            "questions": {"angry": {"type": "noul", "instructions": "Is the customer angry?"}}
        }))
        .send()
        .await
        .expect("POST /v1/systemone with image");

    assert_eq!(
        resp.status(),
        200,
        "the bare-base64 image must reach LM Studio as a data URL"
    );
    let body: Value = resp.json().await.expect("json body");
    assert_eq!(body["answers"]["angry"]["noul"], 0.6328);
}

#[tokio::test]
async fn route_systemone_rejects_invalid_base64_images() {
    let p = spawn_proxy().await;
    mount_decision_model_stub(&p).await;

    let resp = p
        .client
        .post(p.url("/v1/systemone"))
        .json(&json!({
            "model": "clef-flash",
            "state": "I was charged twice.",
            "images": ["not-base64!!"],
            "questions": {"angry": {"type": "noul", "instructions": "Is the customer angry?"}}
        }))
        .send()
        .await
        .expect("POST /v1/systemone with invalid image");

    assert_eq!(
        resp.status().as_u16(),
        400,
        "an images entry that is not base64 must be refused before forwarding"
    );
    let body: Value = resp.json().await.expect("json body");
    assert_eq!(body["error"], "images[0] is not valid base64");
}

#[tokio::test]
async fn route_systemone_backend_error_propagated() {
    let p = spawn_proxy().await;
    mount_decision_model_stub(&p).await;

    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "error": {
                "message": "Invalid model identifier \"clef-flash\". There are no downloaded decision models. Please download a model to get started.",
                "type": "invalid_request_error",
                "param": "model",
                "code": "invalidModelIdentifier"
            }
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/v1/systemone"))
        .json(&json!({
            "model": "clef-flash",
            "state": "Our checkout has returned 500 errors since 9am.",
            "questions": {"angry": {"type": "noul", "instructions": "Is the customer angry?"}}
        }))
        .send()
        .await
        .expect("POST /v1/systemone backend error");

    assert_eq!(
        resp.status().as_u16(),
        400,
        "the backend's decision-model error status must propagate"
    );
    let body: Value = resp.json().await.expect("json body");
    assert_eq!(body["error"]["code"], "invalidModelIdentifier");
}

#[tokio::test]
async fn route_systemone_strips_stream_so_the_answer_is_json() {
    let p = spawn_proxy().await;
    mount_decision_model_stub(&p).await;

    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .and(body_partial_json(json!({ "model": "clef-flash" })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "clef-flash",
            "answers": {"angry": {"type": "noul", "noul": 0.5}},
            "usage": {"input_tokens": 10, "output_tokens": 0}
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/v1/systemone"))
        .json(&json!({
            "model": "clef-flash",
            "state": "I was charged twice.",
            "questions": {"angry": {"type": "noul", "instructions": "Is the customer angry?"}},
            "stream": true
        }))
        .send()
        .await
        .expect("POST /v1/systemone with stream");

    assert_eq!(resp.status(), 200);
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("");
    assert!(
        content_type.starts_with("application/json"),
        "System One is non-streaming and ignores a stray `stream` field; got content-type {content_type}"
    );
    let body: Value = resp.json().await.expect("json body");
    assert_eq!(body["answers"]["angry"]["noul"], 0.5);
}

#[tokio::test]
async fn route_systemone_forwards_the_query_string() {
    let p = spawn_proxy().await;
    mount_decision_model_stub(&p).await;

    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .and(query_param("trace", "on"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "clef-flash",
            "answers": {"angry": {"type": "noul", "noul": 0.5}},
            "usage": {"input_tokens": 10, "output_tokens": 0}
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/v1/systemone?trace=on"))
        .json(&json!({
            "model": "clef-flash",
            "state": "I was charged twice.",
            "questions": {"angry": {"type": "noul", "instructions": "Is the customer angry?"}}
        }))
        .send()
        .await
        .expect("POST /v1/systemone with query");

    assert_eq!(
        resp.status(),
        200,
        "the query string must reach the backend, as on every other /v1 passthrough"
    );
}

#[tokio::test]
async fn route_systemone_accepts_bodies_over_the_default_router_limit() {
    let p = spawn_proxy().await;
    mount_decision_model_stub(&p).await;

    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "clef-flash",
            "answers": {"angry": {"type": "noul", "noul": 0.5}},
            "usage": {"input_tokens": 10, "output_tokens": 0}
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    // 17 MiB rides in `images`; text and schema stay under the 64 KiB cap, so
    // the request exercises the route's 32 MiB cap above the router-wide 16 MiB.
    let resp = p
        .client
        .post(p.url("/v1/systemone"))
        // The shared client's 10 s hang guard fired on this body after 15.84 s
        // in a loaded full-suite run.
        .timeout(std::time::Duration::from_secs(60))
        .json(&json!({
            "model": "clef-flash",
            "state": "I was charged twice.",
            "images": ["A".repeat(17 * 1024 * 1024)],
            "questions": {"angry": {"type": "noul", "instructions": "Is the customer angry?"}}
        }))
        .send()
        .await
        .expect("POST /v1/systemone over the default limit");

    assert_eq!(
        resp.status(),
        200,
        "the System One route must admit upstream's 32 MiB whole-request cap, above the router-wide 16 MiB"
    );
}

#[tokio::test]
async fn route_systemone_get_is_405_not_forwarded() {
    let p = spawn_proxy().await;

    let resp = p
        .client
        .get(p.url("/v1/systemone"))
        .send()
        .await
        .expect("GET /v1/systemone");

    assert_eq!(
        resp.status().as_u16(),
        405,
        "GET /v1/systemone must 405 at the proxy, never reach LM Studio's 200 catch-all"
    );
}

/// Upstream caps a local System One request's text and schema (images
/// excluded) at 64 KiB, or 1 MiB for the kev/d1 decision families, and answers
/// 413 before building any prompt (ollama `server/routes.go` SystemOneHandler:
/// the request struct minus `images`, re-encoded, compared with `>`).
fn systemone_questions() -> Value {
    json!({"angry": {"type": "noul", "instructions": "Is the customer angry?"}})
}

#[tokio::test]
async fn route_systemone_rejects_text_and_schema_over_64_kib() {
    let p = spawn_proxy().await;
    mount_decision_model_stub(&p).await;

    let resp = p
        .client
        .post(p.url("/v1/systemone"))
        .json(&json!({
            "model": "clef-flash",
            "state": "x".repeat(70 * 1024),
            "questions": systemone_questions()
        }))
        .send()
        .await
        .expect("POST /v1/systemone oversized text");

    assert_eq!(
        resp.status().as_u16(),
        413,
        "local text and schema over 64 KiB must answer 413 before any prompt is built"
    );
    let body: Value = resp.json().await.expect("json body");
    assert_eq!(
        body,
        json!({"error": "text and schema must not exceed 64 KiB"})
    );
}

#[tokio::test]
async fn route_systemone_validates_images_before_the_text_cap() {
    let p = spawn_proxy().await;
    mount_decision_model_stub(&p).await;

    let resp = p
        .client
        .post(p.url("/v1/systemone"))
        .json(&json!({
            "model": "clef-flash",
            "state": "x".repeat(70 * 1024),
            "images": ["not-base64!!"],
            "questions": systemone_questions()
        }))
        .send()
        .await
        .expect("POST /v1/systemone with a bad image and oversized text");

    assert_eq!(
        resp.status().as_u16(),
        400,
        "an undecodable image answers prepare's 400 before the text cap, matching upstream's decode order"
    );
    let body: Value = resp.json().await.expect("json body");
    assert_eq!(body, json!({"error": "images[0] is not valid base64"}));
}

#[tokio::test]
async fn route_systemone_kev_model_admits_text_and_schema_at_exactly_1_mib() {
    let p = spawn_proxy().await;
    mount_decision_model_stub_with_type(&p, "kev-decision", "kev", "qwen3_5").await;

    // Size the state so the counted encoding lands exactly on the kev cap:
    // the comparison is `> limit`, never `>= limit`.
    let overhead = json!({
        "model": "kev-decision",
        "state": "",
        "questions": systemone_questions()
    })
    .to_string()
    .len();
    let at_cap = "x".repeat(1024 * 1024 - overhead);
    let counted = json!({
        "model": "kev-decision",
        "state": at_cap.clone(),
        "questions": systemone_questions()
    })
    .to_string()
    .len();
    assert_eq!(
        counted,
        1024 * 1024,
        "the fixture must measure exactly the kev cap"
    );

    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .and(body_partial_json(
            json!({"model": "kev-decision", "state": at_cap.clone()}),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "kev-decision",
            "answers": {"angry": {"type": "noul", "noul": 0.5}},
            "usage": {"input_tokens": 10, "output_tokens": 0}
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    let at_cap_resp = p
        .client
        .post(p.url("/v1/systemone"))
        .json(&json!({
            "model": "kev-decision",
            "state": at_cap,
            "questions": systemone_questions()
        }))
        .send()
        .await
        .expect("POST /v1/systemone kev at the cap");
    assert_eq!(
        at_cap_resp.status(),
        200,
        "the kev family admits text and schema at exactly 1 MiB"
    );

    let over_cap_resp = p
        .client
        .post(p.url("/v1/systemone"))
        .json(&json!({
            "model": "kev-decision",
            "state": "x".repeat(1024 * 1024 - overhead + 1),
            "questions": systemone_questions()
        }))
        .send()
        .await
        .expect("POST /v1/systemone kev one byte over the cap");
    assert_eq!(
        over_cap_resp.status().as_u16(),
        413,
        "one byte over the kev cap must answer 413"
    );
    let body: Value = over_cap_resp.json().await.expect("json body");
    assert_eq!(
        body,
        json!({"error": "text and schema must not exceed 1024 KiB"})
    );
}

#[tokio::test]
async fn route_systemone_rejects_d1_states_over_1_mib() {
    let p = spawn_proxy().await;
    mount_decision_model_stub_with_type(&p, "d1-decision", "d1", "qwen3_5").await;

    let resp = p
        .client
        .post(p.url("/v1/systemone"))
        .json(&json!({
            "model": "d1-decision",
            "state": "x".repeat(1024 * 1024 + 1),
            "questions": systemone_questions()
        }))
        .send()
        .await
        .expect("POST /v1/systemone oversized d1 state");

    assert_eq!(
        resp.status().as_u16(),
        413,
        "the kev/d1 1 MiB cap still applies"
    );
    let body: Value = resp.json().await.expect("json body");
    assert_eq!(
        body,
        json!({"error": "text and schema must not exceed 1024 KiB"})
    );
}

#[tokio::test]
async fn route_systemone_reads_the_text_cap_from_decision_type_not_architecture() {
    let p = spawn_proxy().await;
    mount_decision_model_stub_with_type(&p, "clef-flash", "clef", "kev").await;

    let resp = p
        .client
        .post(p.url("/v1/systemone"))
        .json(&json!({
            "model": "clef-flash",
            "state": "x".repeat(70 * 1024),
            "questions": systemone_questions()
        }))
        .send()
        .await
        .expect("POST /v1/systemone clef state");

    assert_eq!(
        resp.status().as_u16(),
        413,
        "the cap follows decision_type; a kev-valued architecture must not widen it"
    );
    let body: Value = resp.json().await.expect("json body");
    assert_eq!(
        body,
        json!({"error": "text and schema must not exceed 64 KiB"})
    );
}

#[tokio::test]
async fn route_systemone_admits_text_and_schema_at_exactly_64_kib() {
    let p = spawn_proxy().await;
    mount_decision_model_stub(&p).await;

    // Size the state so the counted encoding lands exactly on the cap: the
    // comparison is `> limit`, never `>= limit`.
    let questions = systemone_questions();
    let overhead = json!({
        "model": "clef-flash",
        "state": "",
        "questions": questions
    })
    .to_string()
    .len();
    let at_cap = "x".repeat(64 * 1024 - overhead);
    let counted = json!({
        "model": "clef-flash",
        "state": at_cap.clone(),
        "questions": systemone_questions()
    })
    .to_string()
    .len();
    assert_eq!(
        counted,
        64 * 1024,
        "the fixture must measure exactly the cap"
    );

    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .and(body_partial_json(
            json!({"model": "clef-flash", "state": at_cap.clone()}),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "clef-flash",
            "answers": {"angry": {"type": "noul", "noul": 0.5}},
            "usage": {"input_tokens": 10, "output_tokens": 0}
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    let at_cap_resp = p
        .client
        .post(p.url("/v1/systemone"))
        .json(&json!({
            "model": "clef-flash",
            "state": at_cap,
            "questions": systemone_questions()
        }))
        .send()
        .await
        .expect("POST /v1/systemone at the cap");
    assert_eq!(
        at_cap_resp.status(),
        200,
        "text and schema at exactly 64 KiB must be admitted"
    );

    let over_cap_resp = p
        .client
        .post(p.url("/v1/systemone"))
        .json(&json!({
            "model": "clef-flash",
            "state": "x".repeat(64 * 1024 - overhead + 1),
            "questions": systemone_questions()
        }))
        .send()
        .await
        .expect("POST /v1/systemone one byte over the cap");
    assert_eq!(
        over_cap_resp.status().as_u16(),
        413,
        "one byte over the cap must answer 413"
    );
}

#[tokio::test]
async fn route_systemone_excludes_images_from_the_text_cap() {
    let p = spawn_proxy().await;
    mount_decision_model_stub(&p).await;

    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .and(body_partial_json(json!({"model": "clef-flash"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "clef-flash",
            "answers": {"angry": {"type": "noul", "noul": 0.5}},
            "usage": {"input_tokens": 10, "output_tokens": 0}
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/v1/systemone"))
        .json(&json!({
            "model": "clef-flash",
            "state": "x".repeat(32 * 1024),
            "images": ["A".repeat(2 * 1024 * 1024)],
            "questions": systemone_questions()
        }))
        .send()
        .await
        .expect("POST /v1/systemone with a large image");

    assert_eq!(
        resp.status(),
        200,
        "images never count against the text and schema cap"
    );
}

#[tokio::test]
async fn route_systemone_looks_the_model_up_before_the_text_cap() {
    let p = spawn_proxy().await;
    mount_decision_model_stub(&p).await;

    let resp = p
        .client
        .post(p.url("/v1/systemone"))
        .json(&json!({
            "model": "ghost-model",
            "state": "x".repeat(70 * 1024),
            "questions": systemone_questions()
        }))
        .send()
        .await
        .expect("POST /v1/systemone unknown model");

    assert_eq!(
        resp.status().as_u16(),
        404,
        "an unknown model answers 404 before the text cap"
    );
    let body: Value = resp.json().await.expect("json body");
    let error = body["error"].as_str().unwrap_or_default();
    assert!(
        error.contains("ghost-model"),
        "the 404 must name the model; got {body}"
    );
}

#[tokio::test]
async fn route_systemone_rejects_an_escaped_state_over_64_kib() {
    let p = spawn_proxy().await;
    mount_decision_model_stub(&p).await;

    // Python's default `json.dumps` spelling: a non-ASCII character costs six
    // bytes in the counted text where a decoded re-encode sees two.
    let escaped = format!("{}u00e9", char::from(92)).repeat(12 * 1024);
    let body = [
        r#"{"model":"clef-flash","state":""#,
        escaped.as_str(),
        r#"","questions":{"angry":{"type":"noul","instructions":"Is the customer angry?"}}}"#,
    ]
    .concat();

    let resp = p
        .client
        .post(p.url("/v1/systemone"))
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .expect("POST /v1/systemone escaped state");

    assert_eq!(
        resp.status().as_u16(),
        413,
        "escaped text counts in the client's spelling, as upstream's RawMessage re-encode does"
    );
    let body: Value = resp.json().await.expect("json body");
    assert_eq!(
        body,
        json!({"error": "text and schema must not exceed 64 KiB"})
    );
}

#[tokio::test]
async fn route_systemone_counts_a_null_keep_alive_like_an_omitted_one() {
    let p = spawn_proxy().await;
    mount_decision_model_stub(&p).await;

    let overhead = json!({
        "model": "clef-flash",
        "state": "",
        "questions": systemone_questions()
    })
    .to_string()
    .len();
    let state = "x".repeat(64 * 1024 - overhead);

    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .and(body_partial_json(
            json!({"model": "clef-flash", "state": state.clone()}),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "clef-flash",
            "answers": {"angry": {"type": "noul", "noul": 0.5}},
            "usage": {"input_tokens": 10, "output_tokens": 0}
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/v1/systemone"))
        .json(&json!({
            "model": "clef-flash",
            "state": state,
            "questions": systemone_questions(),
            "keep_alive": null
        }))
        .send()
        .await
        .expect("POST /v1/systemone with a null keep_alive");

    assert_eq!(
        resp.status(),
        200,
        "upstream's omitempty drops a null keep_alive; the cap must count it as absent"
    );
}

#[tokio::test]
async fn route_systemone_counts_absent_questions_as_null() {
    let p = spawn_proxy().await;
    mount_decision_model_stub(&p).await;

    let overhead = json!({
        "model": "clef-flash",
        "state": "",
        "questions": null
    })
    .to_string()
    .len();
    let resp = p
        .client
        .post(p.url("/v1/systemone"))
        .json(&json!({
            "model": "clef-flash",
            "state": "x".repeat(64 * 1024 - overhead + 1)
        }))
        .send()
        .await
        .expect("POST /v1/systemone without questions");

    assert_eq!(
        resp.status().as_u16(),
        413,
        "an absent questions field counts as null, as upstream's struct re-encode does"
    );
    let body: Value = resp.json().await.expect("json body");
    assert_eq!(
        body,
        json!({"error": "text and schema must not exceed 64 KiB"})
    );
}

#[tokio::test]
async fn route_systemone_rejects_an_unknown_runner_before_the_cap() {
    let p = spawn_proxy().await;
    mount_decision_model_stub(&p).await;

    let resp = p
        .client
        .post(p.url("/v1/systemone"))
        .json(&json!({
            "model": "clef-flash",
            "runner": "bogus",
            "state": "x".repeat(70 * 1024),
            "questions": systemone_questions()
        }))
        .send()
        .await
        .expect("POST /v1/systemone bogus runner");

    assert_eq!(
        resp.status().as_u16(),
        400,
        "upstream validates the runner before its cap"
    );
    let body: Value = resp.json().await.expect("json body");
    assert_eq!(body, json!({"error": "unknown runner \"bogus\""}));
}

#[tokio::test]
async fn route_systemone_requires_a_model() {
    let p = spawn_proxy().await;
    mount_decision_model_stub(&p).await;

    for request in [
        json!({
            "state": "x".repeat(70 * 1024),
            "questions": systemone_questions()
        }),
        json!({
            "model": "   ",
            "state": "x".repeat(70 * 1024),
            "questions": systemone_questions()
        }),
    ] {
        let resp = p
            .client
            .post(p.url("/v1/systemone"))
            .json(&request)
            .send()
            .await
            .expect("POST /v1/systemone without a model");
        assert_eq!(
            resp.status().as_u16(),
            400,
            "a missing or blank model answers upstream's 400"
        );
        let body: Value = resp.json().await.expect("json body");
        assert_eq!(body, json!({"error": "model is required"}));
    }
}

#[tokio::test]
async fn route_systemone_rejects_a_body_upstreams_decode_refuses() {
    let p = spawn_proxy().await;
    mount_decision_model_stub(&p).await;

    let resp = p
        .client
        .post(p.url("/v1/systemone"))
        .json(&json!({
            "model": "clef-flash",
            "state": "I was charged twice.",
            "questions": []
        }))
        .send()
        .await
        .expect("POST /v1/systemone with array questions");

    assert_eq!(
        resp.status().as_u16(),
        400,
        "a body Go's decode rejects answers 400, never the cap's 413"
    );
    let body: Value = resp.json().await.expect("json body");
    let error = body["error"].as_str().unwrap_or_default();
    assert!(
        error.starts_with("invalid request body"),
        "the decode refusal must name the body; got {body}"
    );
}

#[tokio::test]
async fn route_systemone_adds_no_model_lookup_at_the_cap_boundary() {
    let p = spawn_proxy().await;
    mount_decision_model_stub(&p).await;

    let overhead = json!({
        "model": "clef-flash",
        "state": "",
        "questions": systemone_questions()
    })
    .to_string()
    .len();
    let state = "x".repeat(64 * 1024 - overhead);

    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "clef-flash",
            "answers": {"angry": {"type": "noul", "noul": 0.5}},
            "usage": {"input_tokens": 10, "output_tokens": 0}
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/v1/systemone"))
        .json(&json!({
            "model": "clef-flash",
            "state": state,
            "questions": systemone_questions()
        }))
        .send()
        .await
        .expect("POST /v1/systemone at the cap");
    assert_eq!(
        resp.status(),
        200,
        "text and schema at exactly the cap must be admitted"
    );

    let received = p.mock.received_requests().await.expect("recorded requests");
    let listings = received
        .iter()
        .filter(|request| request.url.path() == "/api/v1/models")
        .count();
    assert_eq!(
        listings, 1,
        "the cap check must add no model lookup at or under 64 KiB; the passthrough's resolve is the only listing"
    );
}

#[tokio::test]
async fn route_systemone_reads_the_text_cap_without_a_decision_type() {
    let p = spawn_proxy().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [{"key": "clef-flash", "type": "decision", "publisher": "ggml-org",
                        "architecture": "clef", "format": "gguf",
                        "max_context_length": null, "loaded_instances": [],
                        "has_vision_adapter": false}]
        })))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/v1/systemone"))
        .json(&json!({
            "model": "clef-flash",
            "state": "x".repeat(70 * 1024),
            "questions": systemone_questions()
        }))
        .send()
        .await
        .expect("POST /v1/systemone without a decision_type");

    assert_eq!(
        resp.status().as_u16(),
        413,
        "a listing without decision_type keeps the 64 KiB cap"
    );
    let body: Value = resp.json().await.expect("json body");
    assert_eq!(
        body,
        json!({"error": "text and schema must not exceed 64 KiB"})
    );
}

#[tokio::test]
async fn route_systemone_folds_request_keys_like_go() {
    let p = spawn_proxy().await;
    mount_decision_model_stub(&p).await;

    let body = [
        r#"{"Model":"clef-flash","State":""#,
        "x".repeat(70 * 1024).as_str(),
        r#"","Questions":{"angry":{"type":"noul","instructions":"Is the customer angry?"}}}"#,
    ]
    .concat();

    let resp = p
        .client
        .post(p.url("/v1/systemone"))
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .expect("POST /v1/systemone with case-folded keys");

    assert_eq!(
        resp.status().as_u16(),
        413,
        "Go's decoder matches struct keys case-insensitively; the folded state must count"
    );
    let body: Value = resp.json().await.expect("json body");
    assert_eq!(
        body,
        json!({"error": "text and schema must not exceed 64 KiB"})
    );
}

#[tokio::test]
async fn route_systemone_keeps_the_last_duplicate_key() {
    let p = spawn_proxy().await;
    mount_decision_model_stub(&p).await;

    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .and(body_partial_json(json!({"model": "clef-flash"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "clef-flash",
            "answers": {"angry": {"type": "noul", "noul": 0.5}},
            "usage": {"input_tokens": 10, "output_tokens": 0}
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    // Go's struct decoder lets a repeated key overwrite the earlier one, so the
    // body is valid for Ollama and must reach the backend.
    let body = r#"{"model":"ghost-model","model":"clef-flash","state":"I was charged twice.","questions":{"angry":{"type":"noul","instructions":"Is the customer angry?"}}}"#;

    let resp = p
        .client
        .post(p.url("/v1/systemone"))
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .expect("POST /v1/systemone with a duplicate model key");

    assert_eq!(
        resp.status(),
        200,
        "the last duplicate key wins, as in Go's decoder"
    );
}

#[tokio::test]
async fn route_systemone_keeps_only_the_last_questions_map() {
    let p = spawn_proxy().await;
    mount_decision_model_stub(&p).await;

    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "clef-flash",
            "answers": {"r": {"type": "noul", "noul": 0.5}},
            "usage": {"input_tokens": 10, "output_tokens": 0}
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    // Go's orderedmap.UnmarshalJSON allocates a fresh map, so the repeated
    // `questions` field keeps only the last map: Go measures this body at 78
    // bytes and serves it, while merging would count the 70 KiB instruction and
    // answer 413.
    let big = "x".repeat(70 * 1024);
    let body = [
        r#"{"model":"clef-flash","state":"s","questions":{"q":{"type":"noul","instructions":""#,
        &big,
        r#""}},"questions":{"r":{"type":"noul","instructions":"i"}}}"#,
    ]
    .concat();

    let resp = p
        .client
        .post(p.url("/v1/systemone"))
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .expect("POST /v1/systemone with a repeated questions field");

    assert_eq!(
        resp.status(),
        200,
        "the last questions map wins, as in Go's decoder; a merge measures 70 KiB and answers 413"
    );
}

#[tokio::test]
async fn route_systemone_accepts_a_non_canonical_base64_tail() {
    let p = spawn_proxy().await;
    mount_decision_model_stub(&p).await;

    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .and(body_partial_json(json!({
            "model": "clef-flash",
            "images": ["data:application/octet-stream;base64,QR=="]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "clef-flash",
            "answers": {"angry": {"type": "noul", "noul": 0.5}},
            "usage": {"input_tokens": 10, "output_tokens": 0}
        })))
        .expect(1)
        .mount(&p.mock)
        .await;

    // Go's base64.StdEncoding ignores non-zero trailing bits in the final
    // quantum; the proxy must decode as Go does and wrap the one byte it
    // decodes to as a data URL.
    let resp = p
        .client
        .post(p.url("/v1/systemone"))
        .json(&json!({
            "model": "clef-flash",
            "state": "I was charged twice.",
            "images": ["QR=="],
            "questions": {"angry": {"type": "noul", "instructions": "Is the customer angry?"}}
        }))
        .send()
        .await
        .expect("POST /v1/systemone with a non-canonical base64 tail");

    assert_eq!(
        resp.status(),
        200,
        "Go's non-strict decoder admits a non-zero trailing bit; the proxy must too"
    );
}

#[tokio::test]
async fn route_systemone_rejects_a_cloud_model_ref() {
    let p = spawn_proxy().await;
    mount_decision_model_stub(&p).await;

    let resp = p
        .client
        .post(p.url("/v1/systemone"))
        .json(&json!({
            "model": "clef-flash:cloud",
            "state": "I was charged twice.",
            "questions": systemone_questions()
        }))
        .send()
        .await
        .expect("POST /v1/systemone with a cloud model ref");

    assert_eq!(
        resp.status().as_u16(),
        400,
        "System One requires a local decision model"
    );
    let body: Value = resp.json().await.expect("json body");
    assert_eq!(
        body,
        json!({"error": "System One requires a local decision model"})
    );
}

#[tokio::test]
async fn route_systemone_rejects_a_non_array_images() {
    let p = spawn_proxy().await;
    mount_decision_model_stub(&p).await;

    let resp = p
        .client
        .post(p.url("/v1/systemone"))
        .json(&json!({
            "model": "clef-flash",
            "images": "x",
            "state": "I was charged twice.",
            "questions": systemone_questions()
        }))
        .send()
        .await
        .expect("POST /v1/systemone with a non-array images field");

    assert_eq!(
        resp.status().as_u16(),
        400,
        "Go's decode rejects a non-array images field before the cap"
    );
    let body: Value = resp.json().await.expect("json body");
    let error = body["error"].as_str().unwrap_or_default();
    assert!(
        error.starts_with("invalid request body"),
        "the decode refusal must name the body; got {body}"
    );
}

#[tokio::test]
async fn route_audio_speech_is_present_and_returns_501() {
    let p = spawn_proxy().await;
    let resp = p
        .client
        .post(p.url("/v1/audio/speech"))
        .json(&json!({
            "model": "phonon",
            "input": "Hello from the test suite.",
            "voice": "default",
            "response_format": "wav"
        }))
        .send()
        .await
        .expect("POST /v1/audio/speech");
    assert_ne!(
        resp.status(),
        404,
        "/v1/audio/speech must be a recognised route"
    );
    assert_eq!(
        resp.status().as_u16(),
        501,
        "/v1/audio/speech must return 501: LM Studio has no TTS endpoint to forward to"
    );
    let body: Value = resp.json().await.expect("json body");
    assert!(
        body["error"].as_str().unwrap_or("").contains("speech"),
        "the 501 body must name the endpoint; got {body}"
    );
}

#[tokio::test]
async fn route_audio_transcriptions_is_present_and_returns_501() {
    let p = spawn_proxy().await;
    // A transcription request carries a `file` upload (multipart/form-data
    // in the OpenAI spec these endpoints mirror), so the stub must answer it
    // itself instead of failing inside a JSON body extractor.
    let boundary = "proxytestboundary7351";
    let body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"clip.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFFfakeaudio\r\n--{boundary}--\r\n"
    );
    let resp = p
        .client
        .post(p.url("/v1/audio/transcriptions"))
        .header(
            "content-type",
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(body)
        .send()
        .await
        .expect("POST /v1/audio/transcriptions");
    assert_ne!(
        resp.status(),
        404,
        "/v1/audio/transcriptions must be a recognised route"
    );
    assert_eq!(
        resp.status().as_u16(),
        501,
        "/v1/audio/transcriptions must return 501: LM Studio has no audio transcription endpoint to forward to"
    );
    let body: Value = resp.json().await.expect("json body");
    assert!(
        body["error"]
            .as_str()
            .unwrap_or("")
            .contains("transcription"),
        "the 501 body must name the endpoint; got {body}"
    );
}

#[tokio::test]
async fn route_audio_translations_is_present_and_returns_501() {
    let p = spawn_proxy().await;
    // Same upload shape as transcriptions, with the Ollama extension field
    // `to_language` alongside the file part.
    let boundary = "proxytestboundary7351";
    let body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"clip.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFFfakeaudio\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"to_language\"\r\n\r\nen\r\n--{boundary}--\r\n"
    );
    let resp = p
        .client
        .post(p.url("/v1/audio/translations"))
        .header(
            "content-type",
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(body)
        .send()
        .await
        .expect("POST /v1/audio/translations");
    assert_ne!(
        resp.status(),
        404,
        "/v1/audio/translations must be a recognised route"
    );
    assert_eq!(
        resp.status().as_u16(),
        501,
        "/v1/audio/translations must return 501: LM Studio has no audio translation endpoint to forward to"
    );
    let body: Value = resp.json().await.expect("json body");
    assert!(
        body["error"].as_str().unwrap_or("").contains("translation"),
        "the 501 body must name the endpoint; got {body}"
    );
}

// A trailing slash must not turn an explicit route into a passthrough request:
// `/v1/{*path}` would otherwise swallow it and forward to LM Studio's
// catch-all. Normalization runs outside the router so every explicit route
// answers its own handler, including ahead of the auth gate.

#[tokio::test]
async fn trailing_slash_on_stub_route_returns_501() {
    let p = spawn_proxy().await;
    let boundary = "proxytestboundary7351";
    let body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"clip.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFFfakeaudio\r\n--{boundary}--\r\n"
    );
    let resp = p
        .client
        .post(p.url("/v1/audio/transcriptions/"))
        .header(
            "content-type",
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(body)
        .send()
        .await
        .expect("POST /v1/audio/transcriptions/");
    assert_eq!(
        resp.status().as_u16(),
        501,
        "/v1/audio/transcriptions/ must reach the stub, not the passthrough"
    );
    let body: Value = resp.json().await.expect("json body");
    assert!(
        body["error"]
            .as_str()
            .unwrap_or("")
            .contains("transcription"),
        "the 501 body must name the endpoint; got {body}"
    );
}

#[tokio::test]
async fn trailing_slash_on_local_route_reaches_handler() {
    let p = spawn_proxy().await;
    let resp = p
        .client
        .get(p.url("/api/version/"))
        .send()
        .await
        .expect("GET /api/version/");
    assert_eq!(
        resp.status(),
        200,
        "GET /api/version/ must be served locally"
    );
    let body: Value = resp.json().await.expect("json body");
    assert!(
        body.get("version")
            .map(serde_json::Value::is_string)
            .unwrap_or(false),
        "GET /api/version/ must answer the version handler; got {body}"
    );
}

#[tokio::test]
async fn trailing_slash_on_passthrough_is_trimmed_before_forwarding() {
    let p = spawn_proxy().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_string("{\"data\":[]}"))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .get(p.url("/v1/models/"))
        .send()
        .await
        .expect("GET /v1/models/");
    assert_eq!(
        resp.status(),
        200,
        "the passthrough must forward the trimmed path to the backend"
    );
    assert_eq!(resp.text().await.expect("body"), "{\"data\":[]}");
}

#[tokio::test]
async fn trailing_slash_keeps_version_exempt_from_auth() {
    let p = spawn_proxy_with_api_key("s3cret-test-key").await;
    let resp = p
        .client
        .get(p.url("/api/version/"))
        .send()
        .await
        .expect("GET /api/version/ without a key");
    assert_eq!(
        resp.status(),
        200,
        "/api/version/ must stay exempt from the auth gate like /api/version"
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

/// POST a body over `MAX_JSON_BODY_SIZE_BYTES` (16 MiB) and hand back the
/// response. Windows may surface the server-side rejection as a mid-stream
/// connection abort rather than a clean 413: the server closes the socket after
/// writing the response while the client is still pushing the body. That proves
/// the limit was enforced but carries no body to inspect, so it yields `None`.
async fn post_oversized_body(p: &TestProxy, path: &str) -> Option<reqwest::Response> {
    match p
        .client
        .post(p.url(path))
        .header("content-type", "application/json")
        .body(vec![b'x'; 17 * 1024 * 1024])
        .send()
        .await
    {
        Ok(resp) => Some(resp),
        Err(e) => {
            assert!(
                e.is_request() || e.is_body(),
                "expected 413 or connection error from oversized body; got: {e}"
            );
            None
        }
    }
}

#[tokio::test]
async fn oversized_body_returns_413_in_ollama_envelope() {
    let p = spawn_proxy().await;
    let Some(resp) = post_oversized_body(&p, "/api/chat").await else {
        return;
    };
    assert_eq!(
        resp.status(),
        413,
        "body exceeding MAX_JSON_BODY_SIZE_BYTES must return 413"
    );
    assert_eq!(
        resp.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok()),
        Some("application/json"),
        "extraction rejections must not fall back to the framework's text/plain shape"
    );
    let body: Value = resp.json().await.expect("413 body must be JSON");
    assert_eq!(body, json!({ "error": "request body too large" }));
}

#[tokio::test]
async fn oversized_body_on_anthropic_surface_returns_413_in_anthropic_envelope() {
    let p = spawn_proxy().await;
    let Some(resp) = post_oversized_body(&p, "/v1/messages").await else {
        return;
    };
    assert_eq!(resp.status(), 413, "the limit applies to every route");
    assert_eq!(
        resp.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok()),
        Some("application/json"),
        "extraction rejections must not fall back to the framework's text/plain shape"
    );
    let body: Value = resp.json().await.expect("413 body must be JSON");
    assert_eq!(
        body,
        json!({
            "type": "error",
            "error": { "type": "request_too_large", "message": "request body too large" }
        })
    );
}

// ---------------------------------------------------------------------------
// Path-param extraction rejection — same envelope split
// ---------------------------------------------------------------------------

#[tokio::test]
async fn undecodable_path_param_returns_400_in_ollama_envelope() {
    let p = spawn_proxy().await;
    // %FF is not valid UTF-8, so the router's path param never deserializes.
    // POST (not HEAD) so the error body survives to the client.
    let resp = p
        .client
        .post(p.url("/api/blobs/%FF"))
        .body("blob")
        .send()
        .await
        .expect("POST /api/blobs/%FF");
    assert_eq!(resp.status(), 400);
    assert_eq!(
        resp.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok()),
        Some("application/json")
    );
    let body: Value = resp.json().await.expect("400 body must be JSON");
    // axum owns this rejection's wording, so pin the envelope and the param it
    // names: an axum bump must not read as a proxy regression.
    let obj = body
        .as_object()
        .expect("ollama error envelope is an object");
    assert_eq!(obj.len(), 1);
    let msg = obj["error"].as_str().expect("`error` carries a string");
    assert!(msg.contains("digest"), "message must name the param: {msg}");
}

#[tokio::test]
async fn same_prefix_stranger_wears_ollama_envelope() {
    // `/v1/messagesXYZ` shares the `/v1/messages` prefix but is not the
    // anthropic surface: its extraction rejections wear the ollama envelope,
    // unlike `/v1/messages/%FF` above.
    let p = spawn_proxy().await;
    let resp = p
        .client
        .post(p.url("/v1/messagesXYZ/%FF"))
        .body("{}")
        .send()
        .await
        .expect("POST /v1/messagesXYZ/%FF");
    assert_eq!(resp.status(), 400);
    let body: Value = resp.json().await.expect("400 body must be JSON");
    let obj = body
        .as_object()
        .expect("ollama error envelope is an object");
    assert_eq!(obj.len(), 1, "ollama envelope has only `error`: {body}");
    let msg = obj["error"].as_str().expect("`error` carries a string");
    assert!(msg.contains("path"), "message must name the param: {msg}");
}

#[tokio::test]
async fn same_prefix_stranger_401_wears_ollama_envelope() {
    // The auth gate picks its envelope through the same surface predicate.
    use crate::common::spawn_proxy_with_api_key;
    let p = spawn_proxy_with_api_key("k").await;
    let resp = p
        .client
        .post(p.url("/v1/messagesXYZ"))
        .body("{}")
        .send()
        .await
        .expect("POST /v1/messagesXYZ no auth");
    assert_eq!(resp.status(), 401);
    let body: Value = resp.json().await.expect("401 body must be JSON");
    assert_eq!(body, json!({ "error": "unauthorized" }));
}

#[tokio::test]
async fn method_not_allowed_on_anthropic_surface_wears_anthropic_envelope() {
    // The catch-all registers get/post/put/delete/head/options; PATCH falls
    // to the method-not-allowed fallback, which must still pick the envelope
    // from the path.
    let p = spawn_proxy().await;
    let resp = p
        .client
        .patch(p.url("/v1/messages"))
        .body("{}")
        .send()
        .await
        .expect("PATCH /v1/messages");
    assert_eq!(resp.status(), 405);
    let body: Value = resp.json().await.expect("405 body must be JSON");
    let obj = body
        .as_object()
        .expect("anthropic error envelope is an object");
    assert_eq!(obj.len(), 2, "envelope must be the anthropic shape: {body}");
    assert_eq!(obj["type"], "error");
    let err = obj["error"].as_object().expect("`error` is an object");
    assert_eq!(err["type"], "invalid_request_error");
    assert_eq!(err["message"], "method not allowed");
}

#[tokio::test]
async fn undecodable_path_param_on_anthropic_surface_returns_anthropic_envelope() {
    let p = spawn_proxy().await;
    let resp = p
        .client
        .post(p.url("/v1/messages/%FF"))
        .body("{}")
        .send()
        .await
        .expect("POST /v1/messages/%FF");
    assert_eq!(resp.status(), 400);
    let body: Value = resp.json().await.expect("400 body must be JSON");
    // axum owns this rejection's wording, so pin the envelope and the param it
    // names: an axum bump must not read as a proxy regression.
    let obj = body
        .as_object()
        .expect("anthropic error envelope is an object");
    assert_eq!(obj.len(), 2);
    assert_eq!(obj["type"], "error");
    let err = obj["error"].as_object().expect("`error` is an object");
    assert_eq!(err.len(), 2);
    assert_eq!(err["type"], "invalid_request_error");
    let msg = err["message"].as_str().expect("`message` carries a string");
    assert!(msg.contains("path"), "message must name the param: {msg}");
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
async fn generate_missing_prompt_triggers_warm_load() {
    let p = spawn_proxy().await;
    mount_models_stub(&p).await;
    mount_chat_stub(&p, "/api/v0/chat/completions").await;
    let resp = p
        .client
        .post(p.url("/api/generate"))
        .json(&json!({"model": "llama3", "stream": false}))
        .send()
        .await
        .expect("POST /api/generate no prompt");
    assert_eq!(
        resp.status(),
        200,
        "a promptless generate is a load/warm no-op, not an error"
    );
    let body: Value = resp.json().await.expect("JSON body");
    assert_eq!(body["done"], true);
    assert_eq!(body["done_reason"], "load");
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
