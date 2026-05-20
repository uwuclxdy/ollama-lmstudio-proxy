// Integration tests for GET /api/tags, POST /api/show, GET /api/ps,
// GET /api/version, and HEAD /api/version.
//
// The proxy calls GET /api/v1/models on the LM Studio backend for all of
// these — the wiremock mock must be registered on that path.

use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, ResponseTemplate};

use crate::common::spawn_proxy;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Minimal NativeModelData that the proxy's NativeModelsResponse can parse.
fn native_model(key: &str, architecture: &str, loaded: bool) -> Value {
    native_model_with_config(key, architecture, loaded, json!({"context_length": 4096}))
}

fn native_model_with_config(key: &str, architecture: &str, loaded: bool, config: Value) -> Value {
    let loaded_instances = if loaded {
        json!([{"id": format!("{}-inst", key), "config": config}])
    } else {
        json!([])
    };
    json!({
        "key": key,
        "type": "llm",
        "publisher": "test-publisher",
        "architecture": architecture,
        "format": "gguf",
        "quantization": {"name": "Q4_K_M"},
        "max_context_length": 8192,
        "loaded_instances": loaded_instances,
        "params_string": "7B",
        "size_bytes": 4_500_000_000u64
    })
}

fn lms_models(models: Vec<Value>) -> Value {
    json!({ "models": models })
}

// ---------------------------------------------------------------------------
// GET /api/tags
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tags_empty_model_list() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lms_models(vec![])))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .get(p.url("/api/tags"))
        .send()
        .await
        .expect("GET /api/tags");
    assert_eq!(resp.status(), 200);

    let body: Value = resp.json().await.expect("json body");
    let models = body["models"].as_array().expect("models array");
    assert!(models.is_empty(), "expected empty list, got {body}");
}

#[tokio::test]
async fn tags_multiple_models_ollama_shape() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lms_models(vec![
            native_model("llama3.2:3b", "llama", false),
            native_model("mistral:7b", "mistral", false),
        ])))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .get(p.url("/api/tags"))
        .send()
        .await
        .expect("GET /api/tags");
    assert_eq!(resp.status(), 200);

    let body: Value = resp.json().await.expect("json body");
    let models = body["models"].as_array().expect("models array");
    assert_eq!(models.len(), 2, "expected 2 models; got {body}");

    for m in models {
        assert!(m["name"].is_string(), "missing name in {m}");
        assert!(m["model"].is_string(), "missing model in {m}");
        assert!(m["modified_at"].is_string(), "missing modified_at in {m}");
        assert!(m["size"].is_number(), "missing size in {m}");
        let digest = m["digest"].as_str().expect("digest must be a string");
        assert_eq!(digest.len(), 64, "digest must be 64-char hex in {m}");
        assert!(
            digest.bytes().all(|b| b.is_ascii_hexdigit()),
            "digest must be lowercase hex, got {digest:?}"
        );

        let d = &m["details"];
        assert!(d["family"].is_string(), "missing details.family in {m}");
        assert!(
            d["parameter_size"].is_string(),
            "missing details.parameter_size in {m}"
        );
        assert!(
            d["quantization_level"].is_string(),
            "missing details.quantization_level in {m}"
        );
    }
}

#[tokio::test]
async fn tags_model_without_tag_gets_colon_suffix() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(lms_models(vec![native_model("qwen2.5", "qwen2", false)])),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .get(p.url("/api/tags"))
        .send()
        .await
        .expect("GET /api/tags");
    assert_eq!(resp.status(), 200);

    let body: Value = resp.json().await.expect("json body");
    let models = body["models"].as_array().expect("models array");
    let name = models[0]["name"].as_str().expect("name");
    // Key without a colon should gain a :latest suffix.
    assert!(
        name.contains(':'),
        "model without tag should be normalised to 'name:tag', got '{name}'"
    );
}

#[tokio::test]
async fn tags_model_details_omit_parent_model() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lms_models(vec![native_model(
                "llama3.2:3b",
                "llama",
                false,
            )])),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .get(p.url("/api/tags"))
        .send()
        .await
        .expect("GET /api/tags");
    assert_eq!(resp.status(), 200);

    let body: Value = resp.json().await.expect("json body");
    let details = &body["models"][0]["details"];
    assert!(
        details.get("parent_model").is_none(),
        "tags details must not include parent_model; got {details}"
    );
}

#[tokio::test]
async fn tags_model_modified_at_uses_deterministic_fallback() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lms_models(vec![native_model(
                "llama3.2:3b",
                "llama",
                false,
            )])),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .get(p.url("/api/tags"))
        .send()
        .await
        .expect("GET /api/tags");
    assert_eq!(resp.status(), 200);

    let body: Value = resp.json().await.expect("json body");
    assert_eq!(
        body["models"][0]["modified_at"].as_str(),
        Some("1970-01-01T00:00:00Z"),
        "LM Studio model list has no last-modified field, so tags must use a stable fallback; got {body}"
    );
}

#[tokio::test]
async fn tags_virtual_model_created_via_copy_appears_in_list() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lms_models(vec![native_model(
                "llama3.2:3b",
                "llama",
                false,
            )])),
        )
        .mount(&p.mock)
        .await;

    let copy = p
        .client
        .post(p.url("/api/copy"))
        .json(&json!({"source": "llama3.2:3b", "destination": "my-alias:latest"}))
        .send()
        .await
        .expect("POST /api/copy");
    // Accept any non-5xx; the important part is the alias is stored.
    assert!(
        !copy.status().is_server_error(),
        "copy failed: {}",
        copy.status()
    );

    let resp = p
        .client
        .get(p.url("/api/tags"))
        .send()
        .await
        .expect("GET /api/tags after copy");
    assert_eq!(resp.status(), 200);

    let body: Value = resp.json().await.expect("json body");
    let models = body["models"].as_array().expect("models array");
    let has_alias = models.iter().any(|m| {
        m["name"].as_str().is_some_and(|n| n.contains("my-alias"))
            || m["model"].as_str().is_some_and(|n| n.contains("my-alias"))
    });
    assert!(
        has_alias,
        "expected 'my-alias' in /api/tags list; got {body}"
    );
}

#[tokio::test]
async fn tags_backend_5xx_returns_error() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .get(p.url("/api/tags"))
        .send()
        .await
        .expect("GET /api/tags");
    assert!(
        resp.status().is_server_error() || resp.status().is_client_error(),
        "expected error status for backend 5xx; got {}",
        resp.status()
    );
}

// ---------------------------------------------------------------------------
// POST /api/show
// ---------------------------------------------------------------------------

#[tokio::test]
async fn show_present_model_returns_full_shape() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lms_models(vec![native_model(
                "llama3.2:3b",
                "llama",
                false,
            )])),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/show"))
        .json(&json!({"model": "llama3.2:3b"}))
        .send()
        .await
        .expect("POST /api/show");
    assert_eq!(resp.status(), 200);

    let body: Value = resp.json().await.expect("json body");
    // modelfile is not in the ShowResponse schema — must be absent
    assert!(
        body.get("modelfile").is_none(),
        "modelfile must not appear; {body}"
    );
    assert!(body["parameters"].is_string(), "missing parameters; {body}");
    assert!(body["template"].is_string(), "missing template; {body}");
    assert!(body["details"].is_object(), "missing details; {body}");
    // model_info is verbose-only — absent in non-verbose response
    assert!(
        body.get("model_info").is_none(),
        "model_info must be absent in non-verbose; {body}"
    );
    assert!(
        body["capabilities"].is_array(),
        "missing capabilities; {body}"
    );
}

#[tokio::test]
async fn show_loaded_model_uses_configured_context() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lms_models(vec![native_model(
                "llama3.2:3b",
                "llama",
                true,
            )])),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/show"))
        .json(&json!({ "model": "llama3.2:3b", "verbose": true }))
        .send()
        .await
        .expect("POST /api/show");
    assert_eq!(resp.status(), 200);

    let body: Value = resp.json().await.expect("json body");
    let params = body["parameters"].as_str().expect("parameters string");
    assert!(
        params.contains("num_ctx 4096"),
        "unexpected parameters: {params}"
    );
    let model_info = &body["model_info"];
    assert_eq!(model_info["llama.context_length"], json!(4096));
    assert_eq!(model_info["lmstudio.context_length"], json!(4096));
    assert_eq!(model_info["lmstudio.max_context_length"], json!(8192));
}

#[tokio::test]
async fn show_model_info_contains_parameter_count() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lms_models(vec![native_model(
                "llama3:8b",
                "llama",
                false,
            )])),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/show"))
        .json(&json!({"model": "llama3:8b", "verbose": true}))
        .send()
        .await
        .expect("POST /api/show");
    assert_eq!(resp.status(), 200);

    let body: Value = resp.json().await.expect("json body");
    let model_info = &body["model_info"];
    assert!(
        model_info.is_object(),
        "model_info must be an object; {body}"
    );

    let count = &model_info["general.parameter_count"];
    assert!(
        count.is_number(),
        "expected general.parameter_count to be a number; got {model_info}"
    );
    assert!(
        count.as_u64().unwrap_or(0) > 0,
        "general.parameter_count must be > 0; got {count}"
    );
}

#[tokio::test]
async fn show_model_info_contains_architecture() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lms_models(vec![native_model(
                "mistral:7b",
                "mistral",
                false,
            )])),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/show"))
        .json(&json!({"model": "mistral:7b", "verbose": true}))
        .send()
        .await
        .expect("POST /api/show");
    assert_eq!(resp.status(), 200);

    let body: Value = resp.json().await.expect("json body");
    let arch = body["model_info"]["general.architecture"]
        .as_str()
        .expect("architecture");
    assert_eq!(
        arch, "mistral",
        "expected architecture 'mistral'; got '{arch}'"
    );
}

#[tokio::test]
async fn show_missing_model_returns_error_indication() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lms_models(vec![])))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/show"))
        .json(&json!({"model": "no-such-model:latest"}))
        .send()
        .await
        .expect("POST /api/show missing model");

    let status = resp.status();
    let body: Value = resp.json().await.expect("json body");
    let is_err =
        status.is_client_error() || status.is_server_error() || body.get("error").is_some();
    assert!(
        is_err,
        "expected error for missing model; got status={status} body={body}"
    );
}

#[tokio::test]
async fn show_missing_model_field_is_client_error() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lms_models(vec![])))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/show"))
        .json(&json!({}))
        .send()
        .await
        .expect("POST /api/show empty body");

    assert!(
        resp.status().is_client_error(),
        "expected 4xx for missing model field; got {}",
        resp.status()
    );
}

#[tokio::test]
async fn show_virtual_model_includes_alias_metadata() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lms_models(vec![native_model(
                "llama3.2:3b",
                "llama",
                false,
            )])),
        )
        .mount(&p.mock)
        .await;

    // Create a virtual alias via copy.
    let copy = p
        .client
        .post(p.url("/api/copy"))
        .json(&json!({"source": "llama3.2:3b", "destination": "show-alias:v1"}))
        .send()
        .await
        .expect("POST /api/copy");
    assert!(
        !copy.status().is_server_error(),
        "copy failed: {}",
        copy.status()
    );

    let resp = p
        .client
        .post(p.url("/api/show"))
        .json(&json!({"model": "show-alias:v1"}))
        .send()
        .await
        .expect("POST /api/show virtual");
    assert_eq!(resp.status(), 200);

    let body: Value = resp.json().await.expect("json body");
    // Virtual models include extra metadata fields.
    assert!(
        body.get("virtual").is_some()
            || body.get("alias_name").is_some()
            || body.get("source_model").is_some(),
        "expected virtual model metadata in show response; got {body}"
    );
}

// Drift A: keep_alive must be ignored — the proxy must always attempt model
// loading regardless of what keep_alive the caller sends.
#[tokio::test]
async fn show_keep_alive_in_body_does_not_skip_model_load() {
    let p = spawn_proxy().await;

    // Stand in for LM Studio's model list.
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lms_models(vec![native_model(
                "llama3.2:3b",
                "llama",
                false,
            )])),
        )
        .mount(&p.mock)
        .await;

    // Stand in for the model-loading trigger the proxy sends to LM Studio.
    // A 200 with a minimal body is enough; the proxy doesn't inspect it.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]
        })))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/show"))
        .json(&json!({"model": "llama3.2:3b", "keep_alive": 0}))
        .send()
        .await
        .expect("POST /api/show with keep_alive");
    assert_eq!(resp.status(), 200);

    // The proxy must have forwarded a model-loading trigger to LM Studio.
    // keep_alive in a ShowRequest has no meaning and must not suppress this call.
    let received = p.mock.received_requests().await.unwrap_or_default();
    let load_triggered = received
        .iter()
        .any(|r| r.url.path() == "/v1/chat/completions");
    assert!(
        load_triggered,
        "keep_alive in show body must not suppress model loading; \
         no POST to /v1/chat/completions was observed"
    );
}

// Drift B: ShowResponse schema defines no `digest` or `size` fields.
// The proxy must not emit them for non-virtual models.
#[tokio::test]
async fn show_response_omits_digest_and_size() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lms_models(vec![native_model(
                "llama3.2:3b",
                "llama",
                false,
            )])),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/show"))
        .json(&json!({"model": "llama3.2:3b"}))
        .send()
        .await
        .expect("POST /api/show");
    assert_eq!(resp.status(), 200);

    let body: Value = resp.json().await.expect("json body");
    assert!(
        body.get("digest").is_none(),
        "ShowResponse must not include `digest`; got {body}"
    );
    assert!(
        body.get("size").is_none(),
        "ShowResponse must not include `size`; got {body}"
    );
}

// ---------------------------------------------------------------------------
// GET /api/ps
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ps_no_loaded_models_returns_empty() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lms_models(vec![native_model(
                "llama3.2:3b",
                "llama",
                false,
            )])),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .get(p.url("/api/ps"))
        .send()
        .await
        .expect("GET /api/ps");
    assert_eq!(resp.status(), 200);

    let body: Value = resp.json().await.expect("json body");
    let models = body["models"].as_array().expect("models array");
    assert!(models.is_empty(), "expected empty ps list; got {body}");
}

#[tokio::test]
async fn ps_loaded_model_has_expires_at_and_zero_size_vram_without_gpu_signal() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lms_models(vec![native_model(
                "llama3.2:3b",
                "llama",
                true,
            )])),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .get(p.url("/api/ps"))
        .send()
        .await
        .expect("GET /api/ps");
    assert_eq!(resp.status(), 200);

    let body: Value = resp.json().await.expect("json body");
    let models = body["models"].as_array().expect("models array");
    assert_eq!(models.len(), 1, "expected 1 loaded model; got {body}");

    let m = &models[0];
    assert!(m["name"].is_string(), "missing name; {m}");
    // Per docs/lmstudio_vs_ollama.md §"Running models", /api/ps entries
    // expose both `name` (display) and `model` (canonical identifier).
    assert!(m["model"].is_string(), "missing model; {m}");
    assert!(m["expires_at"].is_string(), "missing expires_at; {m}");
    assert_eq!(m["size_vram"], json!(0), "unexpected VRAM usage; {m}");
    assert_eq!(m["context_length"], json!(4096), "unexpected context; {m}");
}

#[tokio::test]
async fn ps_loaded_model_with_kv_cache_gpu_flag_still_reports_zero_size_vram() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lms_models(vec![
            native_model_with_config(
                "llama3.2:3b",
                "llama",
                true,
                json!({
                    "context_length": 4096,
                    "offload_kv_cache_to_gpu": true
                }),
            ),
        ])))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .get(p.url("/api/ps"))
        .send()
        .await
        .expect("GET /api/ps");
    assert_eq!(resp.status(), 200);

    let body: Value = resp.json().await.expect("json body");
    let models = body["models"].as_array().expect("models array");
    assert_eq!(models.len(), 1, "expected 1 loaded model; got {body}");

    let m = &models[0];
    assert_eq!(m["size"], json!(4_500_000_000u64));
    assert_eq!(m["size_vram"], json!(0), "unexpected VRAM usage; {m}");
    assert_eq!(m["context_length"], json!(4096), "unexpected context; {m}");
}

#[tokio::test]
async fn show_unloaded_model_falls_back_to_max_context() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(lms_models(vec![native_model(
                "llama3.2:3b",
                "llama",
                false,
            )])),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/show"))
        .json(&json!({ "model": "llama3.2:3b", "verbose": true }))
        .send()
        .await
        .expect("POST /api/show");
    assert_eq!(resp.status(), 200);

    let body: Value = resp.json().await.expect("json body");
    let params = body["parameters"].as_str().expect("parameters string");
    assert!(
        params.contains("num_ctx 8192"),
        "unexpected parameters: {params}"
    );
    let model_info = &body["model_info"];
    assert_eq!(model_info["llama.context_length"], json!(8192));
    assert_eq!(model_info["lmstudio.context_length"], json!(8192));
    assert_eq!(model_info["lmstudio.max_context_length"], json!(8192));
}

#[tokio::test]
async fn ps_only_loaded_models_in_response() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lms_models(vec![
            native_model("llama3.2:3b", "llama", true),
            native_model("mistral:7b", "mistral", false),
            native_model("qwen2.5:14b", "qwen2", true),
        ])))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .get(p.url("/api/ps"))
        .send()
        .await
        .expect("GET /api/ps");
    assert_eq!(resp.status(), 200);

    let body: Value = resp.json().await.expect("json body");
    let models = body["models"].as_array().expect("models array");
    assert_eq!(
        models.len(),
        2,
        "expected exactly 2 loaded models; got {body}"
    );
}

// ---------------------------------------------------------------------------
// GET /api/version
// ---------------------------------------------------------------------------

#[tokio::test]
async fn version_returns_version_string() {
    let p = spawn_proxy().await;

    let resp = p
        .client
        .get(p.url("/api/version"))
        .send()
        .await
        .expect("GET /api/version");
    assert_eq!(resp.status(), 200);

    let body: Value = resp.json().await.expect("json body");
    let version = body["version"].as_str().expect("version field");
    // Must look like semver (e.g. "0.13.0").
    assert!(
        version.contains('.'),
        "expected semver version string; got '{version}'"
    );
}

#[tokio::test]
async fn version_matches_expected_constant() {
    let p = spawn_proxy().await;

    let resp = p
        .client
        .get(p.url("/api/version"))
        .send()
        .await
        .expect("GET /api/version");
    let body: Value = resp.json().await.expect("json body");
    let version = body["version"].as_str().expect("version field");
    // OLLAMA_SERVER_VERSION in constants.rs is "0.13.0".
    assert_eq!(
        version, "0.13.0",
        "version mismatch; expected '0.13.0', got '{version}'"
    );
}
