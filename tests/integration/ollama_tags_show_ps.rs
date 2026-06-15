// Integration tests for GET /api/tags, POST /api/show, GET /api/ps,
// GET /api/version, and HEAD /api/version.
//
// The proxy calls GET /api/v1/models on the LM Studio backend for all of
// these — the wiremock mock must be registered on that path.

use ollama_lmstudio_proxy::constants::OLLAMA_SERVER_VERSION;
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
        // LM Studio surfaces no per-model mtime; the proxy must omit
        // modified_at rather than fabricate one.
        assert!(
            m.get("modified_at").is_none(),
            "tags entry must omit modified_at when no real mtime is available; got {m}"
        );
        assert!(m["size"].is_number(), "missing size in {m}");
        assert_eq!(
            m["context_length"],
            json!(8192),
            "unexpected context in {m}"
        );
        assert_eq!(
            m["max_context_length"],
            json!(8192),
            "unexpected max context in {m}"
        );
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
async fn tags_response_omits_modified_at_when_no_mtime_available() {
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
    // LM Studio's model list exposes no per-model mtime. The proxy must omit
    // `modified_at` rather than fabricate Utc::now() or the epoch.
    for m in body["models"].as_array().expect("models array") {
        assert!(
            m.get("modified_at").is_none(),
            "tags entry must omit modified_at when no real mtime is available; got {m}"
        );
    }
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
    // parameters and template are only emitted when a virtual alias supplies
    // them; LM Studio has no Modelfile so a native-backed show must omit them.
    assert!(
        body.get("parameters").is_none(),
        "native show must omit parameters; {body}"
    );
    assert!(
        body.get("template").is_none(),
        "native show must omit template; {body}"
    );
    assert!(body["details"].is_object(), "missing details; {body}");
    // model_info is always present so Ollama clients can read context_length
    assert!(
        body["model_info"].is_object(),
        "model_info must always appear; {body}"
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
    // /api/show describes the stable model — architecture-scoped context length
    // mirrors max_context_length regardless of any loaded runtime instance.
    let model_info = &body["model_info"];
    assert_eq!(model_info["llama.context_length"], json!(8192));
    // lmstudio.* fields appear under verbose:true and surface runtime details:
    // the currently-loaded instance reports 4096, while the model max is 8192.
    assert_eq!(model_info["lmstudio.context_length"], json!(4096));
    assert_eq!(model_info["lmstudio.max_context_length"], json!(8192));
    // details.context_length is the stable model value.
    assert_eq!(body["details"]["context_length"], json!(8192));
    assert_eq!(body["details"]["max_context_length"], json!(8192));
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
        .and(path("/api/v0/chat/completions"))
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
        .any(|r| r.url.path() == "/api/v0/chat/completions");
    assert!(
        load_triggered,
        "keep_alive in show body must not suppress model loading; \
         no POST to /api/v0/chat/completions was observed"
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
async fn ps_loaded_model_has_expires_at_and_size_vram_mirrors_size() {
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
    // size_vram mirrors the loaded size (LM Studio gives no GPU/CPU split).
    assert_eq!(
        m["size_vram"], m["size"],
        "size_vram should mirror size; {m}"
    );
    assert_eq!(m["context_length"], json!(4096), "unexpected context; {m}");
    // Real Ollama /api/ps emits an empty details.parent_model.
    assert_eq!(
        m["details"]["parent_model"],
        json!(""),
        "ps details.parent_model must be empty string; {m}"
    );
}

#[tokio::test]
async fn ps_loaded_model_size_vram_mirrors_size() {
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
    // A loaded model is assumed resident → size_vram mirrors size.
    assert_eq!(
        m["size_vram"],
        json!(4_500_000_000u64),
        "size_vram should mirror size; {m}"
    );
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
    // No runtime instance is loaded; both the architecture-scoped context length
    // and LM Studio's runtime mirror the model's max.
    let model_info = &body["model_info"];
    assert_eq!(model_info["llama.context_length"], json!(8192));
    assert_eq!(model_info["lmstudio.context_length"], json!(8192));
    assert_eq!(model_info["lmstudio.max_context_length"], json!(8192));
    assert_eq!(body["details"]["context_length"], json!(8192));
    assert_eq!(body["details"]["max_context_length"], json!(8192));
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

#[tokio::test]
async fn ps_loaded_model_details_have_empty_parent_model() {
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
    let details = &body["models"][0]["details"];
    // Real Ollama /api/ps emits details.parent_model = "" (kept out of /api/tags).
    assert_eq!(
        details["parent_model"],
        json!(""),
        "ps details.parent_model must be empty string; got {details}"
    );
}

#[tokio::test]
async fn show_verbose_surfaces_loaded_tuning_from_instance_config() {
    let p = spawn_proxy().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lms_models(vec![
            native_model_with_config(
                "llama3.2:3b",
                "llama",
                true,
                json!({
                    "context_length": 8192,
                    "flash_attention": true,
                    "eval_batch_size": 512,
                    "parallel": 4
                }),
            ),
        ])))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/show"))
        .json(&json!({"model": "llama3.2:3b", "verbose": true}))
        .send()
        .await
        .expect("POST /api/show");
    assert_eq!(resp.status(), 200);

    let body: Value = resp.json().await.expect("json body");
    let model_info = &body["model_info"];
    // Real per-instance tuning surfaces only under verbose.
    assert_eq!(model_info["lmstudio.flash_attention"], json!(true));
    assert_eq!(model_info["lmstudio.eval_batch_size"], json!(512));
    assert_eq!(model_info["lmstudio.parallel"], json!(4));
    assert_eq!(model_info["lmstudio.context_length"], json!(8192));
}

#[tokio::test]
async fn show_unloaded_model_omits_loaded_tuning() {
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
        .json(&json!({"model": "llama3.2:3b", "verbose": true}))
        .send()
        .await
        .expect("POST /api/show");
    assert_eq!(resp.status(), 200);

    let body: Value = resp.json().await.expect("json body");
    let model_info = &body["model_info"];
    // No loaded instance → tuning fields absent (never fabricated).
    assert!(model_info.get("lmstudio.flash_attention").is_none());
    assert!(model_info.get("lmstudio.eval_batch_size").is_none());
    assert!(model_info.get("lmstudio.parallel").is_none());
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

// T5 — verbose: true must be a strict superset of verbose: false.
#[tokio::test]
async fn show_verbose_true_is_strict_superset_of_verbose_false() {
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

    let concise: Value = p
        .client
        .post(p.url("/api/show"))
        .json(&json!({"model": "llama3.2:3b"}))
        .send()
        .await
        .expect("POST /api/show verbose=false")
        .json()
        .await
        .expect("json body");

    let verbose: Value = p
        .client
        .post(p.url("/api/show"))
        .json(&json!({"model": "llama3.2:3b", "verbose": true}))
        .send()
        .await
        .expect("POST /api/show verbose=true")
        .json()
        .await
        .expect("json body");

    let concise_obj = concise.as_object().expect("concise must be object");
    let verbose_obj = verbose.as_object().expect("verbose must be object");

    for (key, value) in concise_obj {
        // model_info: verbose must be a superset of concise.
        if key == "model_info" {
            let concise_mi = value.as_object().expect("concise model_info");
            let verbose_mi = verbose_obj
                .get("model_info")
                .and_then(|v| v.as_object())
                .expect("verbose model_info");
            for (mk, mv) in concise_mi {
                assert_eq!(
                    verbose_mi.get(mk),
                    Some(mv),
                    "model_info.{mk}: verbose value must equal concise value"
                );
            }
            assert!(
                verbose_mi.len() >= concise_mi.len(),
                "verbose model_info must be >= concise; concise={concise_mi:?} verbose={verbose_mi:?}"
            );
            continue;
        }
        let other = verbose_obj.get(key);
        assert_eq!(
            other,
            Some(value),
            "verbose response missing or differs on top-level key '{key}'"
        );
    }
}

// T5 — without a virtual alias, /api/show must NOT emit fabricated parameters
// or template strings.
#[tokio::test]
async fn show_response_omits_fabricated_parameters_and_template() {
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

    let body: Value = p
        .client
        .post(p.url("/api/show"))
        .json(&json!({"model": "llama3.2:3b"}))
        .send()
        .await
        .expect("POST /api/show")
        .json()
        .await
        .expect("json body");

    assert!(
        body.get("parameters").is_none(),
        "non-alias /api/show must omit `parameters`; got {body}"
    );
    assert!(
        body.get("template").is_none(),
        "non-alias /api/show must omit `template`; got {body}"
    );
}

// T14 — virtual aliases carry a real updated_at and must surface it as
// modified_at on /api/show.
#[tokio::test]
async fn show_response_emits_alias_modified_at_when_provided() {
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
        .json(&json!({"source": "llama3.2:3b", "destination": "alias-mtime:v1"}))
        .send()
        .await
        .expect("POST /api/copy");
    assert!(
        !copy.status().is_server_error(),
        "copy failed: {}",
        copy.status()
    );

    let body: Value = p
        .client
        .post(p.url("/api/show"))
        .json(&json!({"model": "alias-mtime:v1"}))
        .send()
        .await
        .expect("POST /api/show virtual")
        .json()
        .await
        .expect("json body");

    let modified_at = body
        .get("modified_at")
        .and_then(|v| v.as_str())
        .unwrap_or_else(|| {
            panic!("virtual alias show response must surface modified_at; got {body}")
        });
    chrono::DateTime::parse_from_rfc3339(modified_at)
        .unwrap_or_else(|_| panic!("modified_at must be RFC3339; got {modified_at}"));
}

// T14 — /api/show must not fabricate modified_at when LM Studio has no mtime.
#[tokio::test]
async fn show_response_omits_modified_at_when_no_mtime_available() {
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

    let body: Value = p
        .client
        .post(p.url("/api/show"))
        .json(&json!({"model": "llama3.2:3b"}))
        .send()
        .await
        .expect("POST /api/show")
        .json()
        .await
        .expect("json body");

    assert!(
        body.get("modified_at").is_none(),
        "show response for a native model must omit modified_at; got {body}"
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
    assert_eq!(
        version, OLLAMA_SERVER_VERSION,
        "version mismatch; expected '{OLLAMA_SERVER_VERSION}', got '{version}'"
    );
}
