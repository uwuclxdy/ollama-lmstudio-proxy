#![allow(clippy::unwrap_used, clippy::expect_used)]
// A malformed (non-JSON) 200 body is an ANSWERED response: no model load can
// fix it, so a decode failure must be returned verbatim without a load detour.
//
// The retry layer classifies by message text only. A decode error embeds the
// upstream url ("invalid JSON from LM Studio: error decoding response body for
// url (http://host/model-proxy/v1/...)"), so a user-configured base url whose
// path carries a load-word ("model") plus the "invalid" in the proxy's own
// prefix satisfies both halves of the loose two-word classifier and used to
// read as "model loading" — burning model resolution, a POST to
// /api/v1/models/load and the whole load timeout on a body LM Studio already
// delivered.
//
// The fix: the decode site stamps a typed marker on the error and the retry
// layer returns it before any message classification.
//
// Red on the pre-change tree: the detour fires POST /api/v1/models/load
// (expect(0) violated at verify()) and adds the load-timeout wait to the
// response time.

use std::time::{Duration, Instant};

use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, ResponseTemplate};

use crate::common::spawn_proxy_with_lmstudio_prefix;

const OLLAMA_MODEL: &str = "some-model:latest";
const MODEL_KEY: &str = "some-model";
// Base-url path prefix whose words feed the message classifier.
const BASE_PREFIX: &str = "/model-proxy/v1";

#[tokio::test]
async fn decode_error_returns_verbatim_without_load_detour() {
    let p = spawn_proxy_with_lmstudio_prefix(BASE_PREFIX).await;

    // Model resolution may legitimately look the model up; a load detour may
    // not fire the explicit load.
    Mock::given(method("GET"))
        .and(path("/model-proxy/v1/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [{
                "key": MODEL_KEY,
                "type": "llm",
                "publisher": "test",
                "architecture": "llama",
                "format": "gguf",
                "quantization": { "name": "Q4_K_M", "bits_per_weight": 4.5 },
                "max_context_length": 4096,
                "loaded_instances": [],
                "capabilities": { "vision": false, "trained_for_tool_use": false }
            }]
        })))
        .mount(&p.mock)
        .await;

    Mock::given(method("POST"))
        .and(path("/model-proxy/v1/api/v1/models/load"))
        .respond_with(ResponseTemplate::new(200))
        .expect(0)
        .mount(&p.mock)
        .await;

    // The backend ANSWERS 200 with a body that is not JSON.
    Mock::given(method("POST"))
        .and(path("/model-proxy/v1/api/v0/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_string("<html>not json</html>"))
        .mount(&p.mock)
        .await;

    let started = Instant::now();
    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": OLLAMA_MODEL,
            "messages": [{ "role": "user", "content": "hello" }],
            "stream": false
        }))
        .send()
        .await
        .expect("POST /api/chat decode error");
    let elapsed = started.elapsed();

    assert_eq!(resp.status(), 500, "decode failure keeps its 500");
    let body: Value = resp.json().await.expect("proxy error body is json");
    let err = body["error"].as_str().expect("error key is a string");
    assert!(
        err.starts_with("invalid JSON from LM Studio"),
        "client must still see the decode wording verbatim, got: {err}"
    );
    assert!(
        elapsed < Duration::from_millis(1500),
        "no load detour may add a wait to an answered response; took {elapsed:?} \
         (check that POST /api/v1/models/load was not fired)"
    );

    p.mock.verify().await;
}
