#![allow(clippy::unwrap_used, clippy::expect_used)]
// The post-trigger wait on the cold-load path must end on the load's own
// signal — the model's entry in GET /api/v1/models gaining a loaded instance —
// bounded by load_timeout_seconds, not blind-sleep the full timeout.
//
// `trigger_model_loading` returning Ok(true) means its warmup ping already
// talked to the model (success or client-error), so in the common case the
// instance is already resident and the retry must follow immediately. A blind
// `sleep(load_timeout_seconds)` adds the whole timeout to every cold load.

use std::time::{Duration, Instant};

use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, ResponseTemplate};

use crate::common::{TestProxy, spawn_proxy_with_load_timeout};

const MODEL_KEY: &str = "some-model";
const OLLAMA_MODEL: &str = "some-model:latest";

fn model_entry(loaded: bool) -> Value {
    let loaded_instances = if loaded {
        json!([{ "id": format!("{MODEL_KEY}-inst"), "config": { "context_length": 4096 } }])
    } else {
        json!([])
    };
    json!({
        "key": MODEL_KEY,
        "type": "llm",
        "publisher": "test",
        "architecture": "llama",
        "format": "gguf",
        "quantization": { "name": "Q4_K_M", "bits_per_weight": 4.5 },
        "max_context_length": 4096,
        "loaded_instances": loaded_instances,
        "capabilities": { "vision": false, "trained_for_tool_use": false }
    })
}

fn lms_models(entry: Value) -> Value {
    json!({ "models": [entry] })
}

fn chat_ok() -> Value {
    json!({
        "id": "chatcmpl-event-wait",
        "object": "chat.completion",
        "created": 1_700_000_000u64,
        "model": MODEL_KEY,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "loaded and ready" },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8 }
    })
}

fn no_models_loaded_body() -> Value {
    json!({ "error": { "message": "No models loaded. Please load a model first." } })
}

fn load_ok() -> Value {
    json!({
        "status": "loaded",
        "instance_id": MODEL_KEY,
        "load_time_seconds": 0.1
    })
}

async fn mount_chat_400_no_models(p: &TestProxy, expected_hits: u64) {
    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .respond_with(ResponseTemplate::new(400).set_body_json(no_models_loaded_body()))
        .up_to_n_times(expected_hits)
        .expect(expected_hits)
        .mount(&p.mock)
        .await;
}

fn chat_request_body() -> serde_json::Value {
    json!({
        "model": OLLAMA_MODEL,
        "messages": [{ "role": "user", "content": "hello" }],
        "stream": false
    })
}

async fn post_chat(p: &TestProxy) -> (reqwest::Response, Duration) {
    let started = Instant::now();
    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&chat_request_body())
        .send()
        .await
        .expect("POST /api/chat cold load");
    (resp, started.elapsed())
}

// The warmup ping succeeds (200) and the instance is already resident. The
// retry must fire immediately — total time far below the load_timeout_seconds
// bound.
#[tokio::test]
async fn cold_load_retry_follows_the_load_event_not_the_timeout() {
    let p = spawn_proxy_with_load_timeout(5).await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lms_models(model_entry(true))))
        .mount(&p.mock)
        .await;

    mount_chat_400_no_models(&p, 1).await;

    Mock::given(method("POST"))
        .and(path("/api/v1/models/load"))
        .respond_with(ResponseTemplate::new(200).set_body_json(load_ok()))
        .expect(1)
        .mount(&p.mock)
        .await;

    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_ok()))
        .mount(&p.mock)
        .await;

    let (resp, elapsed) = post_chat(&p).await;

    assert_eq!(
        resp.status(),
        200,
        "proxy must return 200 after cold-load retry"
    );
    let body: Value = resp.json().await.expect("JSON body");
    assert_eq!(body["message"]["content"], "loaded and ready");

    // The load event is visible immediately; a blind sleep of the 5 s bound
    // would blow well past this. Generous slack for scheduler jitter.
    assert!(
        elapsed < Duration::from_secs(3),
        "post-trigger wait must end on the load event, not the full timeout; took {elapsed:?}"
    );

    p.mock.verify().await;
}

// The warmup ping answers a client error (the embedder case the trigger treats
// as ok) and the models endpoint only reveals the loaded instance after a
// delay. The wait must track the instance's appearance and still finish inside
// the bound — neither retrying blind nor sleeping the full timeout.
#[tokio::test]
async fn cold_load_poll_tracks_delayed_instance_within_bound() {
    let p = spawn_proxy_with_load_timeout(5).await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(lms_models(model_entry(true)))
                .set_delay(Duration::from_millis(250)),
        )
        .mount(&p.mock)
        .await;

    // The first 400 is the operation, the second is the warmup ping (a
    // client-error ping counts as trigger success); the retry must reach the
    // 200 mock.
    mount_chat_400_no_models(&p, 2).await;

    Mock::given(method("POST"))
        .and(path("/api/v1/models/load"))
        .respond_with(ResponseTemplate::new(200).set_body_json(load_ok()))
        .expect(1)
        .mount(&p.mock)
        .await;

    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_ok()))
        .mount(&p.mock)
        .await;

    let (resp, elapsed) = post_chat(&p).await;

    assert_eq!(
        resp.status(),
        200,
        "the retry must run once the instance appears, not give up or time out"
    );
    let body: Value = resp.json().await.expect("JSON body");
    assert_eq!(body["message"]["content"], "loaded and ready");

    // Instance visible within ~2-3 s (several delayed 250 ms models calls);
    // the 5 s blind sleep would push this past 7 s.
    assert!(
        elapsed < Duration::from_secs(4),
        "wait must track the instance's appearance, not sleep the full timeout; took {elapsed:?}"
    );

    p.mock.verify().await;
}

// The catalog reports the model UNLOADED for its first dozen reads — model
// resolution runs several times before the wait begins (handler resolution,
// the trigger's key resolution and its two load records), so the count stays
// generous enough that the waiter's first polls are non-matches — and loaded
// afterwards. The waiter must keep polling through them and retry on the
// transition, which is the seam this feature exists for: giving up on the
// first non-match (or sleeping the whole bound) is the regression class.
#[tokio::test]
async fn cold_load_waits_through_non_matching_polls_until_the_instance_appears() {
    let p = spawn_proxy_with_load_timeout(5).await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lms_models(model_entry(false))))
        .up_to_n_times(12)
        .mount(&p.mock)
        .await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(lms_models(model_entry(true))))
        .mount(&p.mock)
        .await;

    mount_chat_400_no_models(&p, 2).await;

    Mock::given(method("POST"))
        .and(path("/api/v1/models/load"))
        .respond_with(ResponseTemplate::new(200).set_body_json(load_ok()))
        .expect(1)
        .mount(&p.mock)
        .await;

    Mock::given(method("POST"))
        .and(path("/api/v0/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_ok()))
        .mount(&p.mock)
        .await;

    let (resp, elapsed) = post_chat(&p).await;

    assert_eq!(
        resp.status(),
        200,
        "the retry must run after the transition, not before it"
    );
    let body: Value = resp.json().await.expect("JSON body");
    assert_eq!(body["message"]["content"], "loaded and ready");

    // The first poll is a non-match, so a waiter that keeps polling pays at
    // least one 250 ms cadence gap; one that gives up returns in milliseconds.
    assert!(
        elapsed >= Duration::from_millis(150),
        "waiter gave up before the instance appeared; took only {elapsed:?}"
    );
    assert!(
        elapsed < Duration::from_secs(4),
        "wait must end on the transition, not sleep the full timeout; took {elapsed:?}"
    );

    p.mock.verify().await;
}
