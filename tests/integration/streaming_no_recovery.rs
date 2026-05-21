// T15 — when `enable_chunk_recovery` is OFF and a JSON parse error happens
// mid-stream, the proxy must emit a single `{"error":"…"}` NDJSON line and
// close the stream. No `done:true` line may follow the error.

use futures_util::StreamExt;
use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, ResponseTemplate};

use crate::common::spawn_proxy_with_recovery;

async fn collect_ndjson(resp: reqwest::Response) -> Vec<Value> {
    let mut stream = resp.bytes_stream();
    let mut buf = Vec::new();
    while let Some(chunk) = stream.next().await {
        buf.extend_from_slice(&chunk.expect("stream chunk"));
    }
    let text = String::from_utf8(buf).expect("utf8 body");
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("bad JSON line {l:?}: {e}")))
        .collect()
}

fn sse_response(body: String) -> ResponseTemplate {
    ResponseTemplate::new(200)
        .insert_header("content-type", "text/event-stream")
        .set_body_raw(body.into_bytes(), "text/event-stream")
}

#[tokio::test]
async fn stream_emits_error_line_on_upstream_parse_failure() {
    let p = spawn_proxy_with_recovery(false).await;

    // One good chunk, then a malformed JSON line that recovery cannot rescue
    // (no `{` or `[` at all — pure garbage), then a chunk that would be valid
    // if the stream were still alive.
    let mut body = String::new();
    body.push_str(
        "data: {\"choices\":[{\"delta\":{\"content\":\"A\"},\"finish_reason\":null}]}\n\n",
    );
    body.push_str("data: not json at all\n\n");
    body.push_str(
        "data: {\"choices\":[{\"delta\":{\"content\":\"B\"},\"finish_reason\":\"stop\"}]}\n\n",
    );
    body.push_str("data: [DONE]\n\n");

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(sse_response(body))
        .mount(&p.mock)
        .await;

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

    let resp = p
        .client
        .post(p.url("/api/chat"))
        .json(&json!({
            "model": "llama3",
            "messages": [{"role": "user", "content": "go"}],
            "stream": true
        }))
        .send()
        .await
        .expect("POST /api/chat");

    let chunks = collect_ndjson(resp).await;
    assert!(!chunks.is_empty(), "expected at least one chunk");

    // Exactly one chunk must carry an `error` key.
    let error_chunks: Vec<&Value> = chunks.iter().filter(|c| c.get("error").is_some()).collect();
    assert_eq!(
        error_chunks.len(),
        1,
        "expected exactly one error line; got {chunks:#?}"
    );

    // The error line must be the FINAL line — no done:true (or anything else)
    // may follow once the stream has signalled an error.
    let last = chunks.last().expect("last chunk");
    assert!(
        last.get("error").is_some(),
        "error line must terminate the stream, got trailing: {last}"
    );

    // No `done:true` chunk anywhere — the spec says the error replaces the
    // terminal chunk rather than appearing alongside it.
    let has_done_true = chunks
        .iter()
        .any(|c| c.get("done").and_then(|d| d.as_bool()) == Some(true));
    assert!(
        !has_done_true,
        "no done:true line may follow an error; got {chunks:#?}"
    );
}
