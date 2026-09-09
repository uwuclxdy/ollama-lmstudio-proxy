// The anthropic envelope must survive a response-rebuilding outer layer:
// the envelope choice rides the error itself (set at the rejection site),
// not on response extensions that any rebuilding layer would drop.

use serde_json::Value;

use crate::common::spawn_proxy_with_rebuild_layer;

#[tokio::test]
async fn anthropic_envelope_survives_response_rebuilding_layer() {
    let p = spawn_proxy_with_rebuild_layer().await;

    // An extraction rejection (undecodable path param) on the anthropic
    // surface: the proxy generates the error, no backend involved, and the
    // outermost layer rebuilds the response on its way out. If the envelope
    // choice rode response extensions, the rebuilt response would lose it.
    let resp = p
        .client
        .post(p.url("/v1/messages/%FF"))
        .body("{}")
        .send()
        .await
        .expect("POST /v1/messages/%FF");
    assert_eq!(resp.status(), 400);
    let body: Value = resp.json().await.expect("400 body must be JSON");
    let obj = body
        .as_object()
        .expect("anthropic error envelope is an object");
    assert_eq!(obj.len(), 2, "envelope must be the anthropic shape: {body}");
    assert_eq!(obj["type"], "error");
    let err = obj["error"].as_object().expect("`error` is an object");
    assert_eq!(err.len(), 2);
    assert_eq!(err["type"], "invalid_request_error");
    let msg = err["message"].as_str().expect("`message` carries a string");
    assert!(msg.contains("path"), "message must name the param: {msg}");
}

#[tokio::test]
async fn ollama_envelope_survives_response_rebuilding_layer() {
    let p = spawn_proxy_with_rebuild_layer().await;

    // Same rejection on a non-anthropic surface: the ollama envelope must
    // survive the rebuild too, proving the layer did not swallow errors.
    let resp = p
        .client
        .post(p.url("/api/blobs/%FF"))
        .body("blob")
        .send()
        .await
        .expect("POST /api/blobs/%FF");
    assert_eq!(resp.status(), 400);
    let body: Value = resp.json().await.expect("400 body must be JSON");
    let obj = body
        .as_object()
        .expect("ollama error envelope is an object");
    assert_eq!(obj.len(), 1, "ollama envelope has only `error`: {body}");
    let msg = obj["error"].as_str().expect("`error` carries a string");
    assert!(msg.contains("digest"), "message must name the param: {msg}");
}
