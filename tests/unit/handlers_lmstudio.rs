use super::*;
use bytes::Bytes;
use http_body_util::BodyExt;
use serde_json::{Value, json};

use crate::storage::virtual_models::VirtualModelMetadata;

// handlers/lmstudio.rs exposes one public function (handle_lmstudio_passthrough)
// and one pub struct (LmStudioPassthroughRequest).  Its pure helpers —
// determine_passthrough_endpoint_url, merge_virtual_models_into_listing and
// rebuild_passthrough_response — are private (not re-exported).  All other
// behaviour involves live HTTP or streaming I/O.
//
// What we can test without a network is the LmStudioPassthroughRequest
// struct (construction / field access), the `/v1/models` alias merge, and the
// fact that the module compiles with the expected public surface.

#[test]
fn passthrough_request_fields_accessible() {
    let req = LmStudioPassthroughRequest {
        method: http::Method::GET,
        endpoint: "/v1/models".to_string(),
        body: Bytes::new(),
        headers: http::HeaderMap::new(),
        query: None,
    };

    assert_eq!(req.method, http::Method::GET);
    assert_eq!(req.endpoint, "/v1/models");
    assert!(req.body.is_empty());
    assert!(req.query.is_none());
}

#[test]
fn passthrough_request_with_query() {
    let req = LmStudioPassthroughRequest {
        method: http::Method::POST,
        endpoint: "/v1/chat/completions".to_string(),
        body: Bytes::from_static(b"{}"),
        headers: http::HeaderMap::new(),
        query: Some("stream=true".to_string()),
    };

    assert_eq!(req.query.as_deref(), Some("stream=true"));
    assert!(!req.body.is_empty());
}

fn alias_entry(name: &str, target_model_id: &str) -> VirtualModelEntry {
    VirtualModelEntry {
        name: name.to_string(),
        source_model: target_model_id.to_string(),
        target_model_id: target_model_id.to_string(),
        created_at: jiff::Timestamp::now(),
        updated_at: jiff::Timestamp::now(),
        metadata: VirtualModelMetadata::default(),
    }
}

fn merged_listing(body: &Value, aliases: &[VirtualModelEntry]) -> Value {
    let bytes =
        merge_virtual_models_into_listing(&serde_json::to_vec(&body).expect("body json"), aliases)
            .expect("a model listing with a new alias merges");
    serde_json::from_slice(&bytes).expect("merged json")
}

#[test]
fn models_listing_alias_rows_mirror_their_targets() {
    let merged = merged_listing(
        &json!({
            "object": "list",
            "data": [
                { "id": "first-model", "object": "model", "owned_by": "publisher-one", "created": 111 },
                { "id": "second-model", "object": "model", "owned_by": "publisher-two", "created": 222 }
            ]
        }),
        &[
            alias_entry("alias-b", "second-model"),
            alias_entry("alias-a", "first-model"),
        ],
    );

    assert_eq!(
        merged,
        json!({
            "object": "list",
            "data": [
                { "id": "first-model", "object": "model", "owned_by": "publisher-one", "created": 111 },
                { "id": "second-model", "object": "model", "owned_by": "publisher-two", "created": 222 },
                { "id": "alias-a", "object": "model", "owned_by": "publisher-one", "created": 111 },
                { "id": "alias-b", "object": "model", "owned_by": "publisher-two", "created": 222 }
            ]
        })
    );
}

#[test]
fn models_listing_orphan_alias_falls_back_to_the_first_entries_owned_by() {
    let merged = merged_listing(
        &json!({
            "data": [
                { "id": "first-model", "object": "model", "owned_by": "publisher-one", "created": 111 },
                { "id": "second-model", "object": "model", "owned_by": "publisher-two", "created": 222 }
            ]
        }),
        &[alias_entry("orphan", "removed-model")],
    );

    assert_eq!(
        merged,
        json!({ "data": [
            { "id": "first-model", "object": "model", "owned_by": "publisher-one", "created": 111 },
            { "id": "second-model", "object": "model", "owned_by": "publisher-two", "created": 222 },
            { "id": "orphan", "object": "model", "owned_by": "publisher-one" }
        ] })
    );
}

#[test]
fn models_listing_target_without_owned_by_does_not_borrow_the_first_entries() {
    let merged = merged_listing(
        &json!({
            "data": [
                { "id": "first-model", "object": "model", "owned_by": "publisher-one" },
                { "id": "target-model", "object": "model" }
            ]
        }),
        &[alias_entry("my-alias", "target-model")],
    );

    assert_eq!(
        merged,
        json!({ "data": [
            { "id": "first-model", "object": "model", "owned_by": "publisher-one" },
            { "id": "target-model", "object": "model" },
            { "id": "my-alias", "object": "model" }
        ] })
    );
}

#[test]
fn models_listing_with_no_entries_omits_owned_by() {
    let merged = merged_listing(
        &json!({ "data": [] }),
        &[alias_entry("my-alias", "target-model")],
    );

    assert_eq!(
        merged,
        json!({ "data": [{ "id": "my-alias", "object": "model" }] })
    );
}

#[test]
fn models_listing_keeps_one_entry_per_id() {
    let merged = merged_listing(
        &json!({ "data": [{ "id": "my-alias", "object": "model" }] }),
        &[
            alias_entry("my-alias", "target-model"),
            alias_entry("other-alias", "target-model"),
        ],
    );

    assert_eq!(
        merged,
        json!({ "data": [
            { "id": "my-alias", "object": "model" },
            { "id": "other-alias", "object": "model" }
        ] })
    );
}

#[test]
fn models_listing_forwards_a_non_json_body() {
    assert!(
        merge_virtual_models_into_listing(
            b"service temporarily unavailable",
            &[alias_entry("my-alias", "target-model")]
        )
        .is_none(),
        "a non-JSON body must stay the backend's to shape"
    );
}

#[test]
fn models_listing_forwards_json_that_is_not_a_listing() {
    assert!(
        merge_virtual_models_into_listing(
            br#"{"error":"boom"}"#,
            &[alias_entry("my-alias", "target-model")]
        )
        .is_none(),
        "JSON without a data array is not a model listing"
    );
}

#[test]
fn models_listing_with_no_aliases_keeps_backend_bytes() {
    assert!(
        merge_virtual_models_into_listing(br#"{"data":[]}"#, &[]).is_none(),
        "no alias to add must keep the backend's exact bytes"
    );
}

#[tokio::test]
async fn rebuilt_listing_keeps_status_and_headers_but_drops_replaced_body_headers() {
    let mut headers = http::HeaderMap::new();
    headers.insert("x-lmstudio-backend", http::HeaderValue::from_static("yes"));
    headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("application/json"),
    );
    headers.insert(
        http::header::CONTENT_LENGTH,
        http::HeaderValue::from_static("42"),
    );
    headers.insert(
        http::header::TRANSFER_ENCODING,
        http::HeaderValue::from_static("chunked"),
    );
    headers.insert(
        http::header::CONNECTION,
        http::HeaderValue::from_static("keep-alive"),
    );
    headers.insert("keep-alive", http::HeaderValue::from_static("timeout=5"));
    headers.insert(http::header::TE, http::HeaderValue::from_static("trailers"));
    headers.insert(
        http::header::TRAILER,
        http::HeaderValue::from_static("x-checksum"),
    );
    headers.insert(http::header::UPGRADE, http::HeaderValue::from_static("h2c"));
    headers.insert(
        http::header::PROXY_AUTHENTICATE,
        http::HeaderValue::from_static("Basic"),
    );
    headers.insert(
        http::header::PROXY_AUTHORIZATION,
        http::HeaderValue::from_static("Basic"),
    );
    headers.insert(
        http::header::ETAG,
        http::HeaderValue::from_static("\"deadbeef\""),
    );

    let response = rebuild_passthrough_response(
        http::StatusCode::OK,
        headers,
        Bytes::from_static(br#"{"data":[]}"#),
    );

    assert_eq!(response.status(), http::StatusCode::OK);
    assert_eq!(response.headers().get("x-lmstudio-backend").unwrap(), "yes");
    assert_eq!(
        response.headers().get(http::header::CONTENT_TYPE).unwrap(),
        "application/json"
    );
    for name in [
        "content-length",
        "transfer-encoding",
        "connection",
        "keep-alive",
        "te",
        "trailer",
        "upgrade",
        "proxy-authenticate",
        "proxy-authorization",
        "etag",
    ] {
        assert!(
            response.headers().get(name).is_none(),
            "{name} describes the backend's body, not the rebuilt one"
        );
    }
    let body = BodyExt::collect(response.into_body())
        .await
        .expect("collect body")
        .to_bytes();
    assert_eq!(&body[..], br#"{"data":[]}"#);
}
