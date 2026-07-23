// Integration tests for POST /api/web_search (generic JSON-passthrough provider).
//
// `spawn_proxy_with_search()` configures `--search-url` to the mock server's
// `/search` endpoint; the plain `spawn_proxy()` leaves it unset (501).

use serde_json::{Value, json};
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, ResponseTemplate};

use crate::common::{TEST_SEARCH_API_KEY, spawn_proxy, spawn_proxy_with_search};

#[tokio::test]
async fn web_search_unconfigured_returns_501() {
    let p = spawn_proxy().await;
    let resp = p
        .client
        .post(p.url("/api/web_search"))
        .json(&json!({ "query": "what is ollama" }))
        .send()
        .await
        .expect("POST /api/web_search");
    assert_eq!(resp.status().as_u16(), 501);
}

#[tokio::test]
async fn web_search_forwards_query_and_returns_provider_results_verbatim() {
    let p = spawn_proxy_with_search().await;
    let provider_body = json!({
        "results": [
            { "title": "Ollama", "url": "https://ollama.com/", "content": "Cloud models..." },
            { "title": "Docs", "url": "https://docs.ollama.com/", "content": "Docs..." }
        ]
    });
    Mock::given(method("POST"))
        .and(path("/search"))
        .and(body_partial_json(
            json!({ "query": "what is ollama", "max_results": 3 }),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(&provider_body))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/web_search"))
        .json(&json!({ "query": "what is ollama", "max_results": 3 }))
        .send()
        .await
        .expect("POST /api/web_search");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json body");
    assert_eq!(
        body, provider_body,
        "provider results must pass through verbatim"
    );
    p.mock.verify().await;
}

#[tokio::test]
async fn web_search_forwards_bearer_token() {
    let p = spawn_proxy_with_search().await;
    Mock::given(method("POST"))
        .and(path("/search"))
        .and(header(
            "authorization",
            format!("Bearer {TEST_SEARCH_API_KEY}").as_str(),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "results": [] })))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/web_search"))
        .json(&json!({ "query": "x" }))
        .send()
        .await
        .expect("POST /api/web_search");
    assert_eq!(resp.status(), 200);
    p.mock.verify().await;
}

#[tokio::test]
async fn web_search_whitespace_query_returns_400() {
    let p = spawn_proxy_with_search().await;
    let resp = p
        .client
        .post(p.url("/api/web_search"))
        .json(&json!({ "query": "   " }))
        .send()
        .await
        .expect("POST /api/web_search");
    assert_eq!(resp.status().as_u16(), 400);
}

#[tokio::test]
async fn web_search_non_json_provider_returns_502() {
    let p = spawn_proxy_with_search().await;
    Mock::given(method("POST"))
        .and(path("/search"))
        .respond_with(ResponseTemplate::new(200).set_body_string("definitely not json"))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/web_search"))
        .json(&json!({ "query": "x" }))
        .send()
        .await
        .expect("POST /api/web_search");
    assert_eq!(resp.status().as_u16(), 502);
}

#[tokio::test]
async fn web_search_missing_query_returns_400() {
    let p = spawn_proxy_with_search().await;
    let resp = p
        .client
        .post(p.url("/api/web_search"))
        .json(&json!({ "max_results": 5 }))
        .send()
        .await
        .expect("POST /api/web_search");
    assert_eq!(resp.status().as_u16(), 400);
}

#[tokio::test]
async fn web_search_provider_error_returns_502() {
    let p = spawn_proxy_with_search().await;
    Mock::given(method("POST"))
        .and(path("/search"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/web_search"))
        .json(&json!({ "query": "boom" }))
        .send()
        .await
        .expect("POST /api/web_search");
    assert_eq!(resp.status().as_u16(), 502);
}

#[tokio::test]
async fn web_search_defaults_max_results_to_5() {
    let p = spawn_proxy_with_search().await;
    Mock::given(method("POST"))
        .and(path("/search"))
        .and(body_partial_json(json!({ "max_results": 5 })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "results": [] })))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/web_search"))
        .json(&json!({ "query": "x" }))
        .send()
        .await
        .expect("POST /api/web_search");
    assert_eq!(resp.status(), 200);
    p.mock.verify().await;
}

#[tokio::test]
async fn web_search_clamps_max_results_to_10() {
    let p = spawn_proxy_with_search().await;
    Mock::given(method("POST"))
        .and(path("/search"))
        .and(body_partial_json(json!({ "max_results": 10 })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "results": [] })))
        .expect(1)
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/web_search"))
        .json(&json!({ "query": "x", "max_results": 50 }))
        .send()
        .await
        .expect("POST /api/web_search");
    assert_eq!(resp.status(), 200);
    p.mock.verify().await;
}

#[tokio::test]
async fn web_search_wrong_shape_returns_502() {
    let p = spawn_proxy_with_search().await;
    // Provider returns a non-Ollama shape; the proxy must not forward it verbatim.
    Mock::given(method("POST"))
        .and(path("/search"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({ "data": ["not", "ollama", "shaped"] })),
        )
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/web_search"))
        .json(&json!({ "query": "hi" }))
        .send()
        .await
        .expect("POST /api/web_search wrong shape");

    assert_eq!(
        resp.status().as_u16(),
        502,
        "a wrong-shaped provider response must be rejected as 502"
    );
    let body: Value = resp.json().await.expect("json error body");
    assert!(
        body["error"]
            .as_str()
            .unwrap_or_default()
            .to_lowercase()
            .contains("shape"),
        "502 must describe the shape mismatch; got {body}"
    );
}

#[tokio::test]
async fn web_search_result_missing_field_returns_502() {
    let p = spawn_proxy_with_search().await;
    // The container is a well-formed `{results:[{...}]}`, but an item is
    // missing a documented field — this must not pass as a container check alone.
    Mock::given(method("POST"))
        .and(path("/search"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "results": [{ "foo": 1 }]
        })))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/web_search"))
        .json(&json!({ "query": "hi" }))
        .send()
        .await
        .expect("POST /api/web_search missing field");

    assert_eq!(
        resp.status().as_u16(),
        502,
        "an item missing title/url/content must be rejected as 502"
    );
    let body: Value = resp.json().await.expect("json error body");
    let msg = body["error"].as_str().unwrap_or_default().to_lowercase();
    assert!(
        msg.contains("results[0]") && msg.contains("title"),
        "502 must name the failing index and field; got {body}"
    );
}

#[tokio::test]
async fn web_search_result_wrong_type_field_returns_502() {
    let p = spawn_proxy_with_search().await;
    // Field present but wrong type (`content: null`) is a distinct failure mode
    // from a missing key — a check that only tests presence would miss this.
    Mock::given(method("POST"))
        .and(path("/search"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "results": [{ "title": "x", "url": "y", "content": null }]
        })))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/web_search"))
        .json(&json!({ "query": "hi" }))
        .send()
        .await
        .expect("POST /api/web_search wrong-type field");

    assert_eq!(
        resp.status().as_u16(),
        502,
        "a non-string documented field must be rejected as 502"
    );
    let body: Value = resp.json().await.expect("json error body");
    let msg = body["error"].as_str().unwrap_or_default().to_lowercase();
    assert!(
        msg.contains("results[0]") && msg.contains("content"),
        "502 must name the failing index and field; got {body}"
    );
}

#[tokio::test]
async fn web_search_extra_fields_in_result_still_pass() {
    let p = spawn_proxy_with_search().await;
    // A provider returning MORE than the documented three fields is fine; only
    // a missing/wrong-typed documented field must be rejected.
    let provider_body = json!({
        "results": [
            { "title": "Ollama", "url": "https://ollama.com/", "content": "text", "score": 0.9 }
        ]
    });
    Mock::given(method("POST"))
        .and(path("/search"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&provider_body))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/web_search"))
        .json(&json!({ "query": "hi" }))
        .send()
        .await
        .expect("POST /api/web_search extra fields");
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json body");
    assert_eq!(body, provider_body);
}

#[tokio::test]
async fn web_search_empty_results_stays_200() {
    let p = spawn_proxy_with_search().await;
    Mock::given(method("POST"))
        .and(path("/search"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "results": [] })))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/web_search"))
        .json(&json!({ "query": "no hits" }))
        .send()
        .await
        .expect("POST /api/web_search empty results");
    assert_eq!(
        resp.status(),
        200,
        "an empty results array is a legitimate no-hits response"
    );
}
