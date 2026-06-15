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
