// Integration tests for POST /api/web_fetch.
//
// web_fetch uses its OWN HTTP client against an arbitrary URL (not LM Studio),
// so these tests point it at the same wiremock server (`p.mock`) to stay
// hermetic — no real network egress.

use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, ResponseTemplate};

use crate::common::spawn_proxy;

const PAGE_HTML: &str = r##"<html><head><title>Test &amp; Page</title></head>
<body><h1>Heading</h1><p>Some text.</p>
<a href="/relative">Rel</a>
<a href="https://external.example/x">Ext</a>
<a href="#frag">Frag</a>
<a href="mailto:a@b.com">Mail</a>
</body></html>"##;

#[tokio::test]
async fn web_fetch_returns_title_content_and_links() {
    let p = spawn_proxy().await;
    Mock::given(method("GET"))
        .and(path("/page"))
        .respond_with(ResponseTemplate::new(200).set_body_string(PAGE_HTML))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/web_fetch"))
        .json(&json!({ "url": format!("{}/page", p.mock.uri()) }))
        .send()
        .await
        .expect("POST /api/web_fetch");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json body");

    assert_eq!(body["title"], json!("Test & Page"));

    let content = body["content"].as_str().expect("content string");
    assert!(
        content.contains("Heading"),
        "markdown content should include the heading: {content}"
    );

    let links: Vec<String> = body["links"]
        .as_array()
        .expect("links array")
        .iter()
        .map(|v| v.as_str().unwrap_or_default().to_string())
        .collect();
    // Relative link resolves against the fetched URL; the external one is kept;
    // fragment and mailto are dropped.
    assert!(
        links.contains(&format!("{}/relative", p.mock.uri())),
        "relative link must resolve absolute: {links:?}"
    );
    assert!(
        links.contains(&"https://external.example/x".to_string()),
        "external link must be present: {links:?}"
    );
    assert!(
        !links
            .iter()
            .any(|l| l.contains("mailto") || l.contains('#')),
        "mailto/fragment links must be dropped: {links:?}"
    );
}

#[tokio::test]
async fn web_fetch_missing_url_returns_400() {
    let p = spawn_proxy().await;
    let resp = p
        .client
        .post(p.url("/api/web_fetch"))
        .json(&json!({ "not_url": "x" }))
        .send()
        .await
        .expect("POST /api/web_fetch");
    assert_eq!(resp.status().as_u16(), 400);
}

#[tokio::test]
async fn web_fetch_rejects_non_http_scheme_with_400() {
    let p = spawn_proxy().await;
    let resp = p
        .client
        .post(p.url("/api/web_fetch"))
        .json(&json!({ "url": "ftp://example.com/file" }))
        .send()
        .await
        .expect("POST /api/web_fetch");
    assert_eq!(resp.status().as_u16(), 400);
}

#[tokio::test]
async fn web_fetch_upstream_error_returns_502() {
    let p = spawn_proxy().await;
    Mock::given(method("GET"))
        .and(path("/missing"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/web_fetch"))
        .json(&json!({ "url": format!("{}/missing", p.mock.uri()) }))
        .send()
        .await
        .expect("POST /api/web_fetch");
    assert_eq!(resp.status().as_u16(), 502);
}

#[tokio::test]
async fn web_fetch_upstream_500_returns_502() {
    let p = spawn_proxy().await;
    Mock::given(method("GET"))
        .and(path("/boom"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/web_fetch"))
        .json(&json!({ "url": format!("{}/boom", p.mock.uri()) }))
        .send()
        .await
        .expect("POST /api/web_fetch");
    assert_eq!(resp.status().as_u16(), 502);
}

// Relative links must resolve against the POST-redirect URL, not the original.
#[tokio::test]
async fn web_fetch_resolves_links_against_post_redirect_url() {
    let p = spawn_proxy().await;
    let end = format!("{}/dir2/end", p.mock.uri());

    Mock::given(method("GET"))
        .and(path("/dir1/start"))
        .respond_with(ResponseTemplate::new(302).insert_header("location", end.as_str()))
        .mount(&p.mock)
        .await;
    Mock::given(method("GET"))
        .and(path("/dir2/end"))
        .respond_with(ResponseTemplate::new(200).set_body_string(
            r#"<html><head><title>End</title></head><body><a href="child">c</a></body></html>"#,
        ))
        .mount(&p.mock)
        .await;

    let resp = p
        .client
        .post(p.url("/api/web_fetch"))
        .json(&json!({ "url": format!("{}/dir1/start", p.mock.uri()) }))
        .send()
        .await
        .expect("POST /api/web_fetch redirect");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json body");
    let links: Vec<String> = body["links"]
        .as_array()
        .expect("links array")
        .iter()
        .map(|v| v.as_str().unwrap_or_default().to_string())
        .collect();
    assert!(
        links.contains(&format!("{}/dir2/child", p.mock.uri())),
        "relative link must resolve against the post-redirect URL (/dir2/), got: {links:?}"
    );
}
