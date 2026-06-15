// Unit tests for the pure helpers behind `POST /api/web_fetch`
// (`src/api/web.rs`). The networked handler itself is covered by the
// `web_fetch` integration suite.

use super::*;

fn base() -> Url {
    Url::parse("https://example.com/docs/").expect("base url")
}

// ── normalize_url ────────────────────────────────────────────────────────────

#[test]
fn normalize_url_defaults_bare_host_to_https() {
    let url = normalize_url("ollama.com").expect("bare host");
    assert_eq!(url.scheme(), "https");
    assert_eq!(url.host_str(), Some("ollama.com"));
}

#[test]
fn normalize_url_preserves_explicit_scheme() {
    assert_eq!(
        normalize_url("http://example.com/x")
            .expect("http")
            .scheme(),
        "http"
    );
    assert_eq!(
        normalize_url("https://example.com/x")
            .expect("https")
            .scheme(),
        "https"
    );
}

#[test]
fn normalize_url_rejects_non_http_schemes() {
    assert!(normalize_url("ftp://example.com").is_err());
    assert!(normalize_url("file:///etc/passwd").is_err());
}

#[test]
fn normalize_url_rejects_garbage() {
    assert!(normalize_url("http://").is_err());
}

// ── extract_title ────────────────────────────────────────────────────────────

#[test]
fn extract_title_basic() {
    assert_eq!(
        extract_title("<html><head><title>Hello</title></head></html>"),
        Some("Hello".to_string())
    );
}

#[test]
fn extract_title_is_case_insensitive_and_handles_attributes() {
    assert_eq!(
        extract_title(r#"<TITLE data-x="1">Mixed Case</TITLE>"#),
        Some("Mixed Case".to_string())
    );
}

#[test]
fn extract_title_decodes_common_entities() {
    assert_eq!(
        extract_title("<title>Tom &amp; Jerry &lt;3</title>"),
        Some("Tom & Jerry <3".to_string())
    );
}

#[test]
fn extract_title_absent_or_empty_yields_none() {
    assert_eq!(extract_title("<html><body>no title</body></html>"), None);
    assert_eq!(extract_title("<title>   </title>"), None);
}

// ── extract_links ────────────────────────────────────────────────────────────

#[test]
fn extract_links_resolves_relative_and_absolute() {
    let html = r#"<a href="/a">A</a> <a href='sub/b'>B</a> <a href="https://other.com/c">C</a>"#;
    let links = extract_links(html, &base());
    assert_eq!(
        links,
        vec![
            "https://example.com/a".to_string(),
            "https://example.com/docs/sub/b".to_string(),
            "https://other.com/c".to_string(),
        ]
    );
}

#[test]
fn extract_links_skips_fragments_mailto_javascript_and_dedupes() {
    let html = r##"<a href="#top">top</a><a href="mailto:x@y.com">mail</a>
        <a href="javascript:void(0)">js</a><a href="/dup">d</a><a href="/dup">d2</a>"##;
    let links = extract_links(html, &base());
    assert_eq!(links, vec!["https://example.com/dup".to_string()]);
}

#[test]
fn extract_links_ignores_non_http_targets() {
    let html = r#"<a href="ftp://example.com/file">f</a><a href="tel:+15555">t</a>"#;
    assert!(extract_links(html, &base()).is_empty());
}

#[test]
fn extract_links_handles_no_anchors() {
    assert!(extract_links("<p>plain text, no links</p>", &base()).is_empty());
}

// ── resolve_link ─────────────────────────────────────────────────────────────

#[test]
fn resolve_link_filters_and_resolves() {
    let b = base();
    assert_eq!(
        resolve_link(&b, "page.html"),
        Some("https://example.com/docs/page.html".to_string())
    );
    assert_eq!(resolve_link(&b, ""), None);
    assert_eq!(resolve_link(&b, "#section"), None);
    assert_eq!(resolve_link(&b, "mailto:a@b.com"), None);
}

// ── extract_links: robustness ────────────────────────────────────────────────

#[test]
fn extract_links_recovers_from_unclosed_quoted_href() {
    // The first href opens a single quote that never closes; the scanner must
    // skip it and still find the later well-formed href (best-effort).
    let html = r#"<a href='unclosed single quote <a href="https://valid.example/x">ok</a>"#;
    assert_eq!(
        extract_links(html, &base()),
        vec!["https://valid.example/x".to_string()]
    );
}

#[test]
fn extract_links_handles_unquoted_href() {
    let html = "<a href=/bare>x</a> <a href=https://ext.example/y >z</a>";
    let links = extract_links(html, &base());
    assert!(
        links.contains(&"https://example.com/bare".to_string()),
        "{links:?}"
    );
    assert!(
        links.contains(&"https://ext.example/y".to_string()),
        "{links:?}"
    );
}

#[test]
fn extract_links_is_multibyte_safe() {
    // Multibyte chars in title and href must not break byte-index slicing.
    let html = "<title>Café — Привет</title><a href=\"/café\">x</a>";
    assert_eq!(extract_title(html), Some("Café — Привет".to_string()));
    assert_eq!(extract_links(html, &base()).len(), 1);
}

#[test]
fn extract_links_caps_at_max_links() {
    let mut html = String::new();
    for i in 0..(MAX_LINKS + 50) {
        html.push_str(&format!("<a href=\"/p{i}\">{i}</a>"));
    }
    assert_eq!(extract_links(&html, &base()).len(), MAX_LINKS);
}

// ── SSRF guard ───────────────────────────────────────────────────────────────

#[test]
fn is_blocked_ip_rejects_non_public() {
    for s in [
        "127.0.0.1",
        "10.0.0.1",
        "192.168.1.1",
        "172.16.0.1",
        "169.254.169.254", // cloud metadata
        "0.0.0.0",
        "::1",
        "fc00::1",          // unique-local
        "fe80::1",          // link-local
        "::ffff:127.0.0.1", // v4-mapped loopback
    ] {
        let ip: IpAddr = s.parse().expect("ip");
        assert!(is_blocked_ip(ip), "{s} should be blocked");
    }
}

#[test]
fn is_blocked_ip_allows_public() {
    for s in [
        "1.1.1.1",
        "8.8.8.8",
        "93.184.216.34",
        "2606:4700:4700::1111",
    ] {
        let ip: IpAddr = s.parse().expect("ip");
        assert!(!is_blocked_ip(ip), "{s} should be allowed");
    }
}

#[tokio::test]
async fn validate_public_host_rejects_private_literals() {
    for u in [
        "http://127.0.0.1/x",
        "http://169.254.169.254/latest/meta-data",
        "http://10.1.2.3/admin",
    ] {
        assert!(
            validate_public_host(&Url::parse(u).expect("url"))
                .await
                .is_err(),
            "{u} must be rejected"
        );
    }
}

#[tokio::test]
async fn validate_public_host_allows_public_literal() {
    assert!(
        validate_public_host(&Url::parse("http://1.1.1.1/").expect("url"))
            .await
            .is_ok()
    );
}
