//! `POST /api/web_fetch` — fetch a single URL and return it as
//! `{title, content, links}`, matching Ollama's web-fetch shape
//! (`api-docs/future/ollama/capabilities/web-search.md`).
//!
//! Unlike the rest of the proxy this talks to an arbitrary, user-supplied URL —
//! NOT LM Studio. It therefore uses its own client WITHOUT the LM Studio auth
//! header (the `--lmstudio-token` must never leak to third-party hosts) and
//! only allows `http`/`https`.

use std::collections::HashSet;
use std::net::{IpAddr, SocketAddr};
use std::sync::OnceLock;
use std::time::Duration;

use axum::response::Response;
use futures_util::StreamExt;
use serde_json::{Value, json};
use url::Url;

use crate::error::ProxyError;
use crate::http::json_response;

const WEB_FETCH_TIMEOUT_SECONDS: u64 = 30;
/// Max redirect hops to follow manually (each one is SSRF-revalidated).
const MAX_REDIRECTS: usize = 10;
/// Cap the returned `links` array so a link-heavy page can't bloat the response.
const MAX_LINKS: usize = 100;
/// Cap the fetched body so a hostile/huge response can't exhaust memory.
const MAX_RESPONSE_BYTES: usize = 10 * 1024 * 1024;

/// Dedicated client for outbound web fetches: no default auth header (so the LM
/// Studio token never leaks to third-party hosts), a bounded timeout, and a
/// descriptive user-agent. Redirects are followed MANUALLY (`Policy::none`) so
/// every hop can be re-validated against the SSRF guard.
fn web_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .user_agent(concat!("ollama-lmstudio-proxy/", env!("CARGO_PKG_VERSION")))
            .timeout(Duration::from_secs(WEB_FETCH_TIMEOUT_SECONDS))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap_or_else(|_| reqwest::Client::new())
    })
}

/// Fetch `body.url`, returning `{title, content (markdown), links}`.
///
/// `allow_private_fetch` comes from the per-proxy `Config`; when false (default)
/// the SSRF guard rejects private/loopback/link-local targets.
pub async fn handle_web_fetch(
    body: Value,
    allow_private_fetch: bool,
) -> Result<Response, ProxyError> {
    let raw_url = body
        .get("url")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| ProxyError::bad_request("Missing 'url' field"))?;

    let url = normalize_url(raw_url)?;

    // Follow redirects manually so every hop is SSRF-checked (the guard is
    // skipped only when `--allow-private-fetch` is set).
    let guard_ssrf = !allow_private_fetch;
    let (response, final_url) = fetch_following_redirects(url, guard_ssrf).await?;

    let status = response.status();
    if !status.is_success() {
        return Err(ProxyError::new(
            format!("web_fetch: '{final_url}' returned status {status}"),
            502,
        ));
    }

    // Reject an over-large declared body up front; the streamed read below
    // enforces the same cap for responses that omit Content-Length.
    if let Some(len) = response.content_length()
        && len > MAX_RESPONSE_BYTES as u64
    {
        return Err(ProxyError::new(
            format!("web_fetch: '{final_url}' body too large ({len} bytes)"),
            502,
        ));
    }

    let html = read_body_capped(response, &final_url).await?;

    let title = extract_title(&html).unwrap_or_default();
    // Skip script/style/noscript so their raw CSS/JS never leaks into the
    // markdown (htmd keeps their text content otherwise).
    let content = htmd::HtmlToMarkdownBuilder::new()
        .skip_tags(vec!["script", "style", "noscript"])
        .build()
        .convert(&html)
        .unwrap_or_default();
    let links = extract_links(&html, &final_url);

    Ok(json_response(&json!({
        "title": title,
        "content": content,
        "links": links,
    })))
}

/// Handle `POST /api/web_search` via a generic JSON-passthrough provider.
///
/// Forwards `{query, max_results}` to the operator-configured `search_url`
/// (optionally with a bearer key) and returns the provider's JSON verbatim —
/// expected to already be Ollama-shaped (`{results:[{title,url,content}]}`).
/// Returns 501 when no provider is configured. `search_url` is set by the
/// operator (not the caller), so there is no SSRF surface here.
pub async fn handle_web_search(
    body: Value,
    search_url: Option<&str>,
    search_api_key: Option<&str>,
) -> Result<Response, ProxyError> {
    let Some(provider_url) = search_url else {
        return Err(ProxyError::not_implemented(
            "web_search is not configured: set --search-url to a provider that accepts {query, max_results} and returns {results:[{title,url,content}]}",
        ));
    };

    let query = body
        .get("query")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| ProxyError::bad_request("Missing 'query' field"))?;

    // Ollama's web-search spec: max_results defaults to 5, capped at 10.
    let max_results = body
        .get("max_results")
        .and_then(|v| v.as_u64())
        .unwrap_or(5)
        .clamp(1, 10);

    let mut request = web_client()
        .post(provider_url)
        .json(&json!({ "query": query, "max_results": max_results }));
    if let Some(key) = search_api_key {
        request = request.bearer_auth(key);
    }

    let response = request.send().await.map_err(|e| {
        ProxyError::new(format!("web_search: request to provider failed: {e}"), 502)
    })?;
    let status = response.status();
    if !status.is_success() {
        return Err(ProxyError::new(
            format!("web_search: provider returned status {status}"),
            502,
        ));
    }

    let value: Value = response.json().await.map_err(|e| {
        ProxyError::new(
            format!("web_search: provider response was not valid JSON: {e}"),
            502,
        )
    })?;
    Ok(json_response(&value))
}

/// GET `start`, following up to [`MAX_REDIRECTS`] redirects manually. When
/// `guard_ssrf` is set, every hop's host is resolved and rejected if it maps to
/// a private/loopback/link-local address (SSRF defense). Returns the final
/// response and the URL it was served from (the correct base for relative links).
async fn fetch_following_redirects(
    start: Url,
    guard_ssrf: bool,
) -> Result<(reqwest::Response, Url), ProxyError> {
    let mut current = start;
    for _ in 0..=MAX_REDIRECTS {
        let client = client_for(&current, guard_ssrf).await?;
        let response = client.get(current.clone()).send().await.map_err(|e| {
            ProxyError::new(
                format!("web_fetch: request to '{current}' failed: {e}"),
                502,
            )
        })?;

        if response.status().is_redirection()
            && let Some(location) = response
                .headers()
                .get(reqwest::header::LOCATION)
                .and_then(|v| v.to_str().ok())
        {
            let next = current.join(location).map_err(|e| {
                ProxyError::new(
                    format!("web_fetch: bad redirect target '{location}': {e}"),
                    502,
                )
            })?;
            if !matches!(next.scheme(), "http" | "https") {
                return Err(ProxyError::bad_request(&format!(
                    "web_fetch: redirect to unsupported scheme '{}'",
                    next.scheme()
                )));
            }
            current = next;
            continue;
        }

        return Ok((response, current));
    }
    Err(ProxyError::new(
        format!("web_fetch: too many redirects starting from '{current}'"),
        502,
    ))
}

/// Build the reqwest client for one fetch hop. With the SSRF guard on, resolve
/// the host, reject if ANY address is non-public, and PIN the connection to a
/// validated address via `ClientBuilder::resolve` — so reqwest cannot re-resolve
/// at connect time and land on a DNS-rebinding attacker's private IP (closing
/// the validate-then-connect TOCTOU). With the guard off (`--allow-private-fetch`)
/// the shared client is reused.
async fn client_for(url: &Url, guard_ssrf: bool) -> Result<reqwest::Client, ProxyError> {
    if !guard_ssrf {
        return Ok(web_client().clone());
    }
    let host = url
        .host_str()
        .ok_or_else(|| ProxyError::bad_request("web_fetch: url has no host"))?;
    let addr = resolve_public_addr(url).await?;
    reqwest::Client::builder()
        .user_agent(concat!("ollama-lmstudio-proxy/", env!("CARGO_PKG_VERSION")))
        .timeout(Duration::from_secs(WEB_FETCH_TIMEOUT_SECONDS))
        .redirect(reqwest::redirect::Policy::none())
        .resolve(host, addr)
        .build()
        .map_err(|e| ProxyError::new(format!("web_fetch: client build failed: {e}"), 502))
}

/// Resolve `url`'s host and return a validated public `SocketAddr` to pin the
/// connection to. Rejects if ANY resolved address is non-public (loopback/
/// private/link-local/ULA/etc.), so a public name that also points at a private
/// IP is still refused.
async fn resolve_public_addr(url: &Url) -> Result<SocketAddr, ProxyError> {
    let host = url
        .host_str()
        .ok_or_else(|| ProxyError::bad_request("web_fetch: url has no host"))?;
    let port = url.port_or_known_default().unwrap_or(80);

    let addrs = tokio::net::lookup_host((host, port)).await.map_err(|e| {
        ProxyError::new(
            format!("web_fetch: DNS lookup for '{host}' failed: {e}"),
            502,
        )
    })?;

    let mut pinned: Option<SocketAddr> = None;
    for addr in addrs {
        if is_blocked_ip(addr.ip()) {
            return Err(ProxyError::bad_request(&format!(
                "web_fetch: refusing to fetch non-public address {} (host '{host}'); set --allow-private-fetch to override",
                addr.ip()
            )));
        }
        pinned.get_or_insert(addr);
    }
    pinned.ok_or_else(|| ProxyError::new(format!("web_fetch: host '{host}' did not resolve"), 502))
}

/// Is `ip` a non-public address that must not be reachable via web_fetch?
fn is_blocked_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let oct = v4.octets();
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_broadcast()
                || v4.is_multicast()
                // CGNAT 100.64.0.0/10 — carrier-grade NAT: routable-looking but
                // internal, so a public name resolving here would still hit a LAN.
                || (oct[0] == 100 && (64..=127).contains(&oct[1]))
                // 240.0.0.0/4 — reserved/experimental (255.255.255.255 broadcast is
                // already caught above; this covers the rest of the block).
                || oct[0] >= 240
        }
        IpAddr::V6(v6) => {
            if let Some(mapped) = v6.to_ipv4_mapped() {
                return is_blocked_ip(IpAddr::V4(mapped));
            }
            v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_multicast()
                || v6.is_unique_local()
                || v6.is_unicast_link_local()
        }
    }
}

/// Read the response body into a `String`, aborting if it exceeds
/// [`MAX_RESPONSE_BYTES`]. Streams chunk-by-chunk so a server that omits
/// `Content-Length` (e.g. chunked transfer) still can't exhaust memory.
async fn read_body_capped(response: reqwest::Response, url: &Url) -> Result<String, ProxyError> {
    let mut stream = response.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| {
            ProxyError::new(
                format!("web_fetch: reading body of '{url}' failed: {e}"),
                502,
            )
        })?;
        if buf.len() + chunk.len() > MAX_RESPONSE_BYTES {
            return Err(ProxyError::new(
                format!("web_fetch: '{url}' body exceeds {MAX_RESPONSE_BYTES} bytes"),
                502,
            ));
        }
        buf.extend_from_slice(&chunk);
    }
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// Accept bare hosts like `ollama.com` (Ollama's own examples omit the scheme)
/// by defaulting to `https`; reject anything that isn't `http`/`https`.
fn normalize_url(raw: &str) -> Result<Url, ProxyError> {
    let candidate = if raw.contains("://") {
        raw.to_string()
    } else {
        format!("https://{raw}")
    };
    let url = Url::parse(&candidate)
        .map_err(|e| ProxyError::bad_request(&format!("invalid 'url': {e}")))?;
    match url.scheme() {
        "http" | "https" => Ok(url),
        other => Err(ProxyError::bad_request(&format!(
            "unsupported url scheme '{other}': only http and https are allowed"
        ))),
    }
}

/// Pull the text of the first `<title>…</title>`, with a few common HTML
/// entities decoded. `None` when there is no non-empty title.
fn extract_title(html: &str) -> Option<String> {
    let lower = html.to_ascii_lowercase();
    let open = lower.find("<title")?;
    let content_start = open + lower[open..].find('>')? + 1;
    let close = content_start + lower[content_start..].find("</title>")?;
    let raw = html.get(content_start..close)?.trim();
    if raw.is_empty() {
        None
    } else {
        Some(decode_entities(raw))
    }
}

/// Collect absolute `http`/`https` links from every `<a href=…>` (and any other
/// `href` attribute), resolved against `base`, de-duplicated, order-preserving,
/// capped at [`MAX_LINKS`]. Best-effort scan, not a full HTML parse.
fn extract_links(html: &str, base: &Url) -> Vec<String> {
    let lower = html.to_ascii_lowercase();
    let bytes = html.as_bytes();
    let mut out: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut cursor = 0;

    while let Some(rel) = lower[cursor..].find("href") {
        let after_href = cursor + rel + "href".len();
        cursor = after_href;

        // Expect optional whitespace, '=', optional whitespace, then the value.
        let mut i = after_href;
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] != b'=' {
            continue;
        }
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }

        let value = match bytes[i] {
            quote @ (b'"' | b'\'') => {
                let start = i + 1;
                match html[start..].find(quote as char) {
                    Some(end) => &html[start..start + end],
                    // Unclosed quote: skip this malformed attribute and keep
                    // scanning for later well-formed hrefs (best-effort).
                    None => continue,
                }
            }
            _ => {
                let start = i;
                let end = html[start..]
                    .find(|c: char| c.is_whitespace() || c == '>')
                    .map(|k| start + k)
                    .unwrap_or(html.len());
                &html[start..end]
            }
        };

        if let Some(absolute) = resolve_link(base, value.trim())
            && seen.insert(absolute.clone())
        {
            out.push(absolute);
            if out.len() >= MAX_LINKS {
                break;
            }
        }
    }

    out
}

/// Resolve a single `href` against `base`, keeping only `http`/`https` targets
/// and dropping fragment-only / `javascript:` / `mailto:` / empty values.
fn resolve_link(base: &Url, href: &str) -> Option<String> {
    if href.is_empty() || href.starts_with('#') {
        return None;
    }
    let lower = href.to_ascii_lowercase();
    if lower.starts_with("javascript:") || lower.starts_with("mailto:") || lower.starts_with("tel:")
    {
        return None;
    }
    let joined = base.join(href).ok()?;
    match joined.scheme() {
        "http" | "https" => Some(joined.into()),
        _ => None,
    }
}

/// Decode the handful of HTML entities that commonly show up in `<title>`.
fn decode_entities(text: &str) -> String {
    text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&nbsp;", " ")
}

#[cfg(test)]
#[path = "../../tests/unit/web.rs"]
mod tests;
