//! `POST /api/web_fetch` — fetch a single URL and return it as
//! `{title, content, links}`, matching Ollama's web-fetch shape
//! (`api-docs/future/ollama/capabilities/web-search.md`).
//!
//! Unlike the rest of the proxy this talks to an arbitrary, user-supplied URL —
//! NOT LM Studio. It therefore uses its own client WITHOUT the LM Studio auth
//! header (the `--lmstudio-token` must never leak to third-party hosts) and
//! only allows `http`/`https`.

use std::collections::HashSet;
use std::sync::OnceLock;
use std::time::Duration;

use axum::response::Response;
use futures_util::StreamExt;
use serde_json::{Value, json};
use url::Url;

use crate::error::ProxyError;
use crate::http::json_response;

const WEB_FETCH_TIMEOUT_SECONDS: u64 = 30;
const WEB_FETCH_REDIRECT_LIMIT: usize = 10;
/// Cap the returned `links` array so a link-heavy page can't bloat the response.
const MAX_LINKS: usize = 100;
/// Cap the fetched body so a hostile/huge response can't exhaust memory.
const MAX_RESPONSE_BYTES: usize = 10 * 1024 * 1024;

/// Dedicated client for outbound web fetches: no default auth header (so the LM
/// Studio token never leaks to third-party hosts), a bounded timeout, a capped
/// redirect chain, and a descriptive user-agent.
fn web_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .user_agent(concat!("ollama-lmstudio-proxy/", env!("CARGO_PKG_VERSION")))
            .timeout(Duration::from_secs(WEB_FETCH_TIMEOUT_SECONDS))
            .redirect(reqwest::redirect::Policy::limited(WEB_FETCH_REDIRECT_LIMIT))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new())
    })
}

/// Fetch `body.url`, returning `{title, content (markdown), links}`.
pub async fn handle_web_fetch(body: Value) -> Result<Response, ProxyError> {
    let raw_url = body
        .get("url")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| ProxyError::bad_request("Missing 'url' field"))?;

    let url = normalize_url(raw_url)?;

    let response =
        web_client().get(url.clone()).send().await.map_err(|e| {
            ProxyError::new(format!("web_fetch: request to '{url}' failed: {e}"), 502)
        })?;

    let status = response.status();
    if !status.is_success() {
        return Err(ProxyError::new(
            format!("web_fetch: '{url}' returned status {status}"),
            502,
        ));
    }

    // Reject an over-large declared body up front; the streamed read below
    // enforces the same cap for responses that omit Content-Length.
    if let Some(len) = response.content_length()
        && len > MAX_RESPONSE_BYTES as u64
    {
        return Err(ProxyError::new(
            format!("web_fetch: '{url}' body too large ({len} bytes)"),
            502,
        ));
    }

    // The post-redirect URL is the correct base for resolving relative links.
    let final_url = response.url().clone();
    let html = read_body_capped(response, &url).await?;

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
