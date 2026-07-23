//! Short-circuit handlers for promptless/messageless `/api/generate` and
//! `/api/chat` requests: `keep_alive: 0` unloads on both endpoints; an empty
//! payload with `keep_alive != 0` warms on both (upstream api.md documents
//! the "Load a model" sample for each).
//!
//! Per the Ollama spec (`api-docs/ollama/api/generate.md` and `chat.md`), both
//! `GenerateRequest` and `ChatRequest` only require the `model` field. The
//! documented invocation `{"model":"x","keep_alive":0}` is an unload-only
//! request: no inference is performed, the model is unloaded, and a single
//! `done:true` response chunk is returned. The "Load model" sample
//! (`{"model":"x"}` / `{"model":"x","messages":[]}`, no `keep_alive:0`) is the
//! mirror case: a load/warm no-op with no inference, `done_reason:"load"`.
//!
//! This module builds the response envelope and emits it in either NDJSON
//! (stream:true) or single-JSON (stream:false) form. The actual unload is
//! kicked off via `spawn_model_unload_if_needed`, and the warm via
//! `trigger_model_loading_for_ollama`, exactly as either would run after a
//! normal inference call.
//!
//! A body is "unload-only" when `keep_alive == 0` AND the per-endpoint payload
//! field is missing or empty (generate: `prompt`; chat: `messages`) — this
//! applies to both endpoints (`is_generate_unload_only`/`is_chat_unload_only`).
//! "Warm-only" is the same empty-payload condition with `keep_alive != 0`
//! (`is_generate_warm_only`/`is_chat_warm_only`). With a non-empty payload the
//! request still flows through the regular inference path and the unload races
//! the inference response — see the existing `keep_alive_zero_accepted`
//! integration tests.
//!
//! Unload's `done_reason:"unload"` comes from upstream ollama's `docs/api.md`,
//! mirrored at `api-docs/ollama/repo/api.md` (the docs-site `llms.txt` that
//! feeds the rest of `api-docs/ollama/` does not list that page) — both the
//! generate and chat "Unload a model" samples carry it, and neither carries
//! any duration/eval field, matching `build_done_chunk` below.
//!
//! Warm's `done_reason:"load"` is that same doc's chat sample verbatim; its
//! generate sample omits the field, but live Ollama 0.31.1 emits a real
//! `done_reason:"load"` for generate too, so `build_warm_chunk` sends it
//! unconditionally for both endpoints.

use std::sync::Arc;

use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::api::RequestContext;
use crate::api::ollama::status_stream::stream_status_messages;
use crate::api::retry::trigger_model_loading_for_ollama;
use crate::error::ProxyError;
use crate::http::json_response;
use crate::lmstudio::keep_alive::spawn_model_unload_if_needed;
use crate::model::ModelResolver;

pub struct UnloadOnlyCall<'a> {
    pub context: &'a RequestContext<'a>,
    pub model_resolver: Arc<ModelResolver>,
    pub ollama_model_name: &'a str,
    pub keep_alive_seconds: Option<i64>,
    pub is_chat: bool,
    pub stream: bool,
    pub cancellation_token: CancellationToken,
}

/// Resolves the model (cheaply — no load triggered), spawns the unload, and
/// returns a `done:true` response in the requested wire format.
pub async fn respond_unload_only(
    call: UnloadOnlyCall<'_>,
) -> Result<axum::response::Response, ProxyError> {
    let UnloadOnlyCall {
        context,
        model_resolver,
        ollama_model_name,
        keep_alive_seconds,
        is_chat,
        stream,
        cancellation_token,
    } = call;

    // Verify the model resolves — a 404 here matches what a normal request
    // would return, rather than silently "unloading" an unknown name.
    model_resolver
        .resolve_model_name(ollama_model_name, context.client, cancellation_token)
        .await?;

    spawn_model_unload_if_needed(
        context.client.clone(),
        context.lmstudio_url.to_string(),
        model_resolver,
        ollama_model_name.to_string(),
        keep_alive_seconds,
        // No streaming response to wait for — unload immediately.
        0,
    );

    let payload = build_done_chunk(ollama_model_name, is_chat);

    if stream {
        stream_status_messages(
            vec![payload],
            "failed to create unload-only streaming response",
        )
    } else {
        Ok(json_response(&payload))
    }
}

fn build_done_chunk(ollama_model_name: &str, is_chat: bool) -> Value {
    let mut payload = json!({
        "model": ollama_model_name,
        "created_at": chrono::Utc::now().to_rfc3339(),
        "done": true,
        "done_reason": "unload",
    });

    if let Some(obj) = payload.as_object_mut() {
        if is_chat {
            obj.insert(
                "message".to_string(),
                json!({"role": "assistant", "content": ""}),
            );
        } else {
            obj.insert("response".to_string(), json!(""));
        }
    }

    payload
}

/// Generate: unload-only when `keep_alive == 0` AND `prompt` is missing or
/// an empty string.
pub fn is_generate_unload_only(body: &Value, keep_alive_seconds: Option<i64>) -> bool {
    if !matches!(keep_alive_seconds, Some(0)) {
        return false;
    }
    match body.get("prompt") {
        None | Some(Value::Null) => true,
        Some(Value::String(s)) => s.is_empty(),
        _ => false,
    }
}

/// Chat: unload-only when `keep_alive == 0` AND `messages` is missing or an
/// empty array.
pub fn is_chat_unload_only(body: &Value, keep_alive_seconds: Option<i64>) -> bool {
    if !matches!(keep_alive_seconds, Some(0)) {
        return false;
    }
    match body.get("messages") {
        None | Some(Value::Null) => true,
        Some(Value::Array(arr)) => arr.is_empty(),
        _ => false,
    }
}

/// Generate: warm-only when `prompt` is missing or an empty string AND
/// `keep_alive != 0` (the `keep_alive == 0` case is unload-only, above).
pub fn is_generate_warm_only(body: &Value, keep_alive_seconds: Option<i64>) -> bool {
    if matches!(keep_alive_seconds, Some(0)) {
        return false;
    }
    match body.get("prompt") {
        None | Some(Value::Null) => true,
        Some(Value::String(s)) => s.is_empty(),
        _ => false,
    }
}

/// Chat: warm-only when `messages` is missing or an empty array AND
/// `keep_alive != 0` — the chat twin of `is_generate_warm_only`.
pub fn is_chat_warm_only(body: &Value, keep_alive_seconds: Option<i64>) -> bool {
    if matches!(keep_alive_seconds, Some(0)) {
        return false;
    }
    match body.get("messages") {
        None | Some(Value::Null) => true,
        Some(Value::Array(arr)) => arr.is_empty(),
        _ => false,
    }
}

pub struct WarmOnlyCall<'a> {
    pub context: &'a RequestContext<'a>,
    pub model_resolver: Arc<ModelResolver>,
    pub ollama_model_name: &'a str,
    pub is_chat: bool,
    pub stream: bool,
    pub cancellation_token: CancellationToken,
}

/// Resolves the model (404 parity with a normal request), warms/loads it via
/// the same path `/api/show` uses, and returns a `done:true,
/// done_reason:"load"` response in the requested wire format. No inference is
/// performed.
///
/// `keep_alive` on a warm-only request is not applied as a TTL here (this
/// path never reaches `apply_keep_alive_ttl`, which only runs on the
/// inference request) — a ceiling shared with `/api/show`'s identical
/// unconditional-warm call; a future TTL-aware warm would need to thread
/// `keep_alive_seconds` into a post-warm `spawn_model_unload_if_needed`.
pub async fn respond_warm_only(
    call: WarmOnlyCall<'_>,
) -> Result<axum::response::Response, ProxyError> {
    let WarmOnlyCall {
        context,
        model_resolver,
        ollama_model_name,
        is_chat,
        stream,
        cancellation_token,
    } = call;

    // Verify the model resolves — a 404 here matches what a normal request
    // would return, rather than silently "warming" an unknown name.
    model_resolver
        .resolve_model_name(
            ollama_model_name,
            context.client,
            cancellation_token.clone(),
        )
        .await?;

    trigger_model_loading_for_ollama(context, ollama_model_name, cancellation_token).await?;

    let payload = build_warm_chunk(ollama_model_name, is_chat);

    if stream {
        stream_status_messages(
            vec![payload],
            "failed to create warm-only streaming response",
        )
    } else {
        Ok(json_response(&payload))
    }
}

/// Live Ollama 0.31.1 omits every duration/eval field for a load-only
/// response and includes only `model`, `created_at`, the endpoint's empty
/// payload field, `done`, and a real `done_reason:"load"` — the same shape
/// `build_done_chunk`'s unload envelope emits, with `done_reason:"unload"`
/// instead. Chat carries an empty assistant `message` (upstream chat.md
/// "Load a model" sample); generate an empty `response`.
fn build_warm_chunk(ollama_model_name: &str, is_chat: bool) -> Value {
    let mut payload = json!({
        "model": ollama_model_name,
        "created_at": chrono::Utc::now().to_rfc3339(),
        "done": true,
        "done_reason": "load",
    });

    if let Some(obj) = payload.as_object_mut() {
        if is_chat {
            obj.insert(
                "message".to_string(),
                json!({"role": "assistant", "content": ""}),
            );
        } else {
            obj.insert("response".to_string(), json!(""));
        }
    }

    payload
}
