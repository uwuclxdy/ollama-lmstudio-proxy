//! Short-circuit handlers for promptless/messageless `/api/generate` and
//! `/api/chat` requests: `keep_alive: 0` unloads on both endpoints; a
//! promptless `/api/generate` with `keep_alive != 0` warms (`/api/chat` has
//! no warm-only equivalent yet — a messageless chat request still 400s via
//! the regular required-field validation; parity is a possible follow-up).
//!
//! Per the Ollama spec (`api-docs/ollama/api/generate.md` and `chat.md`), both
//! `GenerateRequest` and `ChatRequest` only require the `model` field. The
//! documented invocation `{"model":"x","keep_alive":0}` is an unload-only
//! request: no inference is performed, the model is unloaded, and a single
//! `done:true` response chunk is returned. `generate.md`'s "Load model" sample
//! (`{"model":"x"}`, no `keep_alive:0`) is the mirror case, generate-only: a
//! load/warm no-op with no inference, `done_reason:"load"`.
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
//! "Warm-only" is `is_generate_warm_only` alone: the same empty-`prompt`
//! condition but `keep_alive != 0`. With a non-empty payload the request
//! still flows through the regular inference path and the unload races the
//! inference response — see the existing `keep_alive_zero_accepted`
//! integration tests.
//!
//! Unload's `done_reason` is omitted: `api-docs/ollama/api/generate.md` and
//! `chat.md` only document `stop | length`, and the project's policy (see
//! `src/streaming/chunks.rs`) is to omit unknown reasons rather than fabricate
//! one. Warm is the documented exception — live Ollama 0.31.1 emits a real
//! `done_reason:"load"` for this case, so `build_warm_chunk` sends it.

use std::sync::Arc;
use std::time::Instant;

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
    pub start_time: Instant,
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
        start_time,
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

    let payload = build_done_chunk(ollama_model_name, is_chat, start_time);

    if stream {
        stream_status_messages(
            vec![payload],
            "failed to create unload-only streaming response",
        )
    } else {
        Ok(json_response(&payload))
    }
}

fn build_done_chunk(ollama_model_name: &str, is_chat: bool, start_time: Instant) -> Value {
    let total_duration_ns = start_time.elapsed().as_nanos() as u64;

    let mut payload = json!({
        "model": ollama_model_name,
        "created_at": chrono::Utc::now().to_rfc3339(),
        "done": true,
        "total_duration": total_duration_ns,
        "load_duration": 0u64,
        "prompt_eval_count": 0u64,
        "prompt_eval_duration": 0u64,
        "eval_count": 0u64,
        "eval_duration": 0u64,
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

pub struct GenerateWarmOnlyCall<'a> {
    pub context: &'a RequestContext<'a>,
    pub model_resolver: Arc<ModelResolver>,
    pub ollama_model_name: &'a str,
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
pub async fn respond_generate_warm_only(
    call: GenerateWarmOnlyCall<'_>,
) -> Result<axum::response::Response, ProxyError> {
    let GenerateWarmOnlyCall {
        context,
        model_resolver,
        ollama_model_name,
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

    let payload = build_warm_chunk(ollama_model_name);

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
/// response and includes only `model`, `created_at`, `response`, `done`, and
/// a real `done_reason:"load"` — unlike `build_done_chunk`'s unload envelope,
/// which zero-fills those fields and omits `done_reason` entirely.
fn build_warm_chunk(ollama_model_name: &str) -> Value {
    json!({
        "model": ollama_model_name,
        "created_at": chrono::Utc::now().to_rfc3339(),
        "response": "",
        "done": true,
        "done_reason": "load",
    })
}
