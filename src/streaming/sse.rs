use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;

use crate::config::get_runtime_config;
use crate::constants::{
    ERROR_CANCELLED, ERROR_TIMEOUT, LOG_PREFIX_CONN, LOG_PREFIX_SUCCESS, SSE_DATA_PREFIX,
    SSE_DONE_MESSAGE, SSE_MESSAGE_BOUNDARY,
};
use crate::error::ProxyError;
use crate::lmstudio::response::{TimingInfo, estimate_tokens_from_bytes};
use crate::logging::log_timed;
use crate::streaming::chunks::{
    ChunkProcessingState, FinalChunkParams, create_cancellation_chunk, create_final_chunk,
    create_ollama_streaming_chunk, extract_first_choice, process_choice_delta, send_chunk,
    send_chunk_and_close_channel, send_error_and_close,
};
use crate::streaming::native::{
    NativeChatEnd, NativeEvent, map_native_event, parse_native_sse_message,
};
use crate::streaming::recovery::recover_json_from_chunk;
use crate::streaming::response::{StreamContentType, create_streaming_response};

static STREAM_COUNTER: AtomicU64 = AtomicU64::new(0);

const STREAM_START_LOADING_THRESHOLD_MS: u128 = 500;

/// Status-gate a to-be-streamed upstream response before any SSE parsing starts.
///
/// A non-2xx status means the body is a plain JSON error, not an SSE stream —
/// parsing it as one yields zero events and a silently fabricated `done:true`
/// 200. Shared by both the v0 (OpenAI-compat) and native streaming entry points.
async fn reject_pre_stream_error(
    lm_studio_response: reqwest::Response,
) -> Result<reqwest::Response, ProxyError> {
    let status = lm_studio_response.status();
    if status.is_success() {
        return Ok(lm_studio_response);
    }

    let body_text = lm_studio_response.text().await.unwrap_or_default();
    let message = serde_json::from_str::<Value>(&body_text)
        .ok()
        .and_then(|v| match v.get("error") {
            Some(Value::Object(obj)) => obj
                .get("message")
                .and_then(|m| m.as_str())
                .map(|s| s.to_string()),
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        })
        .unwrap_or_else(|| format!("LM Studio error: {}", status));
    Err(ProxyError::new(message, status.as_u16()))
}

pub async fn handle_streaming_response(
    lm_studio_response: reqwest::Response,
    is_chat_endpoint: bool,
    ollama_model_name: &str,
    start_time: Instant,
    cancellation_token: CancellationToken,
    stream_timeout_seconds: u64,
    prompt_tokens_estimate: u64,
) -> Result<axum::response::Response, ProxyError> {
    let lm_studio_response = reject_pre_stream_error(lm_studio_response).await?;

    let runtime_config = get_runtime_config();
    let ollama_model_name = ollama_model_name.to_string();
    let (tx, rx) = mpsc::unbounded_channel::<Result<bytes::Bytes, std::io::Error>>();

    let stream_id = STREAM_COUNTER.fetch_add(1, Ordering::Relaxed) % 1_000_000;
    let model_loading_start = Instant::now();

    let model_clone_for_task = ollama_model_name.clone();
    let token_clone = cancellation_token.clone();

    tokio::spawn(async move {
        let mut stream = lm_studio_response.bytes_stream();
        let mut sse_buffer = String::with_capacity(runtime_config.max_buffer_size.min(1024 * 1024));
        let mut chunk_count = 0u64;
        // Streamed output text length (content + thinking) — the token-count
        // estimates derive from this, not from the SSE chunk count.
        let mut streamed_chars = 0usize;
        let mut chunk_state = ChunkProcessingState::default();
        let mut first_chunk_received = false;
        let mut recovery_buffer = String::new();
        let enable_chunk_recovery = runtime_config.enable_chunk_recovery;

        let stream_result = 'stream_loop: loop {
            tokio::select! {
                biased;
                _ = token_clone.cancelled() => {
                    let cancellation_chunk = create_cancellation_chunk(
                        &model_clone_for_task,
                        start_time.elapsed(),
                        prompt_tokens_estimate,
                        estimate_tokens_from_bytes(streamed_chars),
                        is_chat_endpoint,
                    );
                    send_chunk_and_close_channel(&tx, cancellation_chunk).await;
                    break 'stream_loop Err(ERROR_CANCELLED.to_string());
                }

                chunk_result = timeout(Duration::from_secs(stream_timeout_seconds), stream.next()) => {
                    match chunk_result {
                        Ok(Some(Ok(bytes_chunk))) => {
                            if !first_chunk_received {
                                first_chunk_received = true;
                                let time_to_first_chunk = start_time.elapsed();

                                if time_to_first_chunk.as_millis() > STREAM_START_LOADING_THRESHOLD_MS {
                                    log_timed(LOG_PREFIX_SUCCESS, &format!("{} loaded", model_clone_for_task), model_loading_start);
                                }
                            }

                            if let Ok(chunk_str) = std::str::from_utf8(&bytes_chunk) {
                                sse_buffer.push_str(chunk_str);

                                let mut cursor = 0;
                                while let Some(rel_pos) = sse_buffer[cursor..].find(SSE_MESSAGE_BOUNDARY) {
                                    let boundary_pos = cursor + rel_pos;
                                    let message_text = &sse_buffer[cursor..boundary_pos];
                                    cursor = boundary_pos + SSE_MESSAGE_BOUNDARY.len();

                                    if message_text.bytes().all(|b| b.is_ascii_whitespace()) { continue; }

                                    if let Some(data_content) = message_text.strip_prefix(SSE_DATA_PREFIX) {
                                        if data_content.trim() == SSE_DONE_MESSAGE {
                                            break 'stream_loop Ok(());
                                        }

                                        match serde_json::from_str::<Value>(data_content) {
                                            Ok(lm_studio_json_chunk) => {
                                                let mut content_to_send = String::new();
                                                let mut thinking_to_send = String::new();

                                                if let Some(choice) = extract_first_choice(&lm_studio_json_chunk)
                                                    && let Some(delta_payload) = process_choice_delta(choice, &mut chunk_state) {
                                                        content_to_send = delta_payload.content;
                                                        thinking_to_send = delta_payload.thinking;
                                                    }

                                                if !content_to_send.is_empty() || !thinking_to_send.is_empty() {
                                                    streamed_chars += content_to_send.len() + thinking_to_send.len();
                                                    let ollama_chunk = create_ollama_streaming_chunk(
                                                        &model_clone_for_task,
                                                        &content_to_send,
                                                        is_chat_endpoint,
                                                        false,
                                                        None,
                                                        &thinking_to_send,
                                                    );
                                                    chunk_count += 1;
                                                    if !send_chunk(&tx, &ollama_chunk).await {
                                                        break 'stream_loop Ok(());
                                                    }
                                                }
                                            }
                                            Err(e) => {
                                                if enable_chunk_recovery {
                                                    log::warn!("SSE parsing error (attempting recovery): {}", e);
                                                    if let Some(recovered_json) = recover_json_from_chunk(data_content) {
                                                        log::info!("Successfully recovered chunk data");
                                                        let mut content_to_send = String::new();
                                                        let mut thinking_to_send = String::new();

                                                        if let Some(choice) = extract_first_choice(&recovered_json)
                                                            && let Some(delta_payload) = process_choice_delta(choice, &mut chunk_state) {
                                                                content_to_send = delta_payload.content;
                                                                thinking_to_send = delta_payload.thinking;
                                                            }

                                                        if !content_to_send.is_empty() || !thinking_to_send.is_empty() {
                                                            streamed_chars += content_to_send.len() + thinking_to_send.len();
                                                            let ollama_chunk = create_ollama_streaming_chunk(
                                                                &model_clone_for_task,
                                                                &content_to_send,
                                                                is_chat_endpoint,
                                                                false,
                                                                None,
                                                                &thinking_to_send,
                                                            );
                                                            chunk_count += 1;
                                                            if !send_chunk(&tx, &ollama_chunk).await {
                                                                break 'stream_loop Ok(());
                                                            }
                                                        }
                                                    } else {
                                                        log::error!("SSE parsing error (recovery failed): {}", e);
                                                        recovery_buffer.push_str(data_content);
                                                        recovery_buffer.push_str(SSE_MESSAGE_BOUNDARY);
                                                    }
                                                } else {
                                                    // Spec: mid-stream parse failures with recovery off must
                                                    // surface a bare {"error":"…"} NDJSON line and end the
                                                    // stream (no trailing done:true).
                                                    let message = format!("SSE parsing error: {}", e);
                                                    log::error!("{}", message);
                                                    send_error_and_close(&tx, &message).await;
                                                    break 'stream_loop Err(message);
                                                }
                                            }
                                        }
                                    } else {
                                         log::warn!("SSE format: non-standard line: {}", message_text);
                                    }
                                }
                                if cursor > 0 {
                                    sse_buffer.drain(..cursor);
                                }
                            } else {
                                send_error_and_close(&tx, "invalid UTF-8 in stream").await;
                                break 'stream_loop Err("invalid UTF-8".to_string());
                            }
                        }
                        Ok(Some(Err(e))) => {
                            send_error_and_close(&tx, &format!("streaming error: {}", e)).await;
                            break 'stream_loop Err(format!("network error: {}", e));
                        }
                        Ok(None) => {
                            log::warn!("stream ended without [DONE]");
                            if enable_chunk_recovery && !recovery_buffer.is_empty() {
                                log::info!("Attempting to recover from remaining buffer data");
                                if let Some(recovered_json) = recover_json_from_chunk(&recovery_buffer) {
                                    log::info!("Successfully recovered data from remaining buffer");
                                    let mut content_to_send = String::new();
                                    let mut thinking_to_send = String::new();

                                    if let Some(choice) = extract_first_choice(&recovered_json)
                                        && let Some(delta_payload) = process_choice_delta(choice, &mut chunk_state) {
                                            content_to_send = delta_payload.content;
                                            thinking_to_send = delta_payload.thinking;
                                        }

                                    if !content_to_send.is_empty() || !thinking_to_send.is_empty() {
                                        streamed_chars += content_to_send.len() + thinking_to_send.len();
                                        let ollama_chunk = create_ollama_streaming_chunk(
                                            &model_clone_for_task,
                                            &content_to_send,
                                            is_chat_endpoint,
                                            false,
                                            None,
                                            &thinking_to_send,
                                        );
                                        chunk_count += 1;
                                        if !send_chunk(&tx, &ollama_chunk).await {
                                            break 'stream_loop Ok(());
                                        }
                                    }
                                }
                            }
                            break 'stream_loop Ok(());
                        }
                        Err(_) => {
                            send_error_and_close(&tx, ERROR_TIMEOUT).await;
                            break 'stream_loop Err(ERROR_TIMEOUT.to_string());
                        }
                    }
                }
            }
        };

        if stream_result.is_ok() && !token_clone.is_cancelled() {
            // Assembled tool calls go out once, in their own done:false chunk;
            // the final chunk stays tool-free (matching real Ollama).
            if is_chat_endpoint && let Some(tool_calls) = chunk_state.take_tool_calls() {
                let tool_chunk = create_ollama_streaming_chunk(
                    &model_clone_for_task,
                    "",
                    true,
                    false,
                    Some(&tool_calls),
                    "",
                );
                chunk_count += 1;
                send_chunk(&tx, &tool_chunk).await;
            }
            let final_chunk = create_final_chunk(FinalChunkParams {
                model_name: &model_clone_for_task,
                duration: start_time.elapsed(),
                prompt_tokens_estimate,
                output_tokens_estimate: estimate_tokens_from_bytes(streamed_chars),
                is_chat: is_chat_endpoint,
                done_reason: chunk_state.finish_reason(),
            });
            send_chunk_and_close_channel(&tx, final_chunk).await;
        }

        log_timed(
            LOG_PREFIX_CONN,
            &format!("stream [{}] completed | {} chunks", stream_id, chunk_count),
            start_time,
        );
    });

    create_streaming_response(rx, StreamContentType::Ndjson)
}

/// Streaming driver for LM Studio's native `/api/v1/chat` SSE stream.
///
/// Mirrors [`handle_streaming_response`]'s byte-buffering, cancellation and
/// timeout structure, but the native wire format uses named events
/// (`event: <type>\ndata: <json>`) instead of bare `data:` lines. Each SSE block
/// is parsed with [`parse_native_sse_message`] and dispatched via
/// [`map_native_event`]: deltas emit intermediate Ollama chunks, a recoverable
/// `error` event is buffered and folded into the final chunk's `warning` (only a
/// stream that dies without `chat.end` surfaces it as a terminal error line), and
/// `chat.end` drives the final timing chunk from the native `stats` block. Native
/// is always chat-shaped, so chunk recovery (OpenAI-specific) is intentionally
/// skipped.
/// Request-derived extras for the native stream's final chunk: the output cap
/// feeding the `done_reason` heuristic, the dropped-field warning, and the
/// input-length token estimate for stats fallbacks.
#[derive(Default)]
pub struct NativeFinalizeOptions {
    pub requested_max_tokens: Option<u64>,
    pub warning: Option<String>,
    pub prompt_tokens_estimate: u64,
}

pub async fn handle_native_streaming_response(
    lm_studio_response: reqwest::Response,
    ollama_model_name: &str,
    start_time: Instant,
    cancellation_token: CancellationToken,
    stream_timeout_seconds: u64,
    finalize: NativeFinalizeOptions,
) -> Result<axum::response::Response, ProxyError> {
    let lm_studio_response = reject_pre_stream_error(lm_studio_response).await?;

    let runtime_config = get_runtime_config();
    let ollama_model_name = ollama_model_name.to_string();
    let (tx, rx) = mpsc::unbounded_channel::<Result<bytes::Bytes, std::io::Error>>();

    let stream_id = STREAM_COUNTER.fetch_add(1, Ordering::Relaxed) % 1_000_000;
    let model_loading_start = Instant::now();

    let model_clone_for_task = ollama_model_name.clone();
    let token_clone = cancellation_token.clone();

    tokio::spawn(async move {
        let mut stream = lm_studio_response.bytes_stream();
        let mut sse_buffer = String::with_capacity(runtime_config.max_buffer_size.min(1024 * 1024));
        let mut chunk_count = 0u64;
        // Streamed output text length (content + thinking), for the stats
        // fallbacks when chat.end never arrives (and the cancel chunk).
        let mut streamed_chars = 0usize;
        let mut chunk_state = ChunkProcessingState::default();
        let mut first_chunk_received = false;
        // Captured from `chat.end` so the final done chunk can carry native stats.
        let mut chat_end: Option<NativeChatEnd> = None;
        // `model_load.end` load time, used when the chat.end stats omit theirs.
        let mut observed_load_seconds: Option<f64> = None;
        // Native `error` events are recoverable (chat.end still follows with
        // whatever was generated); buffer their messages so they fold into the
        // final chunk's `warning` instead of a terminal error line.
        let mut error_messages: Vec<String> = Vec::new();

        let stream_result = 'stream_loop: loop {
            tokio::select! {
                biased;
                _ = token_clone.cancelled() => {
                    let cancellation_chunk = create_cancellation_chunk(
                        &model_clone_for_task,
                        start_time.elapsed(),
                        finalize.prompt_tokens_estimate,
                        estimate_tokens_from_bytes(streamed_chars),
                        true,
                    );
                    send_chunk_and_close_channel(&tx, cancellation_chunk).await;
                    break 'stream_loop Err(ERROR_CANCELLED.to_string());
                }

                chunk_result = timeout(Duration::from_secs(stream_timeout_seconds), stream.next()) => {
                    match chunk_result {
                        Ok(Some(Ok(bytes_chunk))) => {
                            if !first_chunk_received {
                                first_chunk_received = true;
                                let time_to_first_chunk = start_time.elapsed();

                                if time_to_first_chunk.as_millis() > STREAM_START_LOADING_THRESHOLD_MS {
                                    log_timed(LOG_PREFIX_SUCCESS, &format!("{} loaded", model_clone_for_task), model_loading_start);
                                }
                            }

                            if let Ok(chunk_str) = std::str::from_utf8(&bytes_chunk) {
                                sse_buffer.push_str(chunk_str);

                                let mut cursor = 0;
                                while let Some(rel_pos) = sse_buffer[cursor..].find(SSE_MESSAGE_BOUNDARY) {
                                    let boundary_pos = cursor + rel_pos;
                                    let message_text = &sse_buffer[cursor..boundary_pos];
                                    cursor = boundary_pos + SSE_MESSAGE_BOUNDARY.len();

                                    if message_text.bytes().all(|b| b.is_ascii_whitespace()) { continue; }

                                    let Some((event_type, data)) = parse_native_sse_message(message_text) else {
                                        log::warn!("native SSE: unparsable block: {}", message_text);
                                        continue;
                                    };

                                    match map_native_event(&event_type, &data, &mut chunk_state) {
                                        NativeEvent::Delta(payload) => {
                                            if payload.content.is_empty() && payload.thinking.is_empty() {
                                                continue;
                                            }
                                            streamed_chars += payload.content.len() + payload.thinking.len();
                                            let ollama_chunk = create_ollama_streaming_chunk(
                                                &model_clone_for_task,
                                                &payload.content,
                                                true,
                                                false,
                                                None,
                                                &payload.thinking,
                                            );
                                            chunk_count += 1;
                                            if !send_chunk(&tx, &ollama_chunk).await {
                                                break 'stream_loop Ok(());
                                            }
                                        }
                                        NativeEvent::End(end) => {
                                            chat_end = Some(end);
                                            break 'stream_loop Ok(());
                                        }
                                        NativeEvent::Error(err) => {
                                            // The Ollama `{"error":…}` line is a terminal sentinel:
                                            // clients raise and stop iterating the moment they parse it.
                                            // A native error is recoverable (chat.end still follows with
                                            // whatever was generated), so buffer it and fold it into the
                                            // final chunk's `warning`; only a stream that dies without a
                                            // chat.end emits it as a terminal line.
                                            let message = err.to_message();
                                            log::error!("native stream error: {}", message);
                                            error_messages.push(message);
                                        }
                                        NativeEvent::ModelLoaded(seconds) => {
                                            observed_load_seconds = Some(seconds);
                                        }
                                        NativeEvent::Ignore => {}
                                    }
                                }
                                if cursor > 0 {
                                    sse_buffer.drain(..cursor);
                                }
                            } else {
                                send_error_and_close(&tx, "invalid UTF-8 in stream").await;
                                break 'stream_loop Err("invalid UTF-8".to_string());
                            }
                        }
                        Ok(Some(Err(e))) => {
                            send_error_and_close(&tx, &format!("streaming error: {}", e)).await;
                            break 'stream_loop Err(format!("network error: {}", e));
                        }
                        Ok(None) => {
                            log::warn!("native stream ended without chat.end");
                            break 'stream_loop Ok(());
                        }
                        Err(_) => {
                            send_error_and_close(&tx, ERROR_TIMEOUT).await;
                            break 'stream_loop Err(ERROR_TIMEOUT.to_string());
                        }
                    }
                }
            }
        };

        if stream_result.is_ok() && !token_clone.is_cancelled() {
            if chat_end.is_none() && !error_messages.is_empty() {
                // A recoverable error that never reached chat.end is now
                // genuinely terminal: emit the Ollama error sentinel and no
                // fabricated done:true (mirrors the v0 path's contract).
                send_error_and_close(&tx, &error_messages.join("; ")).await;
            } else {
                // Same contract as the v0 path: assembled tool calls in one
                // done:false chunk, final chunk tool-free.
                if let Some(tool_calls) = chunk_state.take_tool_calls() {
                    let tool_chunk = create_ollama_streaming_chunk(
                        &model_clone_for_task,
                        "",
                        true,
                        false,
                        Some(&tool_calls),
                        "",
                    );
                    chunk_count += 1;
                    send_chunk(&tx, &tool_chunk).await;
                }
                let final_chunk = build_native_final_chunk(
                    &model_clone_for_task,
                    chat_end.as_ref(),
                    start_time,
                    streamed_chars,
                    observed_load_seconds,
                    &finalize,
                    &error_messages,
                );
                send_chunk_and_close_channel(&tx, final_chunk).await;
            }
        }

        log_timed(
            LOG_PREFIX_CONN,
            &format!(
                "native stream [{}] completed | {} chunks",
                stream_id, chunk_count
            ),
            start_time,
        );
    });

    create_streaming_response(rx, StreamContentType::Ndjson)
}

/// Build the final `done:true` chunk for the native streaming path.
///
/// When a `chat.end` was seen, timing comes from its native `stats` block via
/// [`TimingInfo::from_native_stats`] (with the `model_load.end` event's load
/// time as fallback when the stats omit theirs) and `done_reason` from the
/// parsed end event; otherwise (stream ended early) it falls back to the
/// wall-clock heuristics in [`create_final_chunk`].
fn build_native_final_chunk(
    model_name: &str,
    chat_end: Option<&NativeChatEnd>,
    start_time: Instant,
    streamed_chars: usize,
    observed_load_seconds: Option<f64>,
    finalize: &NativeFinalizeOptions,
    recovered_errors: &[String],
) -> Value {
    let output_tokens_estimate = estimate_tokens_from_bytes(streamed_chars);
    let mut chunk = match chat_end {
        Some(end) => {
            let timing = TimingInfo::from_native_stats(
                &end.result,
                start_time,
                finalize.prompt_tokens_estimate,
                output_tokens_estimate.max(1),
                observed_load_seconds,
            );

            let mut chunk = create_ollama_streaming_chunk(model_name, "", true, true, None, "");

            if let Some(obj) = chunk.as_object_mut() {
                // Proxy heuristic — the native API exposes no finish-reason.
                let done_reason = crate::lmstudio::native_chat::native_done_reason_heuristic(
                    timing.eval_count,
                    finalize.requested_max_tokens,
                );
                obj.insert("done_reason".to_string(), json!(done_reason));
                obj.insert("total_duration".to_string(), json!(timing.total_duration));
                obj.insert("load_duration".to_string(), json!(timing.load_duration));
                obj.insert(
                    "prompt_eval_count".to_string(),
                    json!(timing.prompt_eval_count),
                );
                obj.insert(
                    "prompt_eval_duration".to_string(),
                    json!(timing.prompt_eval_duration),
                );
                obj.insert("eval_count".to_string(), json!(timing.eval_count));
                obj.insert("eval_duration".to_string(), json!(timing.eval_duration));
            }

            chunk
        }
        None => create_final_chunk(FinalChunkParams {
            model_name,
            duration: start_time.elapsed(),
            prompt_tokens_estimate: finalize.prompt_tokens_estimate,
            output_tokens_estimate,
            is_chat: true,
            done_reason: None,
        }),
    };

    // Fold the dropped-field notice and any recovered mid-stream error into one
    // `warning`. A recovered native error can't be a terminal `{"error"}` line
    // (chat.end proved the request finished), so it rides along here instead.
    let recovered_notice = (!recovered_errors.is_empty()).then(|| {
        format!(
            "recovered from mid-stream error: {}",
            recovered_errors.join("; ")
        )
    });
    let combined_warning = match (&finalize.warning, &recovered_notice) {
        (Some(drop), Some(rec)) => format!("{drop}; {rec}"),
        (Some(drop), None) => drop.clone(),
        (None, Some(rec)) => rec.clone(),
        (None, None) => String::new(),
    };
    if !combined_warning.is_empty()
        && let Some(obj) = chunk.as_object_mut()
    {
        obj.insert("warning".to_string(), json!(combined_warning));
    }

    chunk
}

/// Wire protocol of a passthrough stream, for shaping proxy-injected
/// cancel/timeout/error frames. The Ollama bare `data:{"error":...}` line is
/// wrong on every passthrough surface: Anthropic clients dispatch on `event:`
/// names, OpenAI clients expect a typed `error` object, `/v1/responses`
/// clients expect a `response.failed` event, and the native `/api/v1` stream
/// uses named `event: error` blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassthroughProtocol {
    /// OpenAI-style SSE: `/v1/chat/completions`, `/v1/completions`, `/api/v0/*`.
    OpenAi,
    /// Anthropic messages: `/v1/messages`.
    Anthropic,
    /// OpenAI responses API: `/v1/responses`.
    Responses,
    /// LM Studio native versioned API: `/api/v1/*` and later.
    NativeV1,
}

impl PassthroughProtocol {
    pub fn from_endpoint(endpoint: &str) -> Self {
        if endpoint.starts_with("/v1/messages") {
            Self::Anthropic
        } else if endpoint.starts_with("/v1/responses") {
            Self::Responses
        } else if endpoint.starts_with("/api/") && !endpoint.starts_with("/api/v0/") {
            Self::NativeV1
        } else {
            Self::OpenAi
        }
    }

    /// Frame a proxy-injected error (cancel/timeout/upstream failure) as a
    /// complete SSE block in this protocol's error shape.
    fn frame_error(&self, message: &str) -> String {
        match self {
            Self::OpenAi => format!(
                "data: {}\n\n",
                json!({"error": {"message": message, "type": "server_error"}})
            ),
            Self::Anthropic => format!(
                "event: error\ndata: {}\n\n",
                json!({"type": "error", "error": {"type": "api_error", "message": message}})
            ),
            Self::Responses => format!(
                "event: response.failed\ndata: {}\n\n",
                json!({
                    "type": "response.failed",
                    "response": {"status": "failed", "error": {"code": "server_error", "message": message}}
                })
            ),
            Self::NativeV1 => format!(
                "event: error\ndata: {}\n\n",
                json!({"type": "error", "error": {"type": "internal_error", "message": message}})
            ),
        }
    }
}

pub async fn handle_passthrough_streaming_response(
    response: reqwest::Response,
    protocol: PassthroughProtocol,
    cancellation_token: CancellationToken,
    stream_timeout_seconds: u64,
) -> Result<axum::response::Response, ProxyError> {
    let (tx, rx) = mpsc::unbounded_channel::<Result<bytes::Bytes, std::io::Error>>();
    let stream_id = STREAM_COUNTER.fetch_add(1, Ordering::Relaxed) % 1_000_000;
    let start_time = Instant::now();

    tokio::spawn(async move {
        let mut stream = response.bytes_stream();
        let mut chunk_count = 0u64;

        loop {
            tokio::select! {
                biased;
                _ = cancellation_token.cancelled() => {
                    let cancel_data = protocol.frame_error(ERROR_CANCELLED);
                    let _ = tx.send(Ok(bytes::Bytes::from(cancel_data)));
                    break;
                }
                chunk_result = timeout(Duration::from_secs(stream_timeout_seconds), stream.next()) => {
                    match chunk_result {
                        Ok(Some(Ok(chunk))) => {
                            chunk_count += 1;
                            if tx.send(Ok(chunk)).is_err() {
                                break;
                            }
                        }
                        Ok(Some(Err(e))) => {
                            let error_data = protocol.frame_error(&format!("streaming error: {}", e));
                            let _ = tx.send(Ok(bytes::Bytes::from(error_data)));
                            break;
                        }
                        Ok(None) => break,
                        Err(_) => {
                            let timeout_data = protocol.frame_error(ERROR_TIMEOUT);
                            let _ = tx.send(Ok(bytes::Bytes::from(timeout_data)));
                            break;
                        }
                    }
                }
            }
        }

        log_timed(
            LOG_PREFIX_CONN,
            &format!(
                "passthrough stream [{}] | {} chunks",
                stream_id, chunk_count
            ),
            start_time,
        );
    });

    create_streaming_response(rx, StreamContentType::Sse)
}

#[cfg(test)]
#[path = "../../tests/unit/streaming_sse.rs"]
mod tests;
