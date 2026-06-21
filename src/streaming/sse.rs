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
use crate::lmstudio::response::TimingInfo;
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

pub async fn handle_streaming_response(
    lm_studio_response: reqwest::Response,
    is_chat_endpoint: bool,
    ollama_model_name: &str,
    start_time: Instant,
    cancellation_token: CancellationToken,
    stream_timeout_seconds: u64,
) -> Result<axum::response::Response, ProxyError> {
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
                        chunk_count,
                        chunk_state.take_tool_calls(),
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
                                                let mut tool_calls_to_send: Option<Value> = None;

                                                if let Some(choice) = extract_first_choice(&lm_studio_json_chunk)
                                                    && let Some(delta_payload) = process_choice_delta(choice, &mut chunk_state) {
                                                        content_to_send = delta_payload.content;
                                                        thinking_to_send = delta_payload.thinking;
                                                        tool_calls_to_send = delta_payload.tool_calls_delta;
                                                    }

                                                if !content_to_send.is_empty() || !thinking_to_send.is_empty() || tool_calls_to_send.is_some() {
                                                    let ollama_chunk = create_ollama_streaming_chunk(
                                                        &model_clone_for_task,
                                                        &content_to_send,
                                                        is_chat_endpoint,
                                                        false,
                                                        tool_calls_to_send.as_ref(),
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
                                                        let mut tool_calls_to_send: Option<Value> = None;

                                                        if let Some(choice) = extract_first_choice(&recovered_json)
                                                            && let Some(delta_payload) = process_choice_delta(choice, &mut chunk_state) {
                                                                content_to_send = delta_payload.content;
                                                                thinking_to_send = delta_payload.thinking;
                                                                tool_calls_to_send = delta_payload.tool_calls_delta;
                                                            }

                                                        if !content_to_send.is_empty() || !thinking_to_send.is_empty() || tool_calls_to_send.is_some() {
                                                            let ollama_chunk = create_ollama_streaming_chunk(
                                                                &model_clone_for_task,
                                                                &content_to_send,
                                                                is_chat_endpoint,
                                                                false,
                                                                tool_calls_to_send.as_ref(),
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
                                    let mut tool_calls_to_send: Option<Value> = None;

                                    if let Some(choice) = extract_first_choice(&recovered_json)
                                        && let Some(delta_payload) = process_choice_delta(choice, &mut chunk_state) {
                                            content_to_send = delta_payload.content;
                                            thinking_to_send = delta_payload.thinking;
                                            tool_calls_to_send = delta_payload.tool_calls_delta;
                                        }

                                    if !content_to_send.is_empty() || !thinking_to_send.is_empty() || tool_calls_to_send.is_some() {
                                        let ollama_chunk = create_ollama_streaming_chunk(
                                            &model_clone_for_task,
                                            &content_to_send,
                                            is_chat_endpoint,
                                            false,
                                            tool_calls_to_send.as_ref(),
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
            let accumulated_tool_calls = chunk_state.take_tool_calls();
            let final_chunk = create_final_chunk(FinalChunkParams {
                model_name: &model_clone_for_task,
                duration: start_time.elapsed(),
                chunk_count,
                is_chat: is_chat_endpoint,
                done_reason: chunk_state.finish_reason(),
                tool_calls: accumulated_tool_calls,
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
/// [`map_native_event`]: deltas emit intermediate Ollama chunks, `error` fails
/// the stream with a bare error line, and `chat.end` drives the final timing
/// chunk from the native `stats` block. Native is always chat-shaped, so chunk
/// recovery (OpenAI-specific) is intentionally skipped.
pub async fn handle_native_streaming_response(
    lm_studio_response: reqwest::Response,
    ollama_model_name: &str,
    start_time: Instant,
    cancellation_token: CancellationToken,
    stream_timeout_seconds: u64,
) -> Result<axum::response::Response, ProxyError> {
    let status = lm_studio_response.status();
    if !status.is_success() {
        let body_text = lm_studio_response.text().await.unwrap_or_default();
        let message = serde_json::from_str::<serde_json::Value>(&body_text)
            .ok()
            .and_then(|v| match v.get("error") {
                Some(serde_json::Value::Object(obj)) => obj
                    .get("message")
                    .and_then(|m| m.as_str())
                    .map(|s| s.to_string()),
                Some(serde_json::Value::String(s)) => Some(s.clone()),
                _ => None,
            })
            .unwrap_or_else(|| format!("LM Studio error: {}", status));
        return Err(ProxyError::new(message, status.as_u16()));
    }

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
        let mut chunk_state = ChunkProcessingState::default();
        let mut first_chunk_received = false;
        // Captured from `chat.end` so the final done chunk can carry native stats.
        let mut chat_end: Option<NativeChatEnd> = None;

        let stream_result = 'stream_loop: loop {
            tokio::select! {
                biased;
                _ = token_clone.cancelled() => {
                    let cancellation_chunk = create_cancellation_chunk(
                        &model_clone_for_task,
                        start_time.elapsed(),
                        chunk_count,
                        chunk_state.take_tool_calls(),
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
                                            if payload.content.is_empty()
                                                && payload.thinking.is_empty()
                                                && payload.tool_calls_delta.is_none()
                                            {
                                                continue;
                                            }
                                            let ollama_chunk = create_ollama_streaming_chunk(
                                                &model_clone_for_task,
                                                &payload.content,
                                                true,
                                                false,
                                                payload.tool_calls_delta.as_ref(),
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
                                            let message = err.to_message();
                                            log::error!("native stream error: {}", message);
                                            send_error_and_close(&tx, &message).await;
                                            break 'stream_loop Err(message);
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
            let accumulated_tool_calls = chunk_state.take_tool_calls();
            let final_chunk = build_native_final_chunk(
                &model_clone_for_task,
                chat_end.as_ref(),
                start_time,
                chunk_count,
                accumulated_tool_calls,
            );
            send_chunk_and_close_channel(&tx, final_chunk).await;
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
/// [`TimingInfo::from_native_stats`] and `done_reason` from the parsed end
/// event; otherwise (stream ended early) it falls back to the wall-clock
/// heuristics in [`create_final_chunk`].
fn build_native_final_chunk(
    model_name: &str,
    chat_end: Option<&NativeChatEnd>,
    start_time: Instant,
    chunk_count: u64,
    tool_calls: Option<Value>,
) -> Value {
    let Some(end) = chat_end else {
        return create_final_chunk(FinalChunkParams {
            model_name,
            duration: start_time.elapsed(),
            chunk_count,
            is_chat: true,
            done_reason: None,
            tool_calls,
        });
    };

    let timing = TimingInfo::from_native_stats(&end.result, start_time, 10, chunk_count.max(1));

    let mut chunk =
        create_ollama_streaming_chunk(model_name, "", true, true, tool_calls.as_ref(), "");

    if let Some(obj) = chunk.as_object_mut() {
        obj.insert("done_reason".to_string(), json!(end.done_reason));
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

pub async fn handle_passthrough_streaming_response(
    response: reqwest::Response,
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
                    let cancel_data = format!("data: {{\"error\": \"{}\", \"cancelled\": true}}\n\n", ERROR_CANCELLED);
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
                            let error_data = format!("data: {{\"error\": \"streaming error: {}\"}}\n\n", e);
                            let _ = tx.send(Ok(bytes::Bytes::from(error_data)));
                            break;
                        }
                        Ok(None) => break,
                        Err(_) => {
                            let timeout_data = format!("data: {{\"error\": \"{}\"}}\n\n", ERROR_TIMEOUT);
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
