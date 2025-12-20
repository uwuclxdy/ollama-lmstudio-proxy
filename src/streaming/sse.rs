use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use serde_json::Value;
use tokio::sync::mpsc;
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;

use crate::config::get_runtime_config;
use crate::constants::{
    ERROR_CANCELLED, ERROR_TIMEOUT, LOG_PREFIX_CONN, LOG_PREFIX_SUCCESS, SSE_DATA_PREFIX,
    SSE_DONE_MESSAGE, SSE_MESSAGE_BOUNDARY,
};
use crate::error::ProxyError;
use crate::logging::log_timed;
use crate::streaming::chunks::{
    ChunkProcessingState, create_cancellation_chunk, create_final_chunk,
    create_ollama_streaming_chunk, extract_first_choice, process_choice_delta, send_chunk,
    send_chunk_and_close_channel, send_error_and_close,
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
) -> Result<warp::reply::Response, ProxyError> {
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

                                while let Some(boundary_pos) = sse_buffer.find(SSE_MESSAGE_BOUNDARY) {
                                    let message_text = sse_buffer[..boundary_pos].to_string();
                                    sse_buffer.drain(..boundary_pos + SSE_MESSAGE_BOUNDARY.len());

                                    if message_text.trim().is_empty() { continue; }

                                    if let Some(data_content) = message_text.strip_prefix(SSE_DATA_PREFIX) {
                                        if data_content.trim() == SSE_DONE_MESSAGE {
                                            break 'stream_loop Ok(());
                                        }

                                        match serde_json::from_str::<Value>(data_content) {
                                            Ok(lm_studio_json_chunk) => {
                                                let mut content_to_send = String::new();
                                                let mut tool_calls_delta: Option<Value> = None;

                                                if let Some(choice) = extract_first_choice(&lm_studio_json_chunk)
                                                    && let Some(delta_payload) = process_choice_delta(choice, &mut chunk_state) {
                                                        content_to_send = delta_payload.content;
                                                        tool_calls_delta = delta_payload.tool_calls_delta;
                                                    }

                                                if !content_to_send.is_empty() || tool_calls_delta.is_some() {
                                                    let ollama_chunk = create_ollama_streaming_chunk(
                                                        &model_clone_for_task,
                                                        &content_to_send,
                                                        is_chat_endpoint,
                                                        false,
                                                        tool_calls_delta.as_ref()
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
                                                    // Attempt to recover by finding valid JSON within the problematic chunk
                                                    if let Some(recovered_json) = recover_json_from_chunk(data_content) {
                                                        log::info!("Successfully recovered chunk data");
                                                        let mut content_to_send = String::new();
                                                        let mut tool_calls_delta: Option<Value> = None;

                                                        if let Some(choice) = extract_first_choice(&recovered_json)
                                                            && let Some(delta_payload) = process_choice_delta(choice, &mut chunk_state) {
                                                                content_to_send = delta_payload.content;
                                                                tool_calls_delta = delta_payload.tool_calls_delta;
                                                            }

                                                        if !content_to_send.is_empty() || tool_calls_delta.is_some() {
                                                            let ollama_chunk = create_ollama_streaming_chunk(
                                                                &model_clone_for_task,
                                                                &content_to_send,
                                                                is_chat_endpoint,
                                                                false,
                                                                tool_calls_delta.as_ref()
                                                            );
                                                            chunk_count += 1;
                                                            if !send_chunk(&tx, &ollama_chunk).await {
                                                                break 'stream_loop Ok(());
                                                            }
                                                        }
                                                    } else {
                                                        log::error!("SSE parsing error (recovery failed): {}", e);
                                                        // Store the problematic chunk for potential later recovery
                                                        recovery_buffer.push_str(data_content);
                                                        recovery_buffer.push_str(SSE_MESSAGE_BOUNDARY);
                                                    }
                                                } else {
                                                    log::error!("SSE parsing error: {}", e);
                                                }
                                            }
                                        }
                                    } else if !message_text.trim().is_empty() {
                                         log::warn!("SSE format: non-standard line: {}", message_text);
                                    }
                                }
                            } else {
                                send_error_and_close(&tx, &model_clone_for_task, "invalid UTF-8 in stream", is_chat_endpoint).await;
                                break 'stream_loop Err("invalid UTF-8".to_string());
                            }
                        }
                        Ok(Some(Err(e))) => {
                            send_error_and_close(&tx, &model_clone_for_task, &format!("streaming error: {}", e), is_chat_endpoint).await;
                            break 'stream_loop Err(format!("network error: {}", e));
                        }
                        Ok(None) => {
                            log::warn!("stream ended without [DONE]");
                            if enable_chunk_recovery && !recovery_buffer.is_empty() {
                                log::info!("Attempting to recover from remaining buffer data");
                                if let Some(recovered_json) = recover_json_from_chunk(&recovery_buffer) {
                                    log::info!("Successfully recovered data from remaining buffer");
                                    let mut content_to_send = String::new();
                                    let mut tool_calls_delta: Option<Value> = None;

                                    if let Some(choice) = extract_first_choice(&recovered_json)
                                        && let Some(delta_payload) = process_choice_delta(choice, &mut chunk_state) {
                                            content_to_send = delta_payload.content;
                                            tool_calls_delta = delta_payload.tool_calls_delta;
                                        }

                                    if !content_to_send.is_empty() || tool_calls_delta.is_some() {
                                        let ollama_chunk = create_ollama_streaming_chunk(
                                            &model_clone_for_task,
                                            &content_to_send,
                                            is_chat_endpoint,
                                            false,
                                            tool_calls_delta.as_ref()
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
                            send_error_and_close(&tx, &model_clone_for_task, ERROR_TIMEOUT, is_chat_endpoint).await;
                            break 'stream_loop Err(ERROR_TIMEOUT.to_string());
                        }
                    }
                }
            }
        };

        if stream_result.is_ok() && !token_clone.is_cancelled() {
            let final_chunk = create_final_chunk(
                &model_clone_for_task,
                start_time.elapsed(),
                chunk_count,
                is_chat_endpoint,
                chunk_state.finish_reason(),
            );
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

pub async fn handle_passthrough_streaming_response(
    response: reqwest::Response,
    cancellation_token: CancellationToken,
    stream_timeout_seconds: u64,
) -> Result<warp::reply::Response, ProxyError> {
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
