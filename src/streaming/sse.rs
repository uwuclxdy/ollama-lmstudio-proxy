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
use crate::logging::log_timed;
use crate::streaming::chunks::{
    create_cancellation_chunk, create_error_chunk, create_final_chunk,
    create_ollama_streaming_chunk,
};
use crate::streaming::response::{
    create_ollama_streaming_response, create_passthrough_streaming_response,
};

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
                                                    if !send_ollama_chunk(&tx, &ollama_chunk).await {
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
                                                            if !send_ollama_chunk(&tx, &ollama_chunk).await {
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
                                        if !send_ollama_chunk(&tx, &ollama_chunk).await {
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

    create_ollama_streaming_response(rx)
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

    create_passthrough_streaming_response(rx)
}

#[derive(Default)]
struct ChunkProcessingState {
    last_finish_reason: Option<String>,
}

impl ChunkProcessingState {
    fn finish_reason(&self) -> Option<&str> {
        self.last_finish_reason.as_deref()
    }

    fn update_finish_reason(&mut self, choice: &Value) {
        if let Some(reason) = choice.get("finish_reason").and_then(|value| value.as_str()) {
            self.last_finish_reason = Some(reason.to_string());
        }
    }
}

struct ChoiceDeltaPayload {
    content: String,
    tool_calls_delta: Option<Value>,
}

fn extract_first_choice(chunk: &Value) -> Option<&Value> {
    chunk
        .get("choices")
        .and_then(|choices| choices.as_array())
        .and_then(|array| array.first())
}

fn process_choice_delta(
    choice: &Value,
    state: &mut ChunkProcessingState,
) -> Option<ChoiceDeltaPayload> {
    state.update_finish_reason(choice);

    let mut content = String::new();
    let mut tool_calls_delta: Option<Value> = None;

    if let Some(delta) = choice.get("delta") {
        if let Some(content_value) = delta.get("content") {
            append_stream_content(content_value, &mut content);
        }
        if let Some(reasoning_value) = delta.get("reasoning") {
            append_stream_content(reasoning_value, &mut content);
        }
        if let Some(new_tool_calls) = delta.get("tool_calls").and_then(|value| value.as_array())
            && !new_tool_calls.is_empty()
        {
            tool_calls_delta = Some(json!(new_tool_calls));
        }
    }

    if content.is_empty() {
        if let Some(text_value) = choice.get("text") {
            append_stream_content(text_value, &mut content);
        } else if let Some(message_content) = choice
            .get("message")
            .and_then(|message| message.get("content"))
        {
            append_stream_content(message_content, &mut content);
        }
    }

    if content.is_empty() && tool_calls_delta.is_none() {
        None
    } else {
        Some(ChoiceDeltaPayload {
            content,
            tool_calls_delta,
        })
    }
}

fn append_stream_content(content_value: &Value, buffer: &mut String) {
    match content_value {
        Value::String(text) => buffer.push_str(text),
        Value::Array(items) => {
            for item in items {
                if let Some(piece_type) = item.get("type").and_then(|t| t.as_str()) {
                    match piece_type {
                        "text" | "reasoning" | "output_text" => {
                            if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                                buffer.push_str(text);
                            }
                        }
                        _ => {
                            if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                                buffer.push_str(text);
                            }
                        }
                    }
                } else if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                    buffer.push_str(text);
                }
            }
        }
        Value::Object(obj) => {
            if let Some(text) = obj.get("text").and_then(|t| t.as_str()) {
                buffer.push_str(text);
            } else if let Some(nested) = obj.get("content") {
                append_stream_content(nested, buffer);
            }
        }
        _ => {}
    }
}

async fn send_ollama_chunk(
    tx: &mpsc::UnboundedSender<Result<bytes::Bytes, std::io::Error>>,
    chunk: &Value,
) -> bool {
    let chunk_json = serde_json::to_string(chunk).unwrap_or_else(|e| {
        log::error!("chunk serialization failed: {}", e);
        String::from("{\"error\":\"internal proxy error: failed to serialize chunk\"}")
    });
    let chunk_with_newline = format!("{}\n", chunk_json);
    tx.send(Ok(bytes::Bytes::from(chunk_with_newline))).is_ok()
}

async fn send_chunk_and_close_channel(
    tx: &mpsc::UnboundedSender<Result<bytes::Bytes, std::io::Error>>,
    chunk: Value,
) {
    let chunk_json = serde_json::to_string(&chunk).unwrap_or_default();
    let chunk_with_newline = format!("{}\n", chunk_json);
    let _ = tx.send(Ok(bytes::Bytes::from(chunk_with_newline)));
}

async fn send_error_and_close(
    tx: &mpsc::UnboundedSender<Result<bytes::Bytes, std::io::Error>>,
    model_ollama_name: &str,
    error_message: &str,
    is_chat_endpoint: bool,
) {
    let error_chunk = create_error_chunk(model_ollama_name, error_message, is_chat_endpoint);
    send_chunk_and_close_channel(tx, error_chunk).await;
}

fn recover_json_from_chunk(chunk_data: &str) -> Option<Value> {
    // Try to find valid JSON within the chunk by looking for common patterns
    // 1. Try to find JSON objects/arrays within the text
    // 2. Try to extract the core structure even if there are extra characters

    // First, try to find the first '{' and last '}' to extract a potential JSON object
    if let Some(start_brace) = chunk_data.find('{')
        && let Some(end_brace) = chunk_data.rfind('}')
        && start_brace < end_brace
    {
        let potential_json = &chunk_data[start_brace..=end_brace];
        if let Ok(parsed) = serde_json::from_str::<Value>(potential_json) {
            return Some(parsed);
        }
    }

    // Try to find JSON array pattern
    if let Some(start_bracket) = chunk_data.find('[')
        && let Some(end_bracket) = chunk_data.rfind(']')
        && start_bracket < end_bracket
    {
        let potential_json = &chunk_data[start_bracket..=end_bracket];
        if let Ok(parsed) = serde_json::from_str::<Value>(potential_json) {
            return Some(parsed);
        }
    }

    // Try to clean up common issues:
    // - Remove trailing commas
    // - Fix missing quotes
    // - Remove extra whitespace
    let cleaned_data = chunk_data
        .replace(",\n}", "\n}") // Remove trailing commas before closing braces
        .replace(",\n]", "\n]") // Remove trailing commas before closing brackets
        .replace(":\n", ": \"\""); // Add missing values for empty fields

    // Try parsing the cleaned version
    if let Ok(parsed) = serde_json::from_str::<Value>(&cleaned_data) {
        return Some(parsed);
    }

    // Try to extract just the choices array if it exists
    if let Some(choices_start) = chunk_data.find("\"choices\":")
        && let Some(array_start) = chunk_data[choices_start..].find('[')
    {
        let choices_start_pos = choices_start + array_start;
        if let Some(array_end) = chunk_data[choices_start_pos..].rfind(']') {
            let choices_json = &chunk_data[choices_start_pos..=choices_start_pos + array_end];
            if let Ok(parsed) = serde_json::from_str::<Value>(choices_json) {
                // Wrap the choices array in a proper response object
                let mut result = json!({});
                result["choices"] = parsed;
                return Some(result);
            }
        }
    }

    None
}
