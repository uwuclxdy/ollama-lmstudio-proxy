use std::time::Duration;

use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::constants::ERROR_CANCELLED;
use crate::handlers::transform::TimingInfo;

#[derive(Default)]
pub struct ChunkProcessingState {
    last_finish_reason: Option<String>,
}

impl ChunkProcessingState {
    pub fn finish_reason(&self) -> Option<&str> {
        self.last_finish_reason.as_deref()
    }

    pub fn update_finish_reason(&mut self, choice: &Value) {
        if let Some(reason) = choice.get("finish_reason").and_then(|value| value.as_str()) {
            self.last_finish_reason = Some(reason.to_string());
        }
    }
}

pub struct ChoiceDeltaPayload {
    pub content: String,
    pub tool_calls_delta: Option<Value>,
}

pub fn extract_first_choice(chunk: &Value) -> Option<&Value> {
    chunk
        .get("choices")
        .and_then(|choices| choices.as_array())
        .and_then(|array| array.first())
}

pub fn process_choice_delta(
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

pub async fn send_chunk(
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

pub async fn send_chunk_and_close_channel(
    tx: &mpsc::UnboundedSender<Result<bytes::Bytes, std::io::Error>>,
    chunk: Value,
) {
    let chunk_json = serde_json::to_string(&chunk).unwrap_or_default();
    let chunk_with_newline = format!("{}\n", chunk_json);
    let _ = tx.send(Ok(bytes::Bytes::from(chunk_with_newline)));
}

pub async fn send_error_and_close(
    tx: &mpsc::UnboundedSender<Result<bytes::Bytes, std::io::Error>>,
    model_ollama_name: &str,
    error_message: &str,
    is_chat_endpoint: bool,
) {
    let error_chunk = create_error_chunk(model_ollama_name, error_message, is_chat_endpoint);
    send_chunk_and_close_channel(tx, error_chunk).await;
}

pub fn create_ollama_streaming_chunk(
    model_ollama_name: &str,
    content: &str,
    is_chat_endpoint: bool,
    done: bool,
    tool_calls_delta: Option<&Value>,
) -> Value {
    let timestamp = chrono::Utc::now().to_rfc3339();

    if is_chat_endpoint {
        let mut message_obj = json!({
            "role": "assistant",
            "content": content
        });
        if let Some(tc_delta) = tool_calls_delta
            && let Some(msg_map) = message_obj.as_object_mut()
        {
            msg_map.insert("tool_calls".to_string(), tc_delta.clone());
        }

        json!({
            "model": model_ollama_name,
            "created_at": timestamp,
            "message": message_obj,
            "done": done
        })
    } else {
        json!({
            "model": model_ollama_name,
            "created_at": timestamp,
            "response": content,
            "done": done
        })
    }
}

pub fn create_error_chunk(
    model_ollama_name: &str,
    error_message: &str,
    is_chat_endpoint: bool,
) -> Value {
    let mut chunk =
        create_ollama_streaming_chunk(model_ollama_name, "", is_chat_endpoint, true, None);
    if let Some(chunk_obj) = chunk.as_object_mut() {
        chunk_obj.insert("error".to_string(), json!(error_message));
        if is_chat_endpoint
            && let Some(msg) = chunk_obj.get_mut("message").and_then(|m| m.as_object_mut())
        {
            msg.insert("content".to_string(), json!(""));
        }
    }
    chunk
}

pub fn create_cancellation_chunk(
    model_ollama_name: &str,
    duration: Duration,
    tokens_generated_estimate: u64,
    is_chat_endpoint: bool,
) -> Value {
    let timing = TimingInfo::from_stream_chunks(
        duration,
        tokens_generated_estimate,
        Some(tokens_generated_estimate),
    );

    let mut chunk =
        create_ollama_streaming_chunk(model_ollama_name, "", is_chat_endpoint, true, None);

    if let Some(chunk_obj) = chunk.as_object_mut() {
        let content_field_value = if tokens_generated_estimate > 0 {
            format!(
                "[Request cancelled after {} tokens generated (estimated)]",
                tokens_generated_estimate
            )
        } else {
            ERROR_CANCELLED.to_string()
        };

        if is_chat_endpoint {
            if let Some(msg) = chunk_obj.get_mut("message").and_then(|m| m.as_object_mut()) {
                msg.insert("content".to_string(), json!(content_field_value));
            }
        } else {
            chunk_obj.insert("response".to_string(), json!(content_field_value));
        }

        chunk_obj.insert("total_duration".to_string(), json!(timing.total_duration));
        chunk_obj.insert("load_duration".to_string(), json!(timing.load_duration));
        chunk_obj.insert(
            "prompt_eval_count".to_string(),
            json!(timing.prompt_eval_count),
        );
        chunk_obj.insert(
            "prompt_eval_duration".to_string(),
            json!(timing.prompt_eval_duration),
        );
        chunk_obj.insert("eval_count".to_string(), json!(timing.eval_count));
        chunk_obj.insert("eval_duration".to_string(), json!(timing.eval_duration));
        chunk_obj.insert("done_reason".to_string(), json!("cancelled"));
    }
    chunk
}

pub fn create_final_chunk(
    model_ollama_name: &str,
    duration: Duration,
    chunk_count_for_token_estimation: u64,
    is_chat_endpoint: bool,
    done_reason: Option<&str>,
) -> Value {
    let timing = TimingInfo::from_stream_chunks(duration, chunk_count_for_token_estimation, None);

    let mut chunk =
        create_ollama_streaming_chunk(model_ollama_name, "", is_chat_endpoint, true, None);

    if let Some(chunk_obj) = chunk.as_object_mut() {
        chunk_obj.insert(
            "done_reason".to_string(),
            json!(done_reason.unwrap_or("stop")),
        );
        chunk_obj.insert("total_duration".to_string(), json!(timing.total_duration));
        chunk_obj.insert("load_duration".to_string(), json!(timing.load_duration));
        chunk_obj.insert(
            "prompt_eval_count".to_string(),
            json!(timing.prompt_eval_count),
        );
        chunk_obj.insert(
            "prompt_eval_duration".to_string(),
            json!(timing.prompt_eval_duration),
        );
        chunk_obj.insert("eval_count".to_string(), json!(timing.eval_count));
        chunk_obj.insert("eval_duration".to_string(), json!(timing.eval_duration));
        if !is_chat_endpoint {
            chunk_obj.insert("context".to_string(), json!([]));
        }
    }
    chunk
}
