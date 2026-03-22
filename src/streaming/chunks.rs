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
    pub thinking: String,
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
    let mut thinking = String::new();
    let mut tool_calls_delta: Option<Value> = None;

    if let Some(delta) = choice.get("delta") {
        if let Some(content_value) = delta.get("content") {
            append_stream_content(content_value, &mut content);
        }
        if let Some(reasoning_value) = delta.get("reasoning") {
            append_stream_content(reasoning_value, &mut thinking);
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

    if content.is_empty() && thinking.is_empty() && tool_calls_delta.is_none() {
        None
    } else {
        Some(ChoiceDeltaPayload {
            content,
            thinking,
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
    let mut buf = serde_json::to_vec(chunk).unwrap_or_else(|e| {
        log::error!("chunk serialization failed: {}", e);
        b"{\"error\":\"internal proxy error: failed to serialize chunk\"}".to_vec()
    });
    buf.push(b'\n');

    tx.send(Ok(bytes::Bytes::from(buf))).is_ok()
}

pub async fn send_chunk_and_close_channel(
    tx: &mpsc::UnboundedSender<Result<bytes::Bytes, std::io::Error>>,
    chunk: Value,
) {
    let mut buf = serde_json::to_vec(&chunk).unwrap_or_default();
    buf.push(b'\n');
    let _ = tx.send(Ok(bytes::Bytes::from(buf)));
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
    thinking: &str,
) -> Value {
    let timestamp = chrono::Utc::now().to_rfc3339();

    if is_chat_endpoint {
        let msg_capacity = 2 + (!thinking.is_empty() as usize) + tool_calls_delta.is_some() as usize;
        let mut msg_map = serde_json::Map::with_capacity(msg_capacity);
        msg_map.insert("role".into(), Value::String("assistant".into()));
        msg_map.insert("content".into(), Value::String(content.into()));
        if !thinking.is_empty() {
            msg_map.insert("thinking".into(), Value::String(thinking.into()));
        }
        if let Some(tc_delta) = tool_calls_delta {
            msg_map.insert("tool_calls".into(), tc_delta.clone());
        }

        let mut map = serde_json::Map::with_capacity(4);
        map.insert("model".into(), Value::String(model_ollama_name.into()));
        map.insert("created_at".into(), Value::String(timestamp));
        map.insert("message".into(), Value::Object(msg_map));
        map.insert("done".into(), Value::Bool(done));
        Value::Object(map)
    } else {
        let capacity = 4 + (!thinking.is_empty() as usize);
        let mut map = serde_json::Map::with_capacity(capacity);
        map.insert("model".into(), Value::String(model_ollama_name.into()));
        map.insert("created_at".into(), Value::String(timestamp));
        map.insert("response".into(), Value::String(content.into()));
        map.insert("done".into(), Value::Bool(done));
        if !thinking.is_empty() {
            map.insert("thinking".into(), Value::String(thinking.into()));
        }
        Value::Object(map)
    }
}

pub fn create_error_chunk(
    model_ollama_name: &str,
    error_message: &str,
    is_chat_endpoint: bool,
) -> Value {
    let mut chunk =
        create_ollama_streaming_chunk(model_ollama_name, "", is_chat_endpoint, true, None, "");
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
        create_ollama_streaming_chunk(model_ollama_name, "", is_chat_endpoint, true, None, "");

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

pub struct FinalChunkParams<'a> {
    pub model_name: &'a str,
    pub duration: Duration,
    pub chunk_count: u64,
    pub is_chat: bool,
    pub done_reason: Option<&'a str>,
}

pub fn create_final_chunk(params: FinalChunkParams<'_>) -> Value {
    let timing = TimingInfo::from_stream_chunks(params.duration, params.chunk_count, None);

    let mut chunk =
        create_ollama_streaming_chunk(params.model_name, "", params.is_chat, true, None, "");

    if let Some(chunk_obj) = chunk.as_object_mut() {
        chunk_obj.insert(
            "done_reason".to_string(),
            json!(params.done_reason.unwrap_or("stop")),
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
        if !params.is_chat {
            chunk_obj.insert("context".to_string(), json!([]));
        }
    }
    chunk
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn choice_with_delta(content: Option<&str>, reasoning: Option<&str>) -> serde_json::Value {
        let mut delta = json!({});
        if let Some(c) = content {
            delta.as_object_mut().unwrap().insert("content".to_string(), json!(c));
        }
        if let Some(r) = reasoning {
            delta.as_object_mut().unwrap().insert("reasoning".to_string(), json!(r));
        }
        json!({ "delta": delta })
    }

    #[test]
    fn reasoning_goes_to_thinking_not_content() {
        let choice = choice_with_delta(Some("answer"), Some("my thinking"));
        let mut state = ChunkProcessingState::default();
        let payload = process_choice_delta(&choice, &mut state).unwrap();
        assert_eq!(payload.content, "answer");
        assert_eq!(payload.thinking, "my thinking");
    }

    #[test]
    fn reasoning_only_chunk_is_not_dropped() {
        // No content, only reasoning — must return Some
        let choice = choice_with_delta(None, Some("reasoning only"));
        let mut state = ChunkProcessingState::default();
        let payload = process_choice_delta(&choice, &mut state);
        assert!(payload.is_some());
        let p = payload.unwrap();
        assert_eq!(p.content, "");
        assert_eq!(p.thinking, "reasoning only");
    }

    #[test]
    fn chat_chunk_thinking_in_message() {
        let chunk = create_ollama_streaming_chunk("m", "hi", true, false, None, "my thought");
        let msg = chunk.get("message").unwrap();
        assert_eq!(msg.get("thinking").and_then(|v| v.as_str()), Some("my thought"));
        assert_eq!(msg.get("content").and_then(|v| v.as_str()), Some("hi"));
    }

    #[test]
    fn chat_chunk_no_thinking_field_when_empty() {
        let chunk = create_ollama_streaming_chunk("m", "hi", true, false, None, "");
        let msg = chunk.get("message").unwrap();
        assert!(msg.get("thinking").is_none());
    }

    #[test]
    fn generate_chunk_thinking_top_level() {
        let chunk = create_ollama_streaming_chunk("m", "response", false, false, None, "thought");
        assert_eq!(chunk.get("thinking").and_then(|v| v.as_str()), Some("thought"));
        // must NOT be nested inside message
        assert!(chunk.get("message").is_none());
    }
}
