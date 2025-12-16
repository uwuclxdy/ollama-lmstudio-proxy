use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::constants::ERROR_CANCELLED;
use crate::handlers::transform::TimingInfo;

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
    let timing = TimingInfo::calculate_legacy(
        Instant::now() - duration,
        10,
        tokens_generated_estimate,
        None,
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
    let timing = TimingInfo::calculate_legacy(
        Instant::now() - duration,
        10,
        chunk_count_for_token_estimation.max(1),
        None,
        None,
    );

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
