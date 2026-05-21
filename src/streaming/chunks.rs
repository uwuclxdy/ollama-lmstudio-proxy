use std::{collections::BTreeMap, time::Duration};

use serde_json::{Map, Value, json};
use tokio::sync::mpsc;

use crate::lmstudio::response::{TimingInfo, convert_tool_calls_to_ollama};

#[derive(Default)]
pub struct ChunkProcessingState {
    last_finish_reason: Option<String>,
    /// Accumulated tool_calls fragments across SSE deltas.
    ///
    /// OpenAI streams tool calls in pieces (name in one delta, arguments spread
    /// across several more). Ollama clients expect ONE chunk containing the
    /// complete `tool_calls` array once the assistant message is done. We merge
    /// fragments by OpenAI call index before converting them to Ollama shape.
    accumulated_tool_calls: BTreeMap<u64, Value>,
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

    pub fn accumulate_tool_calls(&mut self, tool_calls: &[Value]) {
        for (position, tool_call) in tool_calls.iter().enumerate() {
            let index = tool_call_index(tool_call, position);
            let entry = self
                .accumulated_tool_calls
                .entry(index)
                .or_insert_with(|| json!({"index": index, "function": {}}));
            merge_tool_call_fragment(entry, tool_call, index);
        }
    }

    /// Returns the accumulated tool_calls if any were collected, consuming them.
    pub fn take_tool_calls(&mut self) -> Option<Value> {
        if self.accumulated_tool_calls.is_empty() {
            None
        } else {
            let calls: Vec<Value> = std::mem::take(&mut self.accumulated_tool_calls)
                .into_values()
                .collect();
            Some(convert_tool_calls_to_ollama(&calls))
        }
    }
}

pub struct ChoiceDeltaPayload {
    pub content: String,
    pub thinking: String,
    /// Partial tool_calls fragment from THIS delta only, in Ollama shape, for
    /// emission as an intermediate `done:false` chunk. The accumulator in
    /// `ChunkProcessingState` independently merges fragments for the final
    /// `done:true` chunk — this field does not consume that state.
    pub tool_calls_delta: Option<Value>,
}

fn tool_call_index(tool_call: &Value, position: usize) -> u64 {
    tool_call
        .get("index")
        .and_then(|index| index.as_u64())
        .unwrap_or(position as u64)
}

fn merge_tool_call_fragment(accumulated: &mut Value, fragment: &Value, index: u64) {
    let accumulated_object = accumulated.as_object_mut().expect("tool call is an object");
    accumulated_object.insert("index".to_string(), json!(index));

    if let Some(id) = fragment.get("id") {
        accumulated_object.insert("id".to_string(), id.clone());
    }
    if let Some(call_type) = fragment.get("type") {
        accumulated_object.insert("type".to_string(), call_type.clone());
    }

    let function = accumulated_object
        .entry("function".to_string())
        .or_insert_with(|| Value::Object(Map::new()));
    let function_object = function.as_object_mut().expect("function is an object");

    if let Some(name) = fragment
        .get("function")
        .and_then(|function| function.get("name"))
    {
        function_object.insert("name".to_string(), name.clone());
    }
    if let Some(arguments) = fragment
        .get("function")
        .and_then(|function| function.get("arguments"))
    {
        match arguments {
            Value::String(part) => {
                let existing = function_object
                    .entry("arguments".to_string())
                    .or_insert_with(|| Value::String(String::new()));
                match existing {
                    Value::String(buffer) => buffer.push_str(part),
                    other => *other = Value::String(part.clone()),
                }
            }
            other => {
                function_object.insert("arguments".to_string(), other.clone());
            }
        }
    }
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
            // Accumulate into state for the final done chunk, AND surface this
            // delta's fragment to the caller for an intermediate chunk so
            // clients see progressive tool_call data (per ChatStreamEvent spec).
            state.accumulate_tool_calls(new_tool_calls);
            tool_calls_delta = Some(convert_tool_calls_to_ollama(new_tool_calls));
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
    error_message: &str,
) {
    let error_chunk = create_error_chunk(error_message);
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
        let msg_capacity =
            2 + (!thinking.is_empty() as usize) + tool_calls_delta.is_some() as usize;
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

pub fn create_error_chunk(error_message: &str) -> Value {
    // Doc: mid-stream errors are a bare {"error":"…"} line, not a full chunk.
    json!({ "error": error_message })
}

pub fn create_cancellation_chunk(
    model_ollama_name: &str,
    duration: Duration,
    tokens_generated_estimate: u64,
    is_chat_endpoint: bool,
) -> Value {
    // Ollama's spec only documents `done_reason: stop | length`; "cancelled" is not a value
    // real clients expect. Leave content empty and omit `done_reason` rather than fabricating one.
    let timing = TimingInfo::from_stream_chunks(
        duration,
        tokens_generated_estimate,
        Some(tokens_generated_estimate),
    );

    let mut chunk =
        create_ollama_streaming_chunk(model_ollama_name, "", is_chat_endpoint, true, None, "");

    if let Some(chunk_obj) = chunk.as_object_mut() {
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
    }
    chunk
}

pub struct FinalChunkParams<'a> {
    pub model_name: &'a str,
    pub duration: Duration,
    pub chunk_count: u64,
    pub is_chat: bool,
    pub done_reason: Option<&'a str>,
    /// Accumulated tool_calls to emit in this final chunk.
    /// `None` when no tool calls were seen in the stream.
    pub tool_calls: Option<Value>,
}

pub fn create_final_chunk(params: FinalChunkParams<'_>) -> Value {
    // LM Studio's SSE chunks carry no `usage` and `stream_options.include_usage`
    // is not in its supported parameter list, so real per-token timings are unavailable
    // on the streaming path. Wall-clock heuristics are the only option until upstream adds it.
    let timing = TimingInfo::from_stream_chunks(params.duration, params.chunk_count, None);

    let mut chunk = create_ollama_streaming_chunk(
        params.model_name,
        "",
        params.is_chat,
        true,
        params.tool_calls.as_ref(),
        "",
    );

    if let Some(chunk_obj) = chunk.as_object_mut() {
        if let Some(reason) = params.done_reason {
            chunk_obj.insert("done_reason".to_string(), json!(reason));
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
    }
    chunk
}

#[cfg(test)]
#[path = "../../tests/unit/streaming_chunks.rs"]
mod tests;
