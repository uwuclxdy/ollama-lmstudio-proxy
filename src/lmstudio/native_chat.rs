//! Building blocks for LM Studio's native `/api/v1/chat` endpoint.
//!
//! This is the richer alternative to the OpenAI-compat `/v1/chat/completions`
//! path. The native endpoint takes its own request shape (`input` array,
//! `system_prompt`, `max_output_tokens`, `reasoning`, `context_length`) and
//! returns an `output` array of typed items plus a `stats` block.
//!
//! These functions are pure and additive: nothing here is wired into a handler
//! yet. The request builder mirrors `build_lm_studio_request`'s conventions and
//! the converter mirrors `convert_to_ollama_chat`'s output shape.
//!
//! Source of truth:
//! - `api_docs/future/lmstudio/1_developer/2_rest/chat.md` (request + response)

use std::time::Instant;

use serde_json::{Map, Value, json};

use crate::lmstudio::request::normalize_reasoning;
use crate::lmstudio::response::{TimingInfo, convert_tool_calls_to_ollama};
use crate::streaming::chunks::map_done_reason;

/// Parameters for building a native `/api/v1/chat` request body.
///
/// `model_lm_studio_id` is the already-resolved LM Studio model id. `messages`
/// is the Ollama-shaped `messages` array; `system_prompt` is taken verbatim
/// (the caller decides whether it came from a system message or elsewhere).
pub struct NativeChatRequestParams<'a> {
    pub model_lm_studio_id: &'a str,
    pub messages: &'a Value,
    pub system_prompt: Option<&'a str>,
    pub ollama_options: Option<&'a Value>,
    /// Ollama `think` / reasoning value, normalised to LM Studio's `reasoning`.
    pub think: Option<&'a Value>,
    pub stream: bool,
}

/// Map an Ollama chat request to a native LM Studio `/api/v1/chat` JSON body.
///
/// The `input` array is built from the Ollama `messages`: text turns become
/// `{type:"message", role, content}` and any per-message `images` become
/// `{type:"image", data_url}` entries (data URLs sniffed via the shared image
/// helper). Sampling params come from Ollama `options`; note the native field
/// is `max_output_tokens` (not `max_tokens`) and `min_p`/`repeat_penalty` ARE
/// supported here (unlike the OpenAI-compat path).
pub fn build_native_chat_request(params: NativeChatRequestParams<'_>) -> Value {
    let mut body = Map::new();
    body.insert("model".to_string(), json!(params.model_lm_studio_id));
    body.insert("input".to_string(), build_native_input(params.messages));

    if let Some(system_text) = params.system_prompt {
        body.insert("system_prompt".to_string(), json!(system_text));
    }

    apply_native_sampling(params.ollama_options, &mut body);

    if let Some(think_val) = params.think {
        body.insert("reasoning".to_string(), normalize_reasoning(think_val));
    }

    body.insert("stream".to_string(), json!(params.stream));

    Value::Object(body)
}

/// Sampling params that map straight from Ollama `options` to native names.
///
/// `min_p` and `repeat_penalty` are included because the native endpoint
/// documents them, unlike OpenAI-compat where `min_p` is warn-dropped.
const NATIVE_SAMPLING_KEYS: &[&str] = &["temperature", "top_p", "top_k", "min_p", "repeat_penalty"];

fn apply_native_sampling(ollama_options: Option<&Value>, body: &mut Map<String, Value>) {
    let Some(options) = ollama_options else {
        return;
    };

    for key in NATIVE_SAMPLING_KEYS {
        if let Some(value) = options.get(key) {
            body.insert((*key).to_string(), value.clone());
        }
    }

    // Native uses `max_output_tokens`; accept either Ollama spelling as source.
    if let Some(max_tokens) = options
        .get("max_tokens")
        .or_else(|| options.get("num_predict"))
    {
        body.insert("max_output_tokens".to_string(), max_tokens.clone());
    }

    if let Some(ctx) = options.get("num_ctx") {
        body.insert("context_length".to_string(), ctx.clone());
    }
}

/// Build the native `input` array from Ollama `messages`.
///
/// Each message yields a `{type:"message", role, content}` entry (when it has
/// non-empty text content), followed by one `{type:"image", data_url}` entry
/// per image in its `images` sibling array. Non-array `messages` yield an empty
/// input array.
fn build_native_input(messages: &Value) -> Value {
    let Some(msg_array) = messages.as_array() else {
        return Value::Array(Vec::new());
    };

    let mut input = Vec::with_capacity(msg_array.len());
    for msg in msg_array {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");

        if let Some(content) = msg.get("content") {
            let text = content
                .as_str()
                .map(ToString::to_string)
                .unwrap_or_else(|| content.to_string());
            if !text.is_empty() && content.as_str() != Some("") {
                input.push(json!({
                    "type": "message",
                    "role": role,
                    "content": text,
                }));
            }
        }

        if let Some(images) = msg.get("images").and_then(|i| i.as_array()) {
            for image in images {
                if let Some(base64_data) = image.as_str() {
                    input.push(json!({
                        "type": "image",
                        "data_url": crate::lmstudio::images::native_image_data_url(base64_data),
                    }));
                }
            }
        }
    }

    Value::Array(input)
}

/// Convert a native `/api/v1/chat` response to an Ollama chat response object.
///
/// Walks the `output` array: `{type:"message"}` entries concatenate into
/// `message.content`; `{type:"reasoning"}` entries concatenate into
/// `message.thinking` (omitted when empty); `{type:"tool_call"}` entries are
/// collected and shaped via `convert_tool_calls_to_ollama`. Timing comes from
/// the native `stats` block via the shared `TimingInfo::from_native_stats`.
/// Output shape matches `convert_to_ollama_chat`.
pub fn convert_native_to_ollama_chat(
    native_response: &Value,
    model_ollama_name: &str,
    start_time: Instant,
) -> Value {
    let NativeOutput {
        content,
        thinking,
        tool_calls,
    } = collect_native_output(native_response.get("output"));

    let timing = TimingInfo::from_native_stats(
        native_response,
        start_time,
        10,
        crate::lmstudio::response::estimate_token_count(&content),
    );

    let mut ollama_message = json!({
        "role": "assistant",
        "content": content,
    });

    if let Some(msg_obj) = ollama_message.as_object_mut() {
        if !thinking.is_empty() {
            msg_obj.insert("thinking".to_string(), json!(thinking));
        }
        if !tool_calls.is_empty() {
            msg_obj.insert(
                "tool_calls".to_string(),
                convert_tool_calls_to_ollama(&tool_calls),
            );
        }
    }

    // Native streams report finish via stream events, not a per-choice
    // `finish_reason`; a completed non-streaming response always means `stop`.
    let done_reason = "stop";

    let mut response = json!({
        "model": model_ollama_name,
        "created_at": chrono::Utc::now().to_rfc3339(),
        "message": ollama_message,
        "done": true,
        "done_reason": done_reason,
        "total_duration": timing.total_duration,
        "load_duration": timing.load_duration,
        "prompt_eval_count": timing.prompt_eval_count,
        "prompt_eval_duration": timing.prompt_eval_duration,
        "eval_count": timing.eval_count,
        "eval_duration": timing.eval_duration,
    });

    if let Some(response_id) = native_response.get("response_id").and_then(|v| v.as_str())
        && let Some(obj) = response.as_object_mut()
    {
        obj.insert("response_id".to_string(), json!(response_id));
    }

    response
}

/// Aggregated text/reasoning/tool_call data drawn from a native `output` array.
struct NativeOutput {
    content: String,
    thinking: String,
    tool_calls: Vec<Value>,
}

/// Walk a native `output` array, concatenating message and reasoning text and
/// collecting tool_call entries into OpenAI-ish shape for later conversion.
fn collect_native_output(output: Option<&Value>) -> NativeOutput {
    let mut content = String::new();
    let mut thinking = String::new();
    let mut tool_calls = Vec::new();

    let Some(items) = output.and_then(|o| o.as_array()) else {
        return NativeOutput {
            content,
            thinking,
            tool_calls,
        };
    };

    for item in items {
        match item.get("type").and_then(|t| t.as_str()) {
            Some("message") => {
                if let Some(text) = item.get("content").and_then(|c| c.as_str()) {
                    content.push_str(text);
                }
            }
            Some("reasoning") => {
                if let Some(text) = item.get("content").and_then(|c| c.as_str()) {
                    thinking.push_str(text);
                }
            }
            Some("tool_call") => {
                tool_calls.push(native_tool_call_to_openai(item));
            }
            _ => {}
        }
    }

    NativeOutput {
        content,
        thinking,
        tool_calls,
    }
}

/// Reshape a native `{type:"tool_call", tool, arguments, ...}` item into the
/// OpenAI-ish `{function: {name, arguments}}` shape that
/// `convert_tool_calls_to_ollama` consumes. Native `arguments` is already an
/// object, which the converter passes through unchanged.
pub fn native_tool_call_to_openai(item: &Value) -> Value {
    let name = item.get("tool").and_then(|t| t.as_str()).unwrap_or("");
    let arguments = item
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));

    json!({
        "function": {
            "name": name,
            "arguments": arguments,
        }
    })
}

/// Re-exported so callers can build a final timing chunk from the same
/// `done_reason` literal the non-streaming converter uses.
pub fn native_done_reason() -> &'static str {
    // Mirror `map_done_reason("stop")` to stay aligned with the OpenAI path.
    map_done_reason("stop").unwrap_or("stop")
}

#[cfg(test)]
#[path = "../../tests/unit/lmstudio_native_chat.rs"]
mod tests;
