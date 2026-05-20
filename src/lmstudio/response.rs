use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::constants::{
    DEFAULT_LOAD_DURATION_NS, TIMING_EVAL_RATIO, TIMING_PROMPT_RATIO, TOKEN_TO_CHAR_RATIO,
};

/// Timing information for Ollama responses
#[derive(Debug, Clone)]
pub struct TimingInfo {
    pub total_duration: u64,
    pub load_duration: u64,
    pub prompt_eval_count: u64,
    pub prompt_eval_duration: u64,
    pub eval_count: u64,
    pub eval_duration: u64,
}

impl TimingInfo {
    pub fn from_native_stats(
        lm_response: &Value,
        start_time: Instant,
        estimated_input_tokens: u64,
        estimated_output_tokens: u64,
    ) -> Self {
        if let Some(stats) = lm_response.get("stats") {
            // LM Studio publishes two stats shapes:
            //   /api/v0/*       — { generation_time, time_to_first_token, tokens_per_second }
            //   /v1/responses   — { time_to_first_token_seconds, tokens_per_second,
            //                       model_load_time_seconds, input_tokens, total_output_tokens }
            //
            // generation_time (when present) is the post-TTFT eval phase, NOT total.
            // For v1 responses there is no generation_time; derive eval phase from
            // total_output_tokens / tokens_per_second.
            let time_to_first_token = stats
                .get("time_to_first_token_seconds")
                .or_else(|| stats.get("time_to_first_token"))
                .and_then(|t| t.as_f64())
                .unwrap_or(0.0);

            let tokens_per_second = stats
                .get("tokens_per_second")
                .and_then(|t| t.as_f64())
                .unwrap_or(0.0);

            let actual_prompt_tokens = lm_response
                .get("usage")
                .and_then(|u| u.get("prompt_tokens"))
                .and_then(|t| t.as_u64())
                .or_else(|| stats.get("input_tokens").and_then(|t| t.as_u64()))
                .unwrap_or(estimated_input_tokens);

            let actual_completion_tokens = lm_response
                .get("usage")
                .and_then(|u| u.get("completion_tokens"))
                .and_then(|t| t.as_u64())
                .or_else(|| stats.get("total_output_tokens").and_then(|t| t.as_u64()))
                .unwrap_or(estimated_output_tokens);

            let generation_time = match stats.get("generation_time").and_then(|t| t.as_f64()) {
                Some(g) => g,
                None if tokens_per_second > 0.0 && actual_completion_tokens > 0 => {
                    actual_completion_tokens as f64 / tokens_per_second
                }
                _ => 0.0,
            };

            let load_duration_ns = stats
                .get("model_load_time_seconds")
                .and_then(|t| t.as_f64())
                .map(|s| (s * 1_000_000_000.0) as u64)
                .unwrap_or(DEFAULT_LOAD_DURATION_NS);

            let ttft_ns = (time_to_first_token * 1_000_000_000.0) as u64;
            let generation_time_ns = (generation_time * 1_000_000_000.0) as u64;
            let total_duration_ns = ttft_ns + generation_time_ns;

            return Self {
                total_duration: total_duration_ns.max(1),
                load_duration: load_duration_ns,
                prompt_eval_count: actual_prompt_tokens.max(1),
                prompt_eval_duration: ttft_ns.max(1),
                eval_count: actual_completion_tokens.max(1),
                eval_duration: generation_time_ns.max(1),
            };
        }

        Self::from_legacy_estimation(
            start_time,
            estimated_input_tokens,
            estimated_output_tokens,
            lm_response
                .get("usage")
                .and_then(|u| u.get("prompt_tokens"))
                .and_then(|t| t.as_u64()),
            lm_response
                .get("usage")
                .and_then(|u| u.get("completion_tokens"))
                .and_then(|t| t.as_u64()),
        )
    }

    pub fn from_legacy_estimation(
        start_time: Instant,
        input_tokens_estimate: u64,
        output_tokens_estimate: u64,
        actual_prompt_tokens: Option<u64>,
        actual_completion_tokens: Option<u64>,
    ) -> Self {
        let total_duration_ns = start_time.elapsed().as_nanos() as u64;
        Self::from_duration_and_tokens(
            total_duration_ns,
            input_tokens_estimate,
            output_tokens_estimate,
            actual_prompt_tokens,
            actual_completion_tokens,
        )
    }

    pub fn from_stream_chunks(
        duration: Duration,
        chunk_count_estimate: u64,
        actual_completion_tokens: Option<u64>,
    ) -> Self {
        let total_duration_ns = duration.as_nanos() as u64;
        Self::from_duration_and_tokens(
            total_duration_ns,
            10,
            chunk_count_estimate.max(1),
            None,
            actual_completion_tokens,
        )
    }

    fn from_duration_and_tokens(
        total_duration_ns: u64,
        input_tokens_estimate: u64,
        output_tokens_estimate: u64,
        actual_prompt_tokens: Option<u64>,
        actual_completion_tokens: Option<u64>,
    ) -> Self {
        let final_prompt_tokens = actual_prompt_tokens.unwrap_or(input_tokens_estimate).max(1);
        let final_eval_tokens = actual_completion_tokens
            .unwrap_or(output_tokens_estimate)
            .max(1);

        let prompt_eval_duration_ns =
            if final_prompt_tokens + final_eval_tokens > 0 && total_duration_ns > 1000 {
                (total_duration_ns as f64
                    * (final_prompt_tokens as f64
                        / (final_prompt_tokens + final_eval_tokens) as f64)) as u64
            } else {
                total_duration_ns / TIMING_PROMPT_RATIO
            };

        let eval_duration_ns =
            if final_prompt_tokens + final_eval_tokens > 0 && total_duration_ns > 1000 {
                total_duration_ns - prompt_eval_duration_ns
            } else {
                total_duration_ns / TIMING_EVAL_RATIO
            };

        Self {
            total_duration: total_duration_ns,
            load_duration: DEFAULT_LOAD_DURATION_NS,
            prompt_eval_count: final_prompt_tokens,
            prompt_eval_duration: prompt_eval_duration_ns.max(1),
            eval_count: final_eval_tokens,
            eval_duration: eval_duration_ns.max(1),
        }
    }
}

pub struct ResponseTransformer;

impl ResponseTransformer {
    pub fn convert_to_ollama_chat(
        lm_response: &Value,
        model_ollama_name: &str,
        message_count_for_estimation: usize,
        start_time: Instant,
    ) -> Value {
        let content = extract_chat_content(lm_response);
        let thinking = extract_reasoning_content(lm_response);

        let timing = TimingInfo::from_native_stats(
            lm_response,
            start_time,
            (message_count_for_estimation * 10).max(1) as u64,
            estimate_token_count(&content),
        );

        let done_reason = extract_finish_reason(lm_response).unwrap_or("stop");
        let mut ollama_message = json!({
            "role": "assistant",
            "content": content
        });

        if let Some(ref thinking_str) = thinking
            && let Some(msg_obj) = ollama_message.as_object_mut()
        {
            msg_obj.insert("thinking".to_string(), json!(thinking_str));
        }

        if let Some(tool_calls) = lm_response
            .get("choices")
            .and_then(|c| c.as_array()?.first())
            .and_then(|choice| choice.get("message")?.get("tool_calls"))
            .and_then(|tc| tc.as_array())
            && !tool_calls.is_empty()
            && let Some(msg_obj) = ollama_message.as_object_mut()
        {
            msg_obj.insert(
                "tool_calls".to_string(),
                convert_tool_calls_to_ollama(tool_calls),
            );
        }

        // GAP B: images — vision is input-only in LM Studio; assistants never generate
        // image tokens. Forward any upstream image data if present, otherwise omit the
        // field entirely (schema says optional, not nullable).
        let upstream_images = lm_response
            .get("choices")
            .and_then(|c| c.as_array()?.first())
            .and_then(|choice| choice.get("message")?.get("images"))
            .and_then(|imgs| imgs.as_array())
            .filter(|imgs| !imgs.is_empty());

        if let Some(imgs) = upstream_images
            && let Some(msg_obj) = ollama_message.as_object_mut()
        {
            msg_obj.insert("images".to_string(), json!(imgs));
        }

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
            "eval_duration": timing.eval_duration
        });

        // GAP A: logprobs — Ollama expects array<Logprob>; OpenAI wraps the same items in
        // {content: [...]}. Extract .content so the shapes align.
        if let Some(logprobs) = lm_response
            .get("choices")
            .and_then(|c| c.as_array()?.first())
            .and_then(|choice| choice.get("logprobs"))
            .filter(|lp| !lp.is_null())
            .and_then(|lp| lp.get("content"))
            .filter(|content| !content.is_null())
            && let Some(obj) = response.as_object_mut()
        {
            obj.insert("logprobs".to_string(), logprobs.clone());
        }

        response
    }

    pub fn convert_to_ollama_generate(
        lm_response: &Value,
        model_ollama_name: &str,
        prompt_for_estimation: &str,
        start_time: Instant,
    ) -> Value {
        let content = Self::extract_completion_content(lm_response);
        let thinking = extract_completion_thinking(lm_response);

        let timing = TimingInfo::from_native_stats(
            lm_response,
            start_time,
            estimate_token_count(prompt_for_estimation),
            estimate_token_count(&content),
        );

        let done_reason = extract_finish_reason(lm_response).unwrap_or("stop");
        // `context` (token-ID encoding of conversation) is deprecated in Ollama and
        // cannot be synthesized from LM Studio responses, so omit it rather than
        // emitting a misleading empty array that breaks legacy context-chaining
        // clients.
        let mut response_obj = json!({
            "model": model_ollama_name,
            "created_at": chrono::Utc::now().to_rfc3339(),
            "response": content,
            "done": true,
            "done_reason": done_reason,
            "total_duration": timing.total_duration,
            "load_duration": timing.load_duration,
            "prompt_eval_count": timing.prompt_eval_count,
            "prompt_eval_duration": timing.prompt_eval_duration,
            "eval_count": timing.eval_count,
            "eval_duration": timing.eval_duration
        });

        if let Some(ref t) = thinking
            && let Some(obj) = response_obj.as_object_mut()
        {
            obj.insert("thinking".to_string(), json!(t));
        }

        response_obj
    }

    pub fn convert_to_ollama_embeddings(
        lm_response: &Value,
        model_ollama_name: &str,
        start_time: Instant,
    ) -> Value {
        let embeddings = Self::extract_embeddings(lm_response);

        let estimated_input_tokens = 10;
        let estimated_output_tokens = embeddings.len().max(1) as u64;

        let timing = TimingInfo::from_native_stats(
            lm_response,
            start_time,
            estimated_input_tokens,
            estimated_output_tokens,
        );

        json!({
            "model": model_ollama_name,
            "embeddings": embeddings,
            "total_duration": timing.total_duration,
            "load_duration": timing.load_duration,
            "prompt_eval_count": timing.prompt_eval_count
        })
    }

    fn extract_completion_content(lm_response: &Value) -> String {
        lm_response
            .get("choices")
            .and_then(|c| c.as_array()?.first())
            .and_then(|choice| {
                if let Some(text) = choice.get("text").and_then(|t| t.as_str()) {
                    Some(text.to_string())
                } else {
                    choice
                        .get("message")
                        .and_then(|msg| msg.get("content"))
                        .and_then(|content| content.as_str())
                        .map(|content| content.to_string())
                }
            })
            .unwrap_or_default()
    }

    fn extract_embeddings(lm_response: &Value) -> Vec<Value> {
        lm_response
            .get("data")
            .and_then(|d| d.as_array())
            .map(|data_array| {
                data_array
                    .iter()
                    .filter_map(|item| item.get("embedding").cloned())
                    .collect()
            })
            .unwrap_or_default()
    }
}

fn extract_chat_content(lm_response: &Value) -> String {
    lm_response
        .get("choices")
        .and_then(|c| c.as_array()?.first())
        .and_then(|choice| choice.get("message")?.get("content")?.as_str())
        .unwrap_or("")
        .to_string()
}

fn extract_reasoning_content(lm_response: &Value) -> Option<String> {
    let s = lm_response
        .get("choices")
        .and_then(|c| c.as_array()?.first())
        .and_then(|choice| choice.get("message")?.get("reasoning")?.as_str())?;
    if s.is_empty() {
        None
    } else {
        Some(s.to_string())
    }
}

fn extract_completion_thinking(lm_response: &Value) -> Option<String> {
    let choice = lm_response
        .get("choices")
        .and_then(|c| c.as_array()?.first())?;
    let s = choice
        .get("reasoning")
        .or_else(|| choice.get("thinking"))
        .and_then(|v| v.as_str())?;
    if s.is_empty() {
        None
    } else {
        Some(s.to_string())
    }
}

pub fn estimate_token_count(text: &str) -> u64 {
    if text.is_empty() {
        return 0;
    }
    ((text.len() as f64) * TOKEN_TO_CHAR_RATIO).ceil() as u64
}

pub fn extract_finish_reason(lm_response: &Value) -> Option<&str> {
    lm_response
        .get("choices")
        .and_then(|c| c.as_array()?.first())
        .and_then(|choice| choice.get("finish_reason"))
        .and_then(|reason| reason.as_str())
}

/// Convert an OpenAI-format `tool_calls` array to the Ollama format.
///
/// OpenAI represents each tool call as:
/// ```json
/// {"id": "call_abc", "type": "function", "function": {"name": "fn", "arguments": "{\"k\":\"v\"}"}}
/// ```
/// where `arguments` is a **JSON string**.
///
/// Ollama expects:
/// ```json
/// {"function": {"name": "fn", "arguments": {"k": "v"}}}
/// ```
/// where `arguments` is a **JSON object** and the `id`/`type` wrapper fields are absent.
pub fn convert_tool_calls_to_ollama(tool_calls: &[Value]) -> Value {
    let converted: Vec<Value> = tool_calls
        .iter()
        .map(|tc| {
            let name = tc
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .unwrap_or("");

            let raw_args = tc
                .get("function")
                .and_then(|f| f.get("arguments"))
                .cloned()
                .unwrap_or(Value::Object(serde_json::Map::new()));

            // OpenAI serialises arguments as a JSON string; parse it back into an object.
            let arguments = match raw_args {
                Value::String(ref s) => {
                    serde_json::from_str(s).unwrap_or(Value::Object(serde_json::Map::new()))
                }
                other => other,
            };

            json!({
                "function": {
                    "name": name,
                    "arguments": arguments
                }
            })
        })
        .collect();

    json!(converted)
}

pub fn normalize_chat_messages(messages: &[Value], system_prompt: Option<&str>) -> Value {
    if let Some(system_text) = system_prompt {
        let already_has_system = messages.iter().any(|message| {
            message
                .get("role")
                .and_then(|role| role.as_str())
                .map(|role| role.eq_ignore_ascii_case("system"))
                .unwrap_or(false)
        });

        if already_has_system {
            json!(messages)
        } else {
            let mut combined = Vec::with_capacity(messages.len() + 1);
            combined.push(json!({
                "role": "system",
                "content": system_text,
            }));
            combined.extend(messages.iter().cloned());
            Value::Array(combined)
        }
    } else {
        json!(messages)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/lmstudio_response.rs"]
mod tests;
