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
        estimated_input_tokens: u64,
        estimated_output_tokens: u64,
    ) -> Self {
        if let Some(stats) = lm_response.get("stats") {
            let generation_time = stats
                .get("generation_time")
                .and_then(|t| t.as_f64())
                .unwrap_or(0.001);

            let time_to_first_token = stats
                .get("time_to_first_token")
                .and_then(|t| t.as_f64())
                .unwrap_or(0.1);

            let actual_prompt_tokens = lm_response
                .get("usage")
                .and_then(|u| u.get("prompt_tokens"))
                .and_then(|t| t.as_u64())
                .unwrap_or(estimated_input_tokens);

            let actual_completion_tokens = lm_response
                .get("usage")
                .and_then(|u| u.get("completion_tokens"))
                .and_then(|t| t.as_u64())
                .unwrap_or(estimated_output_tokens);

            let generation_time_ns = (generation_time * 1_000_000_000.0) as u64;
            let ttft_ns = (time_to_first_token * 1_000_000_000.0) as u64;

            let prompt_eval_duration_ns = ttft_ns.max(1);
            let eval_duration_ns = generation_time_ns.saturating_sub(ttft_ns).max(1);
            let total_duration_ns =
                generation_time_ns.max(prompt_eval_duration_ns + eval_duration_ns);

            return Self {
                total_duration: total_duration_ns,
                load_duration: DEFAULT_LOAD_DURATION_NS,
                prompt_eval_count: actual_prompt_tokens.max(1),
                prompt_eval_duration: prompt_eval_duration_ns,
                eval_count: actual_completion_tokens.max(1),
                eval_duration: eval_duration_ns,
            };
        }

        Self::from_legacy_estimation(
            Instant::now(),
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
        use_native_stats: bool,
    ) -> Value {
        let content = extract_chat_content(lm_response);
        let thinking = extract_reasoning_content(lm_response);

        let timing = if use_native_stats {
            TimingInfo::from_native_stats(
                lm_response,
                (message_count_for_estimation * 10).max(1) as u64,
                estimate_token_count(&content),
            )
        } else {
            let actual_prompt_tokens = lm_response
                .get("usage")
                .and_then(|u| u.get("prompt_tokens"))
                .and_then(|t| t.as_u64());
            let actual_completion_tokens = lm_response
                .get("usage")
                .and_then(|u| u.get("completion_tokens"))
                .and_then(|t| t.as_u64());

            TimingInfo::from_legacy_estimation(
                start_time,
                (message_count_for_estimation * 10).max(1) as u64,
                estimate_token_count(&content),
                actual_prompt_tokens,
                actual_completion_tokens,
            )
        };

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

        json!({
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
        })
    }

    pub fn convert_to_ollama_generate(
        lm_response: &Value,
        model_ollama_name: &str,
        prompt_for_estimation: &str,
        start_time: Instant,
        use_native_stats: bool,
    ) -> Value {
        let content = Self::extract_completion_content(lm_response);
        let thinking = extract_completion_thinking(lm_response);

        let timing = if use_native_stats {
            TimingInfo::from_native_stats(
                lm_response,
                estimate_token_count(prompt_for_estimation),
                estimate_token_count(&content),
            )
        } else {
            let actual_prompt_tokens = lm_response
                .get("usage")
                .and_then(|u| u.get("prompt_tokens"))
                .and_then(|t| t.as_u64());
            let actual_completion_tokens = lm_response
                .get("usage")
                .and_then(|u| u.get("completion_tokens"))
                .and_then(|t| t.as_u64());

            TimingInfo::from_legacy_estimation(
                start_time,
                estimate_token_count(prompt_for_estimation),
                estimate_token_count(&content),
                actual_prompt_tokens,
                actual_completion_tokens,
            )
        };

        let done_reason = extract_finish_reason(lm_response).unwrap_or("stop");
        let mut response_obj = json!({
            "model": model_ollama_name,
            "created_at": chrono::Utc::now().to_rfc3339(),
            "response": content,
            "done": true,
            "done_reason": done_reason,
            "context": [],
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
        use_native_stats: bool,
    ) -> Value {
        let embeddings = Self::extract_embeddings(lm_response);

        let estimated_input_tokens = 10;
        let estimated_output_tokens = embeddings.len().max(1) as u64;

        let timing = if use_native_stats {
            TimingInfo::from_native_stats(
                lm_response,
                estimated_input_tokens,
                estimated_output_tokens,
            )
        } else {
            let actual_prompt_tokens = lm_response
                .get("usage")
                .and_then(|u| u.get("prompt_tokens"))
                .and_then(|t| t.as_u64());

            TimingInfo::from_legacy_estimation(
                start_time,
                estimated_input_tokens,
                estimated_output_tokens,
                actual_prompt_tokens,
                None,
            )
        };

        json!({
            "model": model_ollama_name,
            "embeddings": embeddings,
            "total_duration": timing.total_duration,
            "load_duration": timing.load_duration,
            "prompt_eval_count": timing.prompt_eval_count,
            "prompt_eval_duration": timing.prompt_eval_duration
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
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::Instant;

    fn lm_chat_response(content: &str, reasoning: Option<&str>) -> serde_json::Value {
        let mut msg = json!({ "content": content });
        if let Some(r) = reasoning {
            msg.as_object_mut()
                .unwrap()
                .insert("reasoning".to_string(), json!(r));
        }
        json!({
            "choices": [{ "message": msg, "finish_reason": "stop" }],
            "usage": { "prompt_tokens": 10, "completion_tokens": 5 }
        })
    }

    fn lm_completion_response(text: &str, reasoning: Option<&str>) -> serde_json::Value {
        let mut choice = json!({ "text": text, "finish_reason": "stop" });
        if let Some(r) = reasoning {
            choice
                .as_object_mut()
                .unwrap()
                .insert("reasoning".to_string(), json!(r));
        }
        json!({
            "choices": [choice],
            "usage": { "prompt_tokens": 10, "completion_tokens": 5 }
        })
    }

    #[test]
    fn tool_calls_arguments_string_becomes_object() {
        let tool_calls = vec![json!({
            "id": "call_abc",
            "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"location\":\"London\"}"}
        })];
        let result = convert_tool_calls_to_ollama(&tool_calls);
        let first = &result.as_array().unwrap()[0];
        assert!(first.get("id").is_none(), "id should be stripped");
        assert!(first.get("type").is_none(), "type should be stripped");
        let args = first.get("function").unwrap().get("arguments").unwrap();
        assert!(
            args.is_object(),
            "arguments should be an object, got {:?}",
            args
        );
        assert_eq!(
            args.get("location").and_then(|v| v.as_str()),
            Some("London")
        );
    }

    #[test]
    fn tool_calls_arguments_already_object_is_preserved() {
        let tool_calls = vec![json!({
            "function": {"name": "fn", "arguments": {"key": "val"}}
        })];
        let result = convert_tool_calls_to_ollama(&tool_calls);
        let first = &result.as_array().unwrap()[0];
        let args = first.get("function").unwrap().get("arguments").unwrap();
        assert!(args.is_object());
        assert_eq!(args.get("key").and_then(|v| v.as_str()), Some("val"));
    }

    #[test]
    fn tool_calls_end_to_end_in_chat_response() {
        let lm = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "my_tool", "arguments": "{\"x\":1}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        });
        let result =
            ResponseTransformer::convert_to_ollama_chat(&lm, "m", 2, Instant::now(), false);
        let msg = result.get("message").unwrap();
        let tc = msg.get("tool_calls").unwrap().as_array().unwrap();
        assert_eq!(tc.len(), 1);
        let args = tc[0].get("function").unwrap().get("arguments").unwrap();
        assert!(args.is_object(), "expected object, got {:?}", args);
        assert_eq!(args.get("x").and_then(|v| v.as_i64()), Some(1));
        assert!(tc[0].get("id").is_none());
    }

    #[test]
    fn chat_response_thinking_in_message_not_content() {
        let lm = lm_chat_response("The answer is 42", Some("Let me think..."));
        let result =
            ResponseTransformer::convert_to_ollama_chat(&lm, "mymodel", 2, Instant::now(), false);
        let msg = result.get("message").unwrap();
        assert_eq!(
            msg.get("content").and_then(|v| v.as_str()),
            Some("The answer is 42")
        );
        assert_eq!(
            msg.get("thinking").and_then(|v| v.as_str()),
            Some("Let me think...")
        );
        // must not be merged into content
        assert!(
            !msg.get("content")
                .unwrap()
                .as_str()
                .unwrap()
                .contains("Reasoning")
        );
    }

    #[test]
    fn chat_response_no_thinking_field_when_absent() {
        let lm = lm_chat_response("The answer is 42", None);
        let result =
            ResponseTransformer::convert_to_ollama_chat(&lm, "mymodel", 2, Instant::now(), false);
        let msg = result.get("message").unwrap();
        assert!(msg.get("thinking").is_none());
    }

    #[test]
    fn generate_response_thinking_top_level() {
        let lm = lm_completion_response("42", Some("Let me reason"));
        let result = ResponseTransformer::convert_to_ollama_generate(
            &lm,
            "mymodel",
            "what is the answer?",
            Instant::now(),
            false,
        );
        assert_eq!(result.get("response").and_then(|v| v.as_str()), Some("42"));
        assert_eq!(
            result.get("thinking").and_then(|v| v.as_str()),
            Some("Let me reason")
        );
    }

    #[test]
    fn generate_response_no_thinking_field_when_absent() {
        let lm = lm_completion_response("42", None);
        let result = ResponseTransformer::convert_to_ollama_generate(
            &lm,
            "mymodel",
            "q",
            Instant::now(),
            false,
        );
        assert!(result.get("thinking").is_none());
    }
}
