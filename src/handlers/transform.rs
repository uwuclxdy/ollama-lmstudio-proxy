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
        let content = Self::extract_chat_content_with_reasoning(lm_response);

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

        if let Some(tool_calls) = lm_response
            .get("choices")
            .and_then(|c| c.as_array()?.first())
            .and_then(|choice| choice.get("message")?.get("tool_calls"))
            .and_then(|tc| tc.as_array())
            && !tool_calls.is_empty()
            && let Some(msg_obj) = ollama_message.as_object_mut()
        {
            msg_obj.insert("tool_calls".to_string(), json!(tool_calls));
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
        json!({
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
        })
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

    fn extract_chat_content_with_reasoning(lm_response: &Value) -> String {
        let base_content = lm_response
            .get("choices")
            .and_then(|c| c.as_array()?.first())
            .and_then(|choice| choice.get("message")?.get("content")?.as_str())
            .unwrap_or("")
            .to_string();

        if let Some(reasoning) = lm_response
            .get("choices")
            .and_then(|c| c.as_array()?.first())
            .and_then(|choice| choice.get("message")?.get("reasoning")?.as_str())
            && !reasoning.is_empty()
        {
            return format!(
                "**Reasoning:**\n{}\n\n**Answer:**\n{}",
                reasoning, base_content
            );
        }
        base_content
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
