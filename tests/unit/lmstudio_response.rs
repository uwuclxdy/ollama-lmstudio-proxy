use super::*;
use serde_json::json;
use std::time::{Duration, Instant};

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
        "index": 0,
        "id": "call_abc",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"location\":\"London\"}"}
    })];
    let result = convert_tool_calls_to_ollama(&tool_calls);
    let first = &result.as_array().unwrap()[0];
    assert!(first.get("id").is_none(), "id should be stripped");
    assert!(first.get("type").is_none(), "type should be stripped");
    let function = first.get("function").unwrap();
    assert_eq!(function.get("index"), Some(&json!(0)));
    let args = function.get("arguments").unwrap();
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
                    "index": 0,
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "my_tool", "arguments": "{\"x\":1}"}
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5}
    });
    let result = ResponseTransformer::convert_to_ollama_chat(&lm, "m", 2, Instant::now());
    let msg = result.get("message").unwrap();
    let tc = msg.get("tool_calls").unwrap().as_array().unwrap();
    assert_eq!(tc.len(), 1);
    let function = tc[0].get("function").unwrap();
    assert_eq!(function.get("index"), Some(&json!(0)));
    let args = function.get("arguments").unwrap();
    assert!(args.is_object(), "expected object, got {:?}", args);
    assert_eq!(args.get("x").and_then(|v| v.as_i64()), Some(1));
    assert!(tc[0].get("id").is_none());
}

#[test]
fn chat_response_thinking_in_message_not_content() {
    let lm = lm_chat_response("The answer is 42", Some("Let me think..."));
    let result = ResponseTransformer::convert_to_ollama_chat(&lm, "mymodel", 2, Instant::now());
    let msg = result.get("message").unwrap();
    assert_eq!(
        msg.get("content").and_then(|v| v.as_str()),
        Some("The answer is 42")
    );
    assert_eq!(
        msg.get("thinking").and_then(|v| v.as_str()),
        Some("Let me think...")
    );
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
    let result = ResponseTransformer::convert_to_ollama_chat(&lm, "mymodel", 2, Instant::now());
    let msg = result.get("message").unwrap();
    assert!(msg.get("thinking").is_none());
}

#[test]
fn chat_response_thinking_from_reasoning_content_field() {
    // LM Studio's `/api/v0/chat/completions` puts reasoning under
    // `reasoning_content`, not `reasoning`. The default chat path must surface it.
    let lm = json!({
        "choices": [{
            "message": { "content": "ok", "reasoning_content": "thinking via v0" },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 10, "completion_tokens": 5 }
    });
    let result = ResponseTransformer::convert_to_ollama_chat(&lm, "mymodel", 2, Instant::now());
    let msg = result.get("message").unwrap();
    assert_eq!(
        msg.get("thinking").and_then(|v| v.as_str()),
        Some("thinking via v0"),
        "reasoning_content must map to the Ollama thinking field"
    );
}

#[test]
fn generate_response_thinking_top_level() {
    let lm = lm_completion_response("42", Some("Let me reason"));
    let result = ResponseTransformer::convert_to_ollama_generate(
        &lm,
        "mymodel",
        "what is the answer?",
        Instant::now(),
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
    let result =
        ResponseTransformer::convert_to_ollama_generate(&lm, "mymodel", "q", Instant::now());
    assert!(result.get("thinking").is_none());
}

// =========================================================================
// generate_context (from translation_generate_context)
// =========================================================================

#[test]
fn generate_response_omits_context_absent_from_schema() {
    // `context` is not part of the Ollama OpenAPI GenerateResponse schema.
    // LM Studio does not return token IDs, so the field must not appear.
    let lm = json!({
        "choices": [{
            "text": "hello",
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1}
    });
    let result = ResponseTransformer::convert_to_ollama_generate(
        &lm,
        "model",
        "prompt text",
        Instant::now(),
    );
    assert!(result.get("context").is_none(), "got {}", result);
}

#[test]
fn chat_response_does_not_emit_context() {
    // /api/chat never had a `context` field — guard against accidental leakage.
    let lm = json!({
        "choices": [{
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1}
    });
    let result = ResponseTransformer::convert_to_ollama_chat(&lm, "model", 1, Instant::now());
    assert!(result.get("context").is_none(), "got {}", result);
}

// =========================================================================
// convert_tool_calls_to_ollama (from translation_misc_helpers)
// =========================================================================

#[test]
fn tool_calls_empty_input_yields_empty_output() {
    let result = convert_tool_calls_to_ollama(&[]);
    assert_eq!(result, json!([]));
}

#[test]
fn tool_calls_missing_function_field_does_not_panic() {
    let tool_calls = vec![json!({ "id": "call_x", "type": "function" })];
    let result = convert_tool_calls_to_ollama(&tool_calls);
    let first = &result.as_array().unwrap()[0];
    assert_eq!(first["function"]["name"], json!(""));
    assert_eq!(first["function"]["arguments"], json!({}));
}

#[test]
fn tool_calls_missing_arguments_field_defaults_to_empty_object() {
    let tool_calls = vec![json!({
        "function": { "name": "fn_no_args" }
    })];
    let result = convert_tool_calls_to_ollama(&tool_calls);
    let args = &result.as_array().unwrap()[0]["function"]["arguments"];
    assert!(args.is_object(), "arguments must be an object, got {args}");
    assert_eq!(args, &json!({}));
}

#[test]
fn tool_calls_malformed_arguments_string_falls_back_to_empty_object() {
    let tool_calls = vec![json!({
        "function": { "name": "fn", "arguments": "not valid json {{" }
    })];
    let result = convert_tool_calls_to_ollama(&tool_calls);
    let args = &result.as_array().unwrap()[0]["function"]["arguments"];
    assert_eq!(args, &json!({}));
}

#[test]
fn tool_calls_multiple_preserves_order_and_strips_wrappers() {
    let tool_calls = vec![
        json!({
            "id": "call_1",
            "type": "function",
            "function": { "name": "first", "arguments": "{\"a\":1}" }
        }),
        json!({
            "id": "call_2",
            "type": "function",
            "function": { "name": "second", "arguments": "{\"b\":2}" }
        }),
    ];
    let result = convert_tool_calls_to_ollama(&tool_calls);
    let arr = result.as_array().unwrap();
    assert_eq!(arr.len(), 2);
    assert_eq!(arr[0]["function"]["name"], json!("first"));
    assert_eq!(arr[1]["function"]["name"], json!("second"));
    assert_eq!(arr[0]["function"]["arguments"]["a"], json!(1));
    assert_eq!(arr[1]["function"]["arguments"]["b"], json!(2));
    for entry in arr {
        assert!(entry.get("id").is_none(), "id wrapper must be stripped");
        assert!(entry.get("type").is_none(), "type wrapper must be stripped");
    }
}

// =========================================================================
// TimingInfo (from translation_timing)
// =========================================================================

/// LM Studio `/api/v0/*` response: `time_to_first_token` is the prompt-processing
/// phase, `generation_time` is the post-TTFT output-generation phase. Both are
/// SEPARATE phases and must NOT be subtracted from each other.
#[test]
fn timing_from_native_v0_stats_does_not_subtract_ttft() {
    let lm = json!({
        "choices": [{
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 24, "completion_tokens": 53, "total_tokens": 77},
        "stats": {
            "tokens_per_second": 51.43709529007664,
            "time_to_first_token": 0.111,
            "generation_time": 0.954,
            "stop_reason": "eosFound"
        }
    });

    let timing = TimingInfo::from_native_stats(&lm, Instant::now(), 24, 53);

    let ttft_ns = 111_000_000u64;
    let gen_ns = 954_000_000u64;

    assert_eq!(
        timing.prompt_eval_duration, ttft_ns,
        "prompt_eval_duration must equal time_to_first_token (got {}, want {})",
        timing.prompt_eval_duration, ttft_ns
    );
    assert_eq!(
        timing.eval_duration, gen_ns,
        "eval_duration must equal generation_time, NOT generation_time - ttft (got {}, want {})",
        timing.eval_duration, gen_ns
    );
    assert_eq!(
        timing.total_duration,
        ttft_ns + gen_ns,
        "total_duration must equal ttft + generation_time (got {}, want {})",
        timing.total_duration,
        ttft_ns + gen_ns
    );
    assert_eq!(timing.prompt_eval_count, 24);
    assert_eq!(timing.eval_count, 53);
}

/// LM Studio `/v1/responses` and newer endpoints expose `time_to_first_token_seconds`
/// and `model_load_time_seconds`.
#[test]
fn timing_from_native_v1_responses_stats() {
    let lm = json!({
        "model_instance_id": "ibm/granite-4-micro",
        "output": [{"type": "message", "content": "hi"}],
        "stats": {
            "input_tokens": 646,
            "total_output_tokens": 586,
            "reasoning_output_tokens": 0,
            "tokens_per_second": 29.753900615398926,
            "time_to_first_token_seconds": 1.088,
            "model_load_time_seconds": 2.656
        }
    });

    let timing = TimingInfo::from_native_stats(&lm, Instant::now(), 0, 0);

    let ttft_ns = 1_088_000_000u64;
    let expected_gen_ns = ((586.0_f64 / 29.753900615398926_f64) * 1_000_000_000.0) as u64;
    let load_ns = 2_656_000_000u64;

    assert_eq!(
        timing.prompt_eval_duration, ttft_ns,
        "prompt_eval_duration must equal time_to_first_token_seconds"
    );
    let drift = timing.eval_duration.abs_diff(expected_gen_ns);
    assert!(
        drift < 5_000_000,
        "eval_duration ≈ output_tokens / tokens_per_second; got {}, want ~{} (drift {})",
        timing.eval_duration,
        expected_gen_ns,
        drift
    );
    assert_eq!(
        timing.load_duration, load_ns,
        "load_duration must come from model_load_time_seconds when present"
    );
    assert_eq!(timing.prompt_eval_count, 646);
    assert_eq!(timing.eval_count, 586);
}

// =========================================================================
// Response shape (from translation_response_shape)
// =========================================================================

/// Per ollama.md /api/chat response: a complete non-streaming response must include
/// `model`, `created_at` (RFC3339), `message.role == "assistant"`, `message.content`,
/// `done == true`, `done_reason`, and all six timing fields.
#[test]
fn chat_response_contains_full_ollama_shape() {
    let lm = json!({
        "choices": [{
            "message": {"role": "assistant", "content": "Hello world"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3}
    });
    let result = ResponseTransformer::convert_to_ollama_chat(&lm, "llama3", 1, Instant::now());

    assert_eq!(result.get("model").and_then(|v| v.as_str()), Some("llama3"));

    let created = result
        .get("created_at")
        .and_then(|v| v.as_str())
        .expect("created_at must be a string");
    assert!(
        chrono::DateTime::parse_from_rfc3339(created).is_ok(),
        "created_at must parse as RFC3339, got {created:?}"
    );

    let msg = result.get("message").expect("message field required");
    assert_eq!(
        msg.get("role").and_then(|v| v.as_str()),
        Some("assistant"),
        "message.role must always be \"assistant\""
    );
    assert_eq!(
        msg.get("content").and_then(|v| v.as_str()),
        Some("Hello world")
    );

    assert_eq!(result.get("done").and_then(|v| v.as_bool()), Some(true));
    assert_eq!(
        result.get("done_reason").and_then(|v| v.as_str()),
        Some("stop")
    );

    for key in [
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
    ] {
        assert!(
            result.get(key).and_then(|v| v.as_u64()).is_some(),
            "{key} must be present as u64"
        );
    }
}

#[test]
fn chat_response_uses_usage_counts_when_present() {
    let lm = json!({
        "choices": [{
            "message": {"role": "assistant", "content": "x"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 42, "completion_tokens": 17}
    });
    let result = ResponseTransformer::convert_to_ollama_chat(&lm, "m", 1, Instant::now());

    assert_eq!(
        result.get("prompt_eval_count").and_then(|v| v.as_u64()),
        Some(42),
        "prompt_eval_count must echo usage.prompt_tokens"
    );
    assert_eq!(
        result.get("eval_count").and_then(|v| v.as_u64()),
        Some(17),
        "eval_count must echo usage.completion_tokens"
    );
}

#[test]
fn chat_response_falls_back_to_estimates_without_usage() {
    let lm = json!({
        "choices": [{
            "message": {"role": "assistant", "content": "abcd"},
            "finish_reason": "stop"
        }]
    });
    let result = ResponseTransformer::convert_to_ollama_chat(&lm, "m", 2, Instant::now());

    let prompt = result
        .get("prompt_eval_count")
        .and_then(|v| v.as_u64())
        .unwrap();
    let eval = result.get("eval_count").and_then(|v| v.as_u64()).unwrap();
    assert!(prompt >= 1, "prompt_eval_count must be ≥ 1, got {prompt}");
    assert!(eval >= 1, "eval_count must be ≥ 1, got {eval}");
}

#[test]
fn chat_response_done_reason_passthrough() {
    for reason in ["stop", "length"] {
        let lm = json!({
            "choices": [{
                "message": {"role": "assistant", "content": "x"},
                "finish_reason": reason
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}
        });
        let result = ResponseTransformer::convert_to_ollama_chat(&lm, "m", 1, Instant::now());
        assert_eq!(
            result.get("done_reason").and_then(|v| v.as_str()),
            Some(reason),
            "finish_reason={reason} must propagate as done_reason"
        );
    }
}

#[test]
fn chat_response_translates_tool_calls_to_stop() {
    let lm = json!({
        "choices": [{
            "message": {"role": "assistant", "content": "x"},
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1}
    });
    let result = ResponseTransformer::convert_to_ollama_chat(&lm, "m", 1, Instant::now());
    assert_eq!(
        result.get("done_reason").and_then(|v| v.as_str()),
        Some("stop"),
        "OpenAI tool_calls finish_reason must translate to Ollama stop"
    );
}

#[test]
fn chat_response_omits_done_reason_for_unknown_finish_reason() {
    let lm = json!({
        "choices": [{
            "message": {"role": "assistant", "content": "x"},
            "finish_reason": "some_unknown_reason"
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1}
    });
    let result = ResponseTransformer::convert_to_ollama_chat(&lm, "m", 1, Instant::now());
    assert!(
        result.get("done_reason").is_none(),
        "unknown finish_reason must be omitted, not propagated"
    );
}

#[test]
fn generate_response_translates_tool_calls_to_stop() {
    let lm = json!({
        "choices": [{"text": "hi", "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 1}
    });
    let result =
        ResponseTransformer::convert_to_ollama_generate(&lm, "m", "prompt", Instant::now());
    assert_eq!(
        result.get("done_reason").and_then(|v| v.as_str()),
        Some("stop"),
        "generate path must also translate tool_calls to stop"
    );
}

#[test]
fn generate_response_omits_done_reason_for_unknown_finish_reason() {
    let lm = json!({
        "choices": [{"text": "hi", "finish_reason": "garbage"}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 1}
    });
    let result =
        ResponseTransformer::convert_to_ollama_generate(&lm, "m", "prompt", Instant::now());
    assert!(
        result.get("done_reason").is_none(),
        "unknown finish_reason must be omitted on generate path"
    );
}

#[test]
fn chat_response_null_content_with_tool_calls_becomes_empty_string() {
    let lm = json!({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"}
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1}
    });
    let result = ResponseTransformer::convert_to_ollama_chat(&lm, "m", 1, Instant::now());
    let msg = result.get("message").unwrap();
    let content = msg
        .get("content")
        .expect("content must be present")
        .as_str()
        .expect("content must be a string, not null");
    assert_eq!(
        content, "",
        "null upstream content must become empty string"
    );
    let tcs = msg
        .get("tool_calls")
        .and_then(|v| v.as_array())
        .expect("tool_calls array must be present");
    assert_eq!(tcs.len(), 1);
}

#[test]
fn chat_response_forwards_multiple_tool_calls() {
    let lm = json!({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "a", "type": "function",
                     "function": {"name": "fn1", "arguments": "{\"a\":1}"}},
                    {"id": "b", "type": "function",
                     "function": {"name": "fn2", "arguments": "{\"b\":2}"}},
                    {"id": "c", "type": "function",
                     "function": {"name": "fn3", "arguments": "{}"}}
                ]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1}
    });
    let result = ResponseTransformer::convert_to_ollama_chat(&lm, "m", 1, Instant::now());
    let tcs = result
        .get("message")
        .unwrap()
        .get("tool_calls")
        .and_then(|v| v.as_array())
        .expect("tool_calls array must be present");
    assert_eq!(tcs.len(), 3);
    let names: Vec<&str> = tcs
        .iter()
        .map(|t| {
            t.get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .unwrap()
        })
        .collect();
    assert_eq!(names, vec!["fn1", "fn2", "fn3"]);
}

#[test]
fn chat_response_role_is_always_assistant() {
    let lm = json!({
        "choices": [{
            "message": {"role": "user", "content": "oddly tagged"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1}
    });
    let result = ResponseTransformer::convert_to_ollama_chat(&lm, "m", 1, Instant::now());
    assert_eq!(
        result
            .get("message")
            .unwrap()
            .get("role")
            .and_then(|v| v.as_str()),
        Some("assistant")
    );
}

#[test]
fn generate_response_contains_full_ollama_shape() {
    let lm = json!({
        "choices": [{"text": "hi", "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 1}
    });
    let result =
        ResponseTransformer::convert_to_ollama_generate(&lm, "model-x", "prompt", Instant::now());
    assert_eq!(
        result.get("model").and_then(|v| v.as_str()),
        Some("model-x")
    );
    let created = result.get("created_at").and_then(|v| v.as_str()).unwrap();
    assert!(chrono::DateTime::parse_from_rfc3339(created).is_ok());
    assert_eq!(result.get("response").and_then(|v| v.as_str()), Some("hi"));
    assert_eq!(result.get("done").and_then(|v| v.as_bool()), Some(true));
    assert_eq!(
        result.get("done_reason").and_then(|v| v.as_str()),
        Some("stop")
    );
    for key in [
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
    ] {
        assert!(result.get(key).is_some(), "{key} must be present");
    }
}

#[test]
fn generate_response_extracts_completion_text() {
    let lm = json!({
        "choices": [{"text": "completion-style output", "finish_reason": "stop"}]
    });
    let result = ResponseTransformer::convert_to_ollama_generate(&lm, "m", "p", Instant::now());
    assert_eq!(
        result.get("response").and_then(|v| v.as_str()),
        Some("completion-style output")
    );
}

#[test]
fn generate_response_falls_back_to_message_content() {
    let lm = json!({
        "choices": [{
            "message": {"role": "assistant", "content": "chat-shaped output"},
            "finish_reason": "stop"
        }]
    });
    let result = ResponseTransformer::convert_to_ollama_generate(&lm, "m", "p", Instant::now());
    assert_eq!(
        result.get("response").and_then(|v| v.as_str()),
        Some("chat-shaped output")
    );
}

#[test]
fn generate_response_omits_thinking_when_empty_or_missing() {
    // missing
    let lm = json!({"choices": [{"text": "x", "finish_reason": "stop"}]});
    let r = ResponseTransformer::convert_to_ollama_generate(&lm, "m", "p", Instant::now());
    assert!(r.get("thinking").is_none(), "no field if absent");

    // empty string
    let lm = json!({"choices": [{"text": "x", "reasoning": "", "finish_reason": "stop"}]});
    let r = ResponseTransformer::convert_to_ollama_generate(&lm, "m", "p", Instant::now());
    assert!(
        r.get("thinking").is_none(),
        "empty reasoning must not emit thinking field"
    );

    // empty `thinking` field on the choice
    let lm = json!({"choices": [{"text": "x", "thinking": "", "finish_reason": "stop"}]});
    let r = ResponseTransformer::convert_to_ollama_generate(&lm, "m", "p", Instant::now());
    assert!(r.get("thinking").is_none());
}

#[test]
fn generate_response_emits_thinking_from_choice_thinking_field() {
    let lm = json!({
        "choices": [{"text": "answer", "thinking": "let me think",
                     "finish_reason": "stop"}]
    });
    let r = ResponseTransformer::convert_to_ollama_generate(&lm, "m", "p", Instant::now());
    assert_eq!(
        r.get("thinking").and_then(|v| v.as_str()),
        Some("let me think")
    );
}

#[test]
fn embeddings_response_shape_matches_ollama_embed_spec() {
    let lm = json!({
        "model": "all-minilm",
        "data": [
            {"embedding": [0.1, 0.2, 0.3]},
            {"embedding": [0.4, 0.5, 0.6]}
        ],
        "usage": {"prompt_tokens": 8}
    });
    let result =
        ResponseTransformer::convert_to_ollama_embeddings(&lm, "all-minilm", Instant::now());
    assert_eq!(
        result.get("model").and_then(|v| v.as_str()),
        Some("all-minilm")
    );
    let embeds = result
        .get("embeddings")
        .and_then(|v| v.as_array())
        .expect("embeddings array must exist");
    assert_eq!(embeds.len(), 2);
    assert_eq!(
        embeds[0].as_array().unwrap()[0].as_f64(),
        Some(0.1),
        "first vector first dim"
    );
    for key in ["total_duration", "load_duration", "prompt_eval_count"] {
        assert!(result.get(key).is_some(), "{key} must be present per spec");
    }
    for key in ["prompt_eval_duration", "eval_count", "eval_duration"] {
        assert!(
            result.get(key).is_none(),
            "{key} must NOT be present on /api/embed (got {result})"
        );
    }
}

#[test]
fn embeddings_response_empty_data_yields_empty_embeddings() {
    let lm = json!({"data": [], "usage": {"prompt_tokens": 0}});
    let result = ResponseTransformer::convert_to_ollama_embeddings(&lm, "m", Instant::now());
    let embeds = result
        .get("embeddings")
        .and_then(|v| v.as_array())
        .expect("embeddings array must exist");
    assert!(embeds.is_empty());
}

#[test]
fn normalize_no_system_prompt_passthrough() {
    let msgs = vec![
        json!({"role": "user", "content": "hi"}),
        json!({"role": "assistant", "content": "hello"}),
    ];
    let out = normalize_chat_messages(&msgs, None);
    let arr = out.as_array().unwrap();
    assert_eq!(arr.len(), 2);
    assert_eq!(arr[0].get("role").and_then(|v| v.as_str()), Some("user"));
}

#[test]
fn normalize_prepends_system_when_absent() {
    let msgs = vec![json!({"role": "user", "content": "hi"})];
    let out = normalize_chat_messages(&msgs, Some("you are helpful"));
    let arr = out.as_array().unwrap();
    assert_eq!(arr.len(), 2);
    assert_eq!(arr[0].get("role").and_then(|v| v.as_str()), Some("system"));
    assert_eq!(
        arr[0].get("content").and_then(|v| v.as_str()),
        Some("you are helpful")
    );
    assert_eq!(arr[1].get("role").and_then(|v| v.as_str()), Some("user"));
}

#[test]
fn normalize_keeps_existing_system_message() {
    let msgs = vec![
        json!({"role": "system", "content": "original system"}),
        json!({"role": "user", "content": "hi"}),
    ];
    let out = normalize_chat_messages(&msgs, Some("override system"));
    let arr = out.as_array().unwrap();
    assert_eq!(arr.len(), 2, "must not duplicate system message");
    assert_eq!(
        arr[0].get("content").and_then(|v| v.as_str()),
        Some("original system"),
        "must not overwrite existing system message"
    );
}

#[test]
fn normalize_system_role_check_is_case_insensitive() {
    let msgs = vec![
        json!({"role": "System", "content": "cap S"}),
        json!({"role": "user", "content": "hi"}),
    ];
    let out = normalize_chat_messages(&msgs, Some("override"));
    let arr = out.as_array().unwrap();
    assert_eq!(
        arr.len(),
        2,
        "case-insensitive role check must detect existing system message"
    );
    assert_eq!(
        arr[0].get("content").and_then(|v| v.as_str()),
        Some("cap S")
    );
}

#[test]
fn estimate_token_count_empty_is_zero() {
    assert_eq!(estimate_token_count(""), 0);
}

#[test]
fn estimate_token_count_uses_ceil_quarter_length() {
    assert_eq!(estimate_token_count("a"), 1);
    assert_eq!(estimate_token_count("abcd"), 1);
    assert_eq!(estimate_token_count("abcde"), 2);
    assert_eq!(estimate_token_count("abcdefgh"), 2);
    assert_eq!(estimate_token_count("abcdefghi"), 3);
}

#[test]
fn extract_finish_reason_missing_choices_is_none() {
    let v = json!({"usage": {"prompt_tokens": 1}});
    assert!(extract_finish_reason(&v).is_none());
}

#[test]
fn extract_finish_reason_empty_choices_is_none() {
    let v = json!({"choices": []});
    assert!(extract_finish_reason(&v).is_none());
}

#[test]
fn extract_finish_reason_present() {
    let v = json!({"choices": [{"finish_reason": "length"}]});
    assert_eq!(extract_finish_reason(&v), Some("length"));
}

#[test]
fn timing_legacy_estimation_no_zero_fields() {
    let timing = TimingInfo::from_legacy_estimation(Instant::now(), 5, 5, None, None);
    assert!(timing.total_duration >= 1);
    assert!(timing.load_duration >= 1);
    assert!(timing.prompt_eval_count >= 1);
    assert!(timing.prompt_eval_duration >= 1);
    assert!(timing.eval_count >= 1);
    assert!(timing.eval_duration >= 1);
}

#[test]
fn timing_legacy_estimation_uses_actual_token_counts() {
    let timing = TimingInfo::from_legacy_estimation(Instant::now(), 1, 1, Some(100), Some(50));
    assert_eq!(
        timing.prompt_eval_count, 100,
        "actual_prompt_tokens must override estimate"
    );
    assert_eq!(
        timing.eval_count, 50,
        "actual_completion_tokens must override estimate"
    );
}

#[test]
fn timing_legacy_estimation_falls_back_to_estimates() {
    let timing = TimingInfo::from_legacy_estimation(Instant::now(), 30, 12, None, None);
    assert_eq!(timing.prompt_eval_count, 30);
    assert_eq!(timing.eval_count, 12);
}

#[test]
fn timing_stream_chunks_no_zero_fields() {
    let timing = TimingInfo::from_stream_chunks(Duration::from_millis(50), 12, None);
    assert!(timing.total_duration >= 1);
    assert!(timing.load_duration >= 1);
    assert!(timing.prompt_eval_count >= 1);
    assert!(timing.prompt_eval_duration >= 1);
    assert!(timing.eval_count >= 1);
    assert!(timing.eval_duration >= 1);
}

#[test]
fn timing_stream_chunks_uses_actual_completion_tokens_when_provided() {
    let timing = TimingInfo::from_stream_chunks(Duration::from_millis(50), 12, Some(77));
    assert_eq!(
        timing.eval_count, 77,
        "actual completion tokens must override chunk-count estimate"
    );
}

#[test]
fn timing_stream_chunks_uses_chunk_count_estimate_as_fallback() {
    let timing = TimingInfo::from_stream_chunks(Duration::from_millis(50), 9, None);
    assert_eq!(
        timing.eval_count, 9,
        "chunk-count estimate must be used as fallback"
    );
}

// =========================================================================
// GAP A: logprobs forwarding
// =========================================================================

#[test]
fn chat_response_logprobs_forwarded_when_upstream_provides_them() {
    let lm = json!({
        "choices": [{
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop",
            "logprobs": {
                "content": [
                    {"token": "hi", "logprob": -0.5, "top_logprobs": []}
                ]
            }
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1}
    });
    let result = ResponseTransformer::convert_to_ollama_chat(&lm, "m", 1, Instant::now());
    // Ollama schema: logprobs is array<Logprob>. OpenAI wraps items in {content: [...]}.
    // The proxy must extract .content so callers see a flat array.
    let logprobs = result
        .get("logprobs")
        .and_then(|v| v.as_array())
        .expect("logprobs must be a flat array (Ollama schema)");
    assert_eq!(logprobs.len(), 1);
    assert_eq!(
        logprobs[0].get("token").and_then(|t| t.as_str()),
        Some("hi")
    );
}

#[test]
fn chat_response_logprobs_absent_when_upstream_omits_them() {
    let lm = json!({
        "choices": [{
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1}
    });
    let result = ResponseTransformer::convert_to_ollama_chat(&lm, "m", 1, Instant::now());
    assert!(
        result.get("logprobs").is_none(),
        "logprobs must be absent when upstream does not return them"
    );
}

#[test]
fn chat_response_logprobs_absent_when_upstream_returns_null() {
    let lm = json!({
        "choices": [{
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop",
            "logprobs": null
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1}
    });
    let result = ResponseTransformer::convert_to_ollama_chat(&lm, "m", 1, Instant::now());
    assert!(
        result.get("logprobs").is_none(),
        "null upstream logprobs must not produce a logprobs field"
    );
}

// =========================================================================
// GAP B: message.images forwarding
// =========================================================================

#[test]
fn chat_response_message_images_absent_when_upstream_omits_them() {
    // LM Studio is vision-input-only; assistants never generate images.
    // The field must be absent rather than null when upstream does not return it.
    let lm = json!({
        "choices": [{
            "message": {"role": "assistant", "content": "here is info"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3}
    });
    let result = ResponseTransformer::convert_to_ollama_chat(&lm, "m", 1, Instant::now());
    let msg = result.get("message").expect("message must be present");
    assert!(
        msg.get("images").is_none(),
        "images field must be absent when upstream does not return it"
    );
}

#[test]
fn chat_response_message_images_forwarded_when_upstream_provides_them() {
    // Though LM Studio does not currently generate image tokens, the proxy must
    // forward any upstream image data if an upstream ever does return it.
    let lm = json!({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "see attached",
                "images": ["aGVsbG8=", "d29ybGQ="]
            },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2}
    });
    let result = ResponseTransformer::convert_to_ollama_chat(&lm, "m", 1, Instant::now());
    let msg = result.get("message").expect("message must be present");
    let images = msg
        .get("images")
        .and_then(|v| v.as_array())
        .expect("images must be an array when upstream returns them");
    assert_eq!(images.len(), 2);
    assert_eq!(images[0].as_str(), Some("aGVsbG8="));
    assert_eq!(images[1].as_str(), Some("d29ybGQ="));
}

#[test]
fn chat_response_message_images_absent_when_upstream_returns_empty_array() {
    let lm = json!({
        "choices": [{
            "message": {"role": "assistant", "content": "hi", "images": []},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1}
    });
    let result = ResponseTransformer::convert_to_ollama_chat(&lm, "m", 1, Instant::now());
    let msg = result.get("message").expect("message must be present");
    assert!(
        msg.get("images").is_none(),
        "empty images array must not produce images field (absent not null)"
    );
}

// =========================================================================
// GAP C: streaming final-chunk timing — heuristic is wall-clock based
// =========================================================================

#[test]
fn timing_stream_chunks_is_wall_clock_heuristic_not_real_stats() {
    // LM Studio's OpenAI-compat streaming does not expose per-token timing data
    // in SSE chunks. This test documents that from_stream_chunks produces non-zero
    // heuristic timings based on wall-clock duration and chunk count.
    // When LM Studio adds streaming usage support, this test should be updated
    // to assert against real per-token stats from upstream.
    let duration = Duration::from_millis(200);
    let timing = TimingInfo::from_stream_chunks(duration, 20, None);
    assert_eq!(
        timing.total_duration,
        duration.as_nanos() as u64,
        "total_duration must reflect wall-clock duration"
    );
    assert!(
        timing.prompt_eval_duration >= 1,
        "heuristic prompt_eval_duration must be ≥ 1"
    );
    assert!(
        timing.eval_duration >= 1,
        "heuristic eval_duration must be ≥ 1"
    );
    assert_eq!(
        timing.eval_count, 20,
        "chunk count used as token estimate when no real usage available"
    );
}

// =========================================================================
// GAP A — inbound tool-result messages (role:"tool")
// =========================================================================

#[test]
fn normalize_tool_role_maps_tool_name_to_name() {
    let msgs = vec![
        json!({"role": "user", "content": "What is the temperature?"}),
        json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_abc",
                "type": "function",
                "function": {"name": "get_temperature", "arguments": {"city": "New York"}}
            }]
        }),
        json!({"role": "tool", "tool_name": "get_temperature", "content": "22°C"}),
    ];
    let out = normalize_chat_messages(&msgs, None);
    let arr = out.as_array().unwrap();
    // The tool result message is the third one.
    let tool_msg = &arr[2];
    assert_eq!(
        tool_msg.get("role").and_then(|v| v.as_str()),
        Some("tool"),
        "role must remain 'tool'"
    );
    assert_eq!(
        tool_msg.get("name").and_then(|v| v.as_str()),
        Some("get_temperature"),
        "tool_name must be mapped to 'name'"
    );
    assert!(
        tool_msg.get("tool_name").is_none(),
        "tool_name key must be removed"
    );
    assert_eq!(
        tool_msg.get("content").and_then(|v| v.as_str()),
        Some("22°C"),
        "content must be preserved"
    );
}

#[test]
fn normalize_tool_role_synthesizes_tool_call_id_from_prior_assistant() {
    let msgs = vec![
        json!({"role": "user", "content": "temp?"}),
        json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_xyz",
                "type": "function",
                "function": {"name": "get_temperature", "arguments": {"city": "London"}}
            }]
        }),
        json!({"role": "tool", "tool_name": "get_temperature", "content": "15°C"}),
    ];
    let out = normalize_chat_messages(&msgs, None);
    let arr = out.as_array().unwrap();
    let tool_msg = &arr[2];
    assert_eq!(
        tool_msg.get("tool_call_id").and_then(|v| v.as_str()),
        Some("call_xyz"),
        "tool_call_id must be copied from the matching prior assistant tool_calls entry"
    );
}

#[test]
fn normalize_tool_role_omits_tool_call_id_when_no_prior_match() {
    // No assistant message before the tool result — tool_call_id is omitted.
    let msgs = vec![json!({"role": "tool", "tool_name": "unknown_fn", "content": "result"})];
    let out = normalize_chat_messages(&msgs, None);
    let arr = out.as_array().unwrap();
    let tool_msg = &arr[0];
    assert!(
        tool_msg.get("tool_call_id").is_none(),
        "tool_call_id must be absent when no prior assistant tool_calls entry matches"
    );
    assert_eq!(
        tool_msg.get("name").and_then(|v| v.as_str()),
        Some("unknown_fn")
    );
}

#[test]
fn normalize_tool_role_parallel_tool_calls_match_by_name() {
    // Multiple tool results each matching a different tool_call in the same assistant message.
    let msgs = vec![
        json!({"role": "user", "content": "both"}),
        json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "get_temperature", "arguments": {"city": "NY"}}},
                {"id": "call_2", "type": "function",
                 "function": {"name": "get_conditions", "arguments": {"city": "NY"}}}
            ]
        }),
        json!({"role": "tool", "tool_name": "get_temperature", "content": "22°C"}),
        json!({"role": "tool", "tool_name": "get_conditions", "content": "Sunny"}),
    ];
    let out = normalize_chat_messages(&msgs, None);
    let arr = out.as_array().unwrap();
    assert_eq!(
        arr[2].get("tool_call_id").and_then(|v| v.as_str()),
        Some("call_1"),
        "get_temperature result must reference call_1"
    );
    assert_eq!(
        arr[3].get("tool_call_id").and_then(|v| v.as_str()),
        Some("call_2"),
        "get_conditions result must reference call_2"
    );
}

// =========================================================================
// GAP B — inbound assistant tool_calls with object arguments
// =========================================================================

#[test]
fn normalize_assistant_tool_calls_stringifies_object_arguments() {
    // Ollama sends arguments as a JSON object; OpenAI/LM Studio requires a JSON string.
    let msgs = vec![
        json!({"role": "user", "content": "What is the temp?"}),
        json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": "get_temperature",
                    "arguments": {"city": "Tokyo"}
                }
            }]
        }),
    ];
    let out = normalize_chat_messages(&msgs, None);
    let arr = out.as_array().unwrap();
    let assistant = &arr[1];
    let calls = assistant
        .get("tool_calls")
        .and_then(|v| v.as_array())
        .unwrap();
    assert_eq!(calls.len(), 1);
    let args = calls[0].get("function").unwrap().get("arguments").unwrap();
    assert!(
        args.is_string(),
        "arguments must be serialized to a JSON string, got {:?}",
        args
    );
    let parsed: serde_json::Value = serde_json::from_str(args.as_str().unwrap()).unwrap();
    assert_eq!(
        parsed.get("city").and_then(|v| v.as_str()),
        Some("Tokyo"),
        "round-trip through JSON string must preserve the original value"
    );
}

#[test]
fn normalize_assistant_tool_calls_adds_id_and_type_when_absent() {
    let msgs = vec![json!({
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "function": {"name": "fn_no_id", "arguments": {"k": "v"}}
        }]
    })];
    let out = normalize_chat_messages(&msgs, None);
    let arr = out.as_array().unwrap();
    let calls = arr[0].get("tool_calls").and_then(|v| v.as_array()).unwrap();
    let call = &calls[0];
    assert!(
        call.get("id").and_then(|v| v.as_str()).is_some(),
        "id must be synthesized when absent"
    );
    assert_eq!(
        call.get("type").and_then(|v| v.as_str()),
        Some("function"),
        "type must be 'function'"
    );
}

#[test]
fn normalize_assistant_tool_calls_preserves_existing_string_arguments() {
    // If arguments is already a string (well-formed caller), it must not be double-serialized.
    let msgs = vec![json!({
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call_ok",
            "type": "function",
            "function": {"name": "fn", "arguments": "{\"x\":1}"}
        }]
    })];
    let out = normalize_chat_messages(&msgs, None);
    let arr = out.as_array().unwrap();
    let calls = arr[0].get("tool_calls").and_then(|v| v.as_array()).unwrap();
    let args = calls[0].get("function").unwrap().get("arguments").unwrap();
    assert!(args.is_string(), "string arguments must remain a string");
    assert_eq!(args.as_str().unwrap(), "{\"x\":1}");
}

#[test]
fn normalize_non_tool_messages_pass_through_unmodified() {
    let msgs = vec![
        json!({"role": "user", "content": "hello"}),
        json!({"role": "assistant", "content": "hi there"}),
        json!({"role": "system", "content": "be helpful"}),
    ];
    let out = normalize_chat_messages(&msgs, None);
    let arr = out.as_array().unwrap();
    // user and system pass through exactly; assistant without tool_calls passes through.
    assert_eq!(arr[0], json!({"role": "user", "content": "hello"}));
    assert_eq!(arr[1], json!({"role": "assistant", "content": "hi there"}));
    assert_eq!(arr[2], json!({"role": "system", "content": "be helpful"}));
}
