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
    let result = ResponseTransformer::convert_to_ollama_chat(&lm, "m", 2, Instant::now());
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
fn generate_response_does_not_emit_empty_context_array() {
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
    assert!(
        result.get("context").is_none(),
        "context must be omitted (proxy has no token IDs to fabricate); got {}",
        result
    );
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
    for reason in ["stop", "length", "tool_calls"] {
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
    for key in [
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "prompt_eval_duration",
    ] {
        assert!(result.get(key).is_some(), "{key} must be present per spec");
    }
    assert!(
        result.get("eval_count").is_none(),
        "eval_count must NOT be present on /api/embed (got {result})"
    );
    assert!(
        result.get("eval_duration").is_none(),
        "eval_duration must NOT be present on /api/embed (got {result})"
    );
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
