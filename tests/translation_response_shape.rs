//! Tests for LM Studio → Ollama non-streaming response translation.
//!
//! Reference docs:
//!   api_docs/ollama.md
//!     - /api/chat response: `model`, `created_at`, `message.{role,content,thinking,tool_calls}`,
//!       `done`, `done_reason`, timings (lines ~210-260)
//!     - /api/generate response: `model`, `created_at`, `response`, `done`, `done_reason`,
//!       timings, deprecated `context` (lines ~90-130)
//!     - /api/embed response: `model`, `embeddings`, `total_duration`, `load_duration`,
//!       `prompt_eval_count` only — NO `eval_count`/`eval_duration` (lines 1728-1740)
//!     - /api/embeddings legacy: `{"embedding": [...]}` (lines 1844-1850)
//!   api_docs/lmstudio/1_developer/3_openai-compat/chat-completions.md
//!     - `choices[].message`, `finish_reason`, `usage`, `tool_calls`
//!   api_docs/lmstudio/1_developer/2_rest/endpoints.mdx lines 145-185
//!     - /api/v0/chat/completions with `stats`
//!   OpenAI tool_calls: `function.arguments` is a JSON-encoded STRING; Ollama expects OBJECT

#[path = "../src/constants.rs"]
#[allow(dead_code)]
mod constants;

#[path = "../src/handlers/transform.rs"]
#[allow(dead_code)]
mod transform;

use serde_json::json;
use std::time::{Duration, Instant};
use transform::{
    ResponseTransformer, TimingInfo, estimate_token_count, extract_finish_reason,
    normalize_chat_messages,
};

// ---------- convert_to_ollama_chat ----------

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
    let result =
        ResponseTransformer::convert_to_ollama_chat(&lm, "llama3", 1, Instant::now(), false);

    assert_eq!(result.get("model").and_then(|v| v.as_str()), Some("llama3"));

    // RFC3339 timestamp — chrono::Utc::now().to_rfc3339() produces e.g. 2024-…T…Z
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

/// Per LM Studio chat-completions doc, `usage.{prompt_tokens,completion_tokens}` carries
/// authoritative counts. The translator must prefer those over heuristic estimates when
/// `use_native_stats=false` (legacy path).
#[test]
fn chat_response_uses_usage_counts_when_present() {
    let lm = json!({
        "choices": [{
            "message": {"role": "assistant", "content": "x"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 42, "completion_tokens": 17}
    });
    let result = ResponseTransformer::convert_to_ollama_chat(&lm, "m", 1, Instant::now(), false);

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

/// When `usage` is absent the translator must fall back to estimates (must still be ≥ 1
/// so clients don't divide-by-zero when computing tokens/sec).
#[test]
fn chat_response_falls_back_to_estimates_without_usage() {
    let lm = json!({
        "choices": [{
            "message": {"role": "assistant", "content": "abcd"},
            "finish_reason": "stop"
        }]
    });
    let result = ResponseTransformer::convert_to_ollama_chat(&lm, "m", 2, Instant::now(), false);

    let prompt = result
        .get("prompt_eval_count")
        .and_then(|v| v.as_u64())
        .unwrap();
    let eval = result.get("eval_count").and_then(|v| v.as_u64()).unwrap();
    assert!(prompt >= 1, "prompt_eval_count must be ≥ 1, got {prompt}");
    assert!(eval >= 1, "eval_count must be ≥ 1, got {eval}");
}

/// LM Studio `finish_reason` values pass through to Ollama `done_reason` unchanged.
/// Reference: ollama.md /api/chat (`done_reason`) + LM Studio chat-completions
/// (`finish_reason` ∈ {"stop","length","tool_calls",…}).
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
        let result =
            ResponseTransformer::convert_to_ollama_chat(&lm, "m", 1, Instant::now(), false);
        assert_eq!(
            result.get("done_reason").and_then(|v| v.as_str()),
            Some(reason),
            "finish_reason={reason} must propagate as done_reason"
        );
    }
}

/// Per ollama.md /api/chat: `message.content` is a string (not null). LM Studio (OpenAI
/// shape) emits `content: null` when the model returns only `tool_calls`. The proxy must
/// translate that to an empty string while preserving the `tool_calls` array on the
/// message — Ollama clients deserializing into `String` would otherwise crash.
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
    let result = ResponseTransformer::convert_to_ollama_chat(&lm, "m", 1, Instant::now(), false);
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

/// Per ollama.md /api/chat: `tool_calls` is an array; if the upstream returns several,
/// all must be forwarded.
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
    let result = ResponseTransformer::convert_to_ollama_chat(&lm, "m", 1, Instant::now(), false);
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

/// Per ollama.md /api/chat response: `message.role` is always `"assistant"`. Even if
/// upstream returns a different role (a misbehaving LM Studio build, say), the proxy
/// must normalize it.
#[test]
fn chat_response_role_is_always_assistant() {
    let lm = json!({
        "choices": [{
            "message": {"role": "user", "content": "oddly tagged"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1}
    });
    let result = ResponseTransformer::convert_to_ollama_chat(&lm, "m", 1, Instant::now(), false);
    assert_eq!(
        result
            .get("message")
            .unwrap()
            .get("role")
            .and_then(|v| v.as_str()),
        Some("assistant")
    );
}

// ---------- convert_to_ollama_generate ----------

/// Per ollama.md /api/generate response: top-level `model`, `created_at`, `response`,
/// `done`, `done_reason`, and the six timing fields must all be present.
#[test]
fn generate_response_contains_full_ollama_shape() {
    let lm = json!({
        "choices": [{"text": "hi", "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 1}
    });
    let result = ResponseTransformer::convert_to_ollama_generate(
        &lm,
        "model-x",
        "prompt",
        Instant::now(),
        false,
    );
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

/// Per LM Studio /v1/completions: text lives at `choices[0].text`. The proxy must read
/// it for /api/generate translation.
#[test]
fn generate_response_extracts_completion_text() {
    let lm = json!({
        "choices": [{"text": "completion-style output", "finish_reason": "stop"}]
    });
    let result =
        ResponseTransformer::convert_to_ollama_generate(&lm, "m", "p", Instant::now(), false);
    assert_eq!(
        result.get("response").and_then(|v| v.as_str()),
        Some("completion-style output")
    );
}

/// Some chat-shaped LM Studio responses are routed through /api/generate (LM Studio's
/// own native /api/v0/chat/completions returns `choices[0].message.content`). The
/// translator falls back to `choices[0].message.content` when `text` is absent so the
/// /api/generate adapter keeps working with chat-shaped upstreams.
#[test]
fn generate_response_falls_back_to_message_content() {
    let lm = json!({
        "choices": [{
            "message": {"role": "assistant", "content": "chat-shaped output"},
            "finish_reason": "stop"
        }]
    });
    let result =
        ResponseTransformer::convert_to_ollama_generate(&lm, "m", "p", Instant::now(), false);
    assert_eq!(
        result.get("response").and_then(|v| v.as_str()),
        Some("chat-shaped output")
    );
}

/// Per ollama.md /api/generate: `thinking` is optional. The translator must omit it
/// when the upstream returns neither `reasoning` nor `thinking`, or returns an empty
/// string.
#[test]
fn generate_response_omits_thinking_when_empty_or_missing() {
    // missing
    let lm = json!({"choices": [{"text": "x", "finish_reason": "stop"}]});
    let r = ResponseTransformer::convert_to_ollama_generate(&lm, "m", "p", Instant::now(), false);
    assert!(r.get("thinking").is_none(), "no field if absent");

    // empty string
    let lm = json!({"choices": [{"text": "x", "reasoning": "", "finish_reason": "stop"}]});
    let r = ResponseTransformer::convert_to_ollama_generate(&lm, "m", "p", Instant::now(), false);
    assert!(
        r.get("thinking").is_none(),
        "empty reasoning must not emit thinking field"
    );

    // empty `thinking` field on the choice
    let lm = json!({"choices": [{"text": "x", "thinking": "", "finish_reason": "stop"}]});
    let r = ResponseTransformer::convert_to_ollama_generate(&lm, "m", "p", Instant::now(), false);
    assert!(r.get("thinking").is_none());
}

/// Per ollama.md /api/generate: `thinking` appears at the TOP level (not inside any
/// message). When the upstream `choices[0].thinking` is non-empty, the proxy must
/// surface it.
#[test]
fn generate_response_emits_thinking_from_choice_thinking_field() {
    let lm = json!({
        "choices": [{"text": "answer", "thinking": "let me think",
                     "finish_reason": "stop"}]
    });
    let r = ResponseTransformer::convert_to_ollama_generate(&lm, "m", "p", Instant::now(), false);
    assert_eq!(
        r.get("thinking").and_then(|v| v.as_str()),
        Some("let me think")
    );
}

// ---------- convert_to_ollama_embeddings ----------

/// Per ollama.md /api/embed (lines 1728-1740) the response shape is:
///   {"model", "embeddings": [[..], [..]], "total_duration", "load_duration",
///    "prompt_eval_count"}
/// Note: `eval_count` and `eval_duration` are NOT in the spec.
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
        ResponseTransformer::convert_to_ollama_embeddings(&lm, "all-minilm", Instant::now(), false);
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
    // Ollama /api/embed does NOT define eval_count/eval_duration (it is not an
    // autoregressive eval). Verify the proxy does not invent them.
    assert!(
        result.get("eval_count").is_none(),
        "eval_count must NOT be present on /api/embed (got {result})"
    );
    assert!(
        result.get("eval_duration").is_none(),
        "eval_duration must NOT be present on /api/embed (got {result})"
    );
}

/// Robustness: an empty `data` array (no embeddings produced) must yield an empty
/// `embeddings` array, not a panic. The Ollama spec doesn't forbid empty results.
#[test]
fn embeddings_response_empty_data_yields_empty_embeddings() {
    let lm = json!({"data": [], "usage": {"prompt_tokens": 0}});
    let result = ResponseTransformer::convert_to_ollama_embeddings(&lm, "m", Instant::now(), false);
    let embeds = result
        .get("embeddings")
        .and_then(|v| v.as_array())
        .expect("embeddings array must exist");
    assert!(embeds.is_empty());
}

// ---------- normalize_chat_messages ----------

/// No `system_prompt` ⇒ the message array passes through unchanged.
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

/// `system_prompt` provided and no existing system message ⇒ a new system message is
/// prepended (preserving relative order of other messages).
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

/// `system_prompt` provided AND the messages already include a system message ⇒ the
/// original list is returned untouched (no duplicated system role).
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

/// Role match is case-insensitive — "System" still counts as an existing system
/// message. (Some clients capitalize roles.)
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

// ---------- estimate_token_count ----------

/// Empty string ⇒ 0 tokens (boundary).
#[test]
fn estimate_token_count_empty_is_zero() {
    assert_eq!(estimate_token_count(""), 0);
}

/// Non-empty: ceil(len * TOKEN_TO_CHAR_RATIO) where TOKEN_TO_CHAR_RATIO = 0.25.
/// "abcd" (4 bytes) → ceil(4 * 0.25) = 1; "abcde" (5 bytes) → ceil(1.25) = 2;
/// "a" → ceil(0.25) = 1.
#[test]
fn estimate_token_count_uses_ceil_quarter_length() {
    assert_eq!(estimate_token_count("a"), 1);
    assert_eq!(estimate_token_count("abcd"), 1);
    assert_eq!(estimate_token_count("abcde"), 2);
    assert_eq!(estimate_token_count("abcdefgh"), 2);
    assert_eq!(estimate_token_count("abcdefghi"), 3);
}

// ---------- extract_finish_reason ----------

/// Missing `choices` ⇒ None.
#[test]
fn extract_finish_reason_missing_choices_is_none() {
    let v = json!({"usage": {"prompt_tokens": 1}});
    assert!(extract_finish_reason(&v).is_none());
}

/// Empty `choices` array ⇒ None (no first element to inspect).
#[test]
fn extract_finish_reason_empty_choices_is_none() {
    let v = json!({"choices": []});
    assert!(extract_finish_reason(&v).is_none());
}

/// Present `finish_reason` ⇒ Some(&str).
#[test]
fn extract_finish_reason_present() {
    let v = json!({"choices": [{"finish_reason": "length"}]});
    assert_eq!(extract_finish_reason(&v), Some("length"));
}

// ---------- TimingInfo::from_legacy_estimation ----------

/// All six timing fields must be ≥ 1 even for a trivially short elapsed time, so
/// downstream `tokens / duration` calculations on the Ollama client side cannot
/// divide by zero. Reference: ollama.md timing fields are all u64 nanoseconds.
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

/// `usage`-derived counts (passed in as Some(_)) must override the estimates.
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

/// When the upstream omits `usage`, estimates fill in.
#[test]
fn timing_legacy_estimation_falls_back_to_estimates() {
    let timing = TimingInfo::from_legacy_estimation(Instant::now(), 30, 12, None, None);
    assert_eq!(timing.prompt_eval_count, 30);
    assert_eq!(timing.eval_count, 12);
}

// ---------- TimingInfo::from_stream_chunks ----------

/// `from_stream_chunks` derives counts from the chunk-count estimate when no usage
/// is provided, and respects actual usage when supplied. All six fields must be ≥ 1.
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
