use super::*;
use serde_json::{Value, json};

#[test]
fn collects_unsupported_options() {
    let options = json!({ "template": "{{.Prompt}}", "mirostat": 1, "temperature": 0.7 });
    let unsupported = collect_unsupported_keys(&options);
    assert!(
        unsupported.contains(&"template"),
        "expected template in {:?}",
        unsupported
    );
    assert!(
        unsupported.contains(&"mirostat"),
        "expected mirostat in {:?}",
        unsupported
    );
    assert!(
        !unsupported.contains(&"temperature"),
        "temperature should not appear in {:?}",
        unsupported
    );
}

#[test]
fn num_ctx_is_not_treated_as_unsupported() {
    let options = json!({ "num_ctx": 4096 });
    let unsupported = collect_unsupported_keys(&options);
    assert!(
        !unsupported.contains(&"num_ctx"),
        "num_ctx must not appear in unsupported: {:?}",
        unsupported
    );
}

#[test]
fn log_unsupported_options_does_not_panic() {
    let options = json!({ "num_ctx": 4096, "mirostat": 1 });
    log_unsupported_options(&options);

    let empty = json!({ "temperature": 0.7 });
    log_unsupported_options(&empty);
}

#[test]
fn top_level_params_think_true_emits_reasoning_on() {
    let think_val = json!(true);
    let top = TopLevelParams {
        think: Some(&think_val),
        logprobs: None,
        top_logprobs: None,
    };
    let request = build_lm_studio_request(
        "mymodel",
        LMStudioRequestType::Completion {
            prompt: std::borrow::Cow::Borrowed("hello"),
            stream: false,
        },
        None,
        None,
        None,
        Some(&top),
    );
    assert_eq!(request.get("reasoning"), Some(&json!("on")));
}

#[test]
fn top_level_params_think_false_emits_reasoning_off() {
    let think_val = json!(false);
    let top = TopLevelParams {
        think: Some(&think_val),
        logprobs: None,
        top_logprobs: None,
    };
    let request = build_lm_studio_request(
        "mymodel",
        LMStudioRequestType::Completion {
            prompt: std::borrow::Cow::Borrowed("hello"),
            stream: false,
        },
        None,
        None,
        None,
        Some(&top),
    );
    assert_eq!(request.get("reasoning"), Some(&json!("off")));
}

#[test]
fn top_level_params_think_string_passes_through() {
    let think_val = json!("high");
    let top = TopLevelParams {
        think: Some(&think_val),
        logprobs: None,
        top_logprobs: None,
    };
    let request = build_lm_studio_request(
        "mymodel",
        LMStudioRequestType::Completion {
            prompt: std::borrow::Cow::Borrowed("hello"),
            stream: false,
        },
        None,
        None,
        None,
        Some(&top),
    );
    assert_eq!(request.get("reasoning"), Some(&json!("high")));
}

#[test]
fn top_level_params_absent_think_emits_no_reasoning() {
    let top = TopLevelParams {
        think: None,
        logprobs: None,
        top_logprobs: None,
    };
    let request = build_lm_studio_request(
        "mymodel",
        LMStudioRequestType::Completion {
            prompt: std::borrow::Cow::Borrowed("hello"),
            stream: false,
        },
        None,
        None,
        None,
        Some(&top),
    );
    assert!(request.get("reasoning").is_none());
}

#[test]
fn top_level_params_logprobs_forwarded() {
    let lp = json!(true);
    let top = TopLevelParams {
        think: None,
        logprobs: Some(&lp),
        top_logprobs: None,
    };
    let request = build_lm_studio_request(
        "mymodel",
        LMStudioRequestType::Completion {
            prompt: std::borrow::Cow::Borrowed("hello"),
            stream: false,
        },
        None,
        None,
        None,
        Some(&top),
    );
    assert_eq!(request.get("logprobs"), Some(&json!(true)));
}

#[test]
fn top_level_params_work_on_chat_type_too() {
    let think_val = json!("medium");
    let top = TopLevelParams {
        think: Some(&think_val),
        logprobs: None,
        top_logprobs: None,
    };
    let messages = json!([{"role": "user", "content": "hi"}]);
    let request = build_lm_studio_request(
        "mymodel",
        LMStudioRequestType::Chat {
            messages: &messages,
            stream: false,
        },
        None,
        None,
        None,
        Some(&top),
    );
    assert_eq!(request.get("reasoning"), Some(&json!("medium")));
}

// =========================================================================
// Tests from translation_request_mapping
// =========================================================================

#[test]
fn forwards_temperature() {
    let options = json!({ "temperature": 0.7 });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(params.get("temperature"), Some(&json!(0.7)));
}

#[test]
fn forwards_top_p() {
    let options = json!({ "top_p": 0.9 });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(params.get("top_p"), Some(&json!(0.9)));
}

#[test]
fn forwards_top_k() {
    let options = json!({ "top_k": 40 });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(params.get("top_k"), Some(&json!(40)));
}

#[test]
fn forwards_seed() {
    let options = json!({ "seed": 42 });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(params.get("seed"), Some(&json!(42)));
}

#[test]
fn forwards_stop_array() {
    let options = json!({ "stop": ["\nUser:", "\nAssistant:"] });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(
        params.get("stop"),
        Some(&json!(["\nUser:", "\nAssistant:"]))
    );
}

#[test]
fn forwards_truncate_when_present() {
    let options = json!({ "truncate": false });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(params.get("truncate"), Some(&json!(false)));
}

#[test]
fn forwards_dimensions_when_present() {
    let options = json!({ "dimensions": 768 });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(params.get("dimensions"), Some(&json!(768)));
}

#[test]
fn forwards_logit_bias_when_present() {
    let options = json!({ "logit_bias": { "50256": -100 } });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(params.get("logit_bias"), Some(&json!({ "50256": -100 })));
}

#[test]
fn forwards_system_when_present_inside_options() {
    let options = json!({ "system": "You are a helpful assistant." });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(
        params.get("system"),
        Some(&json!("You are a helpful assistant."))
    );
}

#[test]
fn max_tokens_wins_over_num_predict() {
    let options = json!({ "max_tokens": 100, "num_predict": 500 });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(params.get("max_tokens"), Some(&json!(100)));
}

#[test]
fn num_predict_maps_to_max_tokens_when_max_tokens_absent() {
    let options = json!({ "num_predict": 256 });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(params.get("max_tokens"), Some(&json!(256)));
    assert!(
        params.get("num_predict").is_none(),
        "num_predict must be renamed to max_tokens, got {:?}",
        params
    );
}

#[test]
fn repeat_penalty_alone_forwarded_as_repeat_penalty() {
    let options = json!({ "repeat_penalty": 1.1 });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(params.get("repeat_penalty"), Some(&json!(1.1)));
    assert!(params.get("frequency_penalty").is_none());
}

#[test]
fn repeat_penalty_routes_to_frequency_when_presence_set() {
    let options = json!({ "repeat_penalty": 1.1, "presence_penalty": 0.5 });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(params.get("presence_penalty"), Some(&json!(0.5)));
    assert_eq!(params.get("frequency_penalty"), Some(&json!(1.1)));
    assert!(
        params.get("repeat_penalty").is_none(),
        "repeat_penalty must be rerouted, not forwarded: {:?}",
        params
    );
}

#[test]
fn repeat_penalty_dropped_when_both_penalties_set() {
    let options = json!({
        "repeat_penalty": 1.1,
        "presence_penalty": 0.5,
        "frequency_penalty": 0.3,
    });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(params.get("presence_penalty"), Some(&json!(0.5)));
    assert_eq!(params.get("frequency_penalty"), Some(&json!(0.3)));
    assert!(
        params.get("repeat_penalty").is_none(),
        "repeat_penalty must be dropped when both slots are taken: {:?}",
        params
    );
}

#[test]
fn format_string_json_becomes_json_object() {
    let options = json!({ "format": "json" });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(
        params.get("response_format"),
        Some(&json!({ "type": "json_object" }))
    );
}

#[test]
fn format_string_text_becomes_text() {
    let options = json!({ "format": "text" });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(
        params.get("response_format"),
        Some(&json!({ "type": "text" }))
    );
}

#[test]
fn format_object_becomes_json_schema() {
    let schema = json!({
        "type": "object",
        "properties": { "name": { "type": "string" } },
        "required": ["name"]
    });
    let options = json!({ "format": schema.clone() });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    let rf = params
        .get("response_format")
        .expect("response_format must be set");
    assert_eq!(rf.get("type"), Some(&json!("json_schema")));
    let js = rf
        .get("json_schema")
        .expect("json_schema sub-object must be set");
    assert_eq!(js.get("strict"), Some(&json!(true)));
    assert_eq!(js.get("schema"), Some(&schema));
}

#[test]
fn format_absent_omits_response_format() {
    let options = json!({ "temperature": 0.7 });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert!(
        params.get("response_format").is_none(),
        "response_format must be absent when format not provided: {:?}",
        params
    );
}

#[test]
fn structured_format_arg_overrides_options_format() {
    let options = json!({ "format": "text" });
    let structured = json!("json");
    let params = map_ollama_to_lmstudio_params(Some(&options), Some(&structured));
    assert_eq!(
        params.get("response_format"),
        Some(&json!({ "type": "json_object" })),
        "structured_format must take precedence; got {:?}",
        params.get("response_format")
    );
}

#[test]
fn unsupported_keys_present_in_collect_absent_from_mapped_params() {
    let options = json!({
        "template": "{{.Prompt}}",
        "repeat_last_n": 64,
        "mirostat": 1,
        "mirostat_tau": 5.0,
        "mirostat_eta": 0.1,
        "tfs_z": 1.0,
        "typical_p": 1.0,
        "num_keep": 5,
        "num_batch": 512,
        "num_gpu": 1,
        "num_thread": 8,
        "numa": false,
        "use_mmap": true,
        "use_mlock": false,
        "vocab_only": false,
        "penalize_newline": true,
    });
    let unsupported = collect_unsupported_keys(&options);
    for key in [
        "template",
        "repeat_last_n",
        "mirostat",
        "mirostat_tau",
        "mirostat_eta",
        "tfs_z",
        "typical_p",
        "num_keep",
        "num_batch",
        "num_gpu",
        "num_thread",
        "numa",
        "use_mmap",
        "use_mlock",
        "vocab_only",
        "penalize_newline",
    ] {
        assert!(
            unsupported.contains(&key),
            "expected {} in unsupported list: {:?}",
            key,
            unsupported
        );
    }

    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    for key in [
        "template",
        "repeat_last_n",
        "mirostat",
        "mirostat_tau",
        "mirostat_eta",
        "tfs_z",
        "typical_p",
        "num_keep",
        "num_batch",
        "num_gpu",
        "num_thread",
        "numa",
        "use_mmap",
        "use_mlock",
        "vocab_only",
        "penalize_newline",
    ] {
        assert!(
            params.get(key).is_none(),
            "unsupported key {} must not appear in mapped params: {:?}",
            key,
            params
        );
    }
}

#[test]
fn build_chat_request_contains_model_messages_stream() {
    let messages = json!([{ "role": "user", "content": "hi" }]);
    let request = build_lm_studio_request(
        "my-model",
        LMStudioRequestType::Chat {
            messages: &messages,
            stream: true,
        },
        None,
        None,
        None,
        None,
    );
    assert_eq!(request.get("model"), Some(&json!("my-model")));
    assert_eq!(request.get("messages"), Some(&messages));
    assert_eq!(request.get("stream"), Some(&json!(true)));
}

#[test]
fn build_chat_request_includes_tools_when_array_non_empty() {
    let messages = json!([{ "role": "user", "content": "hi" }]);
    let tools = json!([{
        "type": "function",
        "function": { "name": "get_weather", "parameters": {} }
    }]);
    let request = build_lm_studio_request(
        "my-model",
        LMStudioRequestType::Chat {
            messages: &messages,
            stream: false,
        },
        None,
        Some(&tools),
        None,
        None,
    );
    assert_eq!(request.get("tools"), Some(&tools));
}

#[test]
fn build_chat_request_omits_tools_when_array_empty() {
    let messages = json!([{ "role": "user", "content": "hi" }]);
    let tools = json!([]);
    let request = build_lm_studio_request(
        "my-model",
        LMStudioRequestType::Chat {
            messages: &messages,
            stream: false,
        },
        None,
        Some(&tools),
        None,
        None,
    );
    assert!(
        request.get("tools").is_none(),
        "empty tools array must be omitted, got {}",
        request
    );
}

#[test]
fn build_completion_request_contains_model_prompt_stream() {
    let request = build_lm_studio_request(
        "my-model",
        LMStudioRequestType::Completion {
            prompt: std::borrow::Cow::Borrowed("once upon a time"),
            stream: false,
        },
        None,
        None,
        None,
        None,
    );
    assert_eq!(request.get("model"), Some(&json!("my-model")));
    assert_eq!(request.get("prompt"), Some(&json!("once upon a time")));
    assert_eq!(request.get("stream"), Some(&json!(false)));
}

#[test]
fn build_embeddings_request_contains_model_and_input() {
    let input = json!("hello world");
    let request = build_lm_studio_request(
        "my-embed-model",
        LMStudioRequestType::Embeddings { input: &input },
        None,
        None,
        None,
        None,
    );
    assert_eq!(request.get("model"), Some(&json!("my-embed-model")));
    assert_eq!(request.get("input"), Some(&input));
    assert!(
        request.get("stream").is_none(),
        "embeddings must not carry a stream flag, got {}",
        request
    );
}

#[test]
fn top_level_none_inserts_no_reasoning_logprobs_keys() {
    let messages = json!([{ "role": "user", "content": "hi" }]);
    let request = build_lm_studio_request(
        "my-model",
        LMStudioRequestType::Chat {
            messages: &messages,
            stream: false,
        },
        None,
        None,
        None,
        None,
    );
    assert!(request.get("reasoning").is_none());
    assert!(request.get("logprobs").is_none());
    assert!(request.get("top_logprobs").is_none());
}

#[test]
fn top_level_all_none_fields_inserts_nothing() {
    let top = TopLevelParams {
        think: None,
        logprobs: None,
        top_logprobs: None,
    };
    let messages = json!([{ "role": "user", "content": "hi" }]);
    let request = build_lm_studio_request(
        "my-model",
        LMStudioRequestType::Chat {
            messages: &messages,
            stream: false,
        },
        None,
        None,
        None,
        Some(&top),
    );
    assert!(request.get("reasoning").is_none());
    assert!(request.get("logprobs").is_none());
    assert!(request.get("top_logprobs").is_none());
}

#[test]
fn prepare_request_body_with_json_returns_serialized_bytes_marked_json() {
    let value = json!({ "model": "x", "messages": [] });
    let prepared = prepare_request_body(Some(value.clone()), b"")
        .expect("prepare must not fail on valid json");
    assert!(prepared.is_json, "is_json must be true for json body");
    let bytes = prepared.bytes.expect("bytes must be present");
    let round: Value = serde_json::from_slice(&bytes).expect("bytes must be valid json");
    assert_eq!(round, value);
}

#[test]
fn prepare_request_body_with_none_and_original_bytes_forwards_raw() {
    let original = b"raw-form-data";
    let prepared = prepare_request_body(None, original).expect("prepare must not fail");
    assert!(
        !prepared.is_json,
        "is_json must be false for raw passthrough"
    );
    assert_eq!(prepared.bytes.as_deref(), Some(&original[..]));
}

#[test]
fn prepare_request_body_with_none_and_empty_bytes_returns_none() {
    let prepared = prepare_request_body(None, b"").expect("prepare must not fail");
    assert!(
        prepared.bytes.is_none(),
        "bytes must be None for empty input"
    );
    assert!(!prepared.is_json);
}
