//! Tests for request-side translation from Ollama options to LM Studio params.
//!
//! References:
//!   - api_docs/ollama.md (Modelfile parameters, /api/chat, /api/generate, /api/embed)
//!   - api_docs/lmstudio/1_developer/3_openai-compat/chat-completions.md
//!   - api_docs/lmstudio/1_developer/3_openai-compat/completions.mdx
//!   - api_docs/lmstudio/1_developer/2_rest/endpoints.mdx (/api/v0/* params)
//!   - api_docs/lmstudio/1_developer/3_openai-compat/structured-output.md
//!   - README.md lines 130-160 (option mapping table)

#[path = "../src/http/request.rs"]
#[allow(dead_code)]
mod request;

use std::borrow::Cow;

use request::{
    LMStudioRequestType, TopLevelParams, build_lm_studio_request, collect_unsupported_keys,
    map_ollama_to_lmstudio_params, prepare_request_body,
};
use serde_json::{Value, json};

// --- DIRECT_MAPPINGS: each Ollama option name forwards verbatim ---

#[test]
fn forwards_temperature() {
    // ollama.md §Modelfile parameters: temperature is a direct passthrough sampler param
    let options = json!({ "temperature": 0.7 });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(params.get("temperature"), Some(&json!(0.7)));
}

#[test]
fn forwards_top_p() {
    // ollama.md §Modelfile parameters: top_p is a direct passthrough
    let options = json!({ "top_p": 0.9 });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(params.get("top_p"), Some(&json!(0.9)));
}

#[test]
fn forwards_top_k() {
    // ollama.md §Modelfile parameters: top_k is a direct passthrough
    let options = json!({ "top_k": 40 });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(params.get("top_k"), Some(&json!(40)));
}

#[test]
fn forwards_seed() {
    // ollama.md §Modelfile parameters: seed is a direct passthrough
    let options = json!({ "seed": 42 });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(params.get("seed"), Some(&json!(42)));
}

#[test]
fn forwards_stop_array() {
    // ollama.md §Modelfile parameters: stop is a list of stop sequences
    let options = json!({ "stop": ["\nUser:", "\nAssistant:"] });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(
        params.get("stop"),
        Some(&json!(["\nUser:", "\nAssistant:"]))
    );
}

#[test]
fn forwards_presence_penalty_direct() {
    // chat-completions.md: presence_penalty is an OpenAI-compatible sampler param
    let options = json!({ "presence_penalty": 0.5 });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(params.get("presence_penalty"), Some(&json!(0.5)));
}

#[test]
fn forwards_frequency_penalty_direct() {
    // chat-completions.md: frequency_penalty is an OpenAI-compatible sampler param
    let options = json!({ "frequency_penalty": 0.3 });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(params.get("frequency_penalty"), Some(&json!(0.3)));
}

#[test]
fn forwards_min_p_direct() {
    // ollama.md §Modelfile parameters: min_p is a direct passthrough sampler param
    let options = json!({ "min_p": 0.05 });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(params.get("min_p"), Some(&json!(0.05)));
}

// --- truncate / dimensions (live in DIRECT_MAPPINGS, embeddings-only by spec) ---

#[test]
fn forwards_truncate_when_present() {
    // ollama.md §/api/embed: truncate is an embedding-only top-level param (lifted into options)
    let options = json!({ "truncate": false });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(params.get("truncate"), Some(&json!(false)));
}

#[test]
fn forwards_dimensions_when_present() {
    // ollama.md §/api/embed: dimensions is an embedding-only top-level param (lifted into options)
    let options = json!({ "dimensions": 768 });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(params.get("dimensions"), Some(&json!(768)));
}

// --- logit_bias / system ---

#[test]
fn forwards_logit_bias_when_present() {
    // chat-completions.md: logit_bias is an OpenAI-compatible map of token-id to bias
    let options = json!({ "logit_bias": { "50256": -100 } });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(params.get("logit_bias"), Some(&json!({ "50256": -100 })));
}

#[test]
fn forwards_system_when_present_inside_options() {
    // ollama.md /api/generate: system is a top-level field; proxy also accepts inside options
    let options = json!({ "system": "You are a helpful assistant." });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(
        params.get("system"),
        Some(&json!("You are a helpful assistant."))
    );
}

// --- max_tokens vs num_predict precedence ---

#[test]
fn max_tokens_wins_over_num_predict() {
    // ollama.md §Modelfile parameters: num_predict caps generation; OpenAI alias is max_tokens.
    // Both map to LM Studio's max_tokens; explicit max_tokens takes precedence.
    let options = json!({ "max_tokens": 100, "num_predict": 500 });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(params.get("max_tokens"), Some(&json!(100)));
}

#[test]
fn num_predict_maps_to_max_tokens_when_max_tokens_absent() {
    // ollama.md §Modelfile parameters: num_predict is the Ollama-native key
    let options = json!({ "num_predict": 256 });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(params.get("max_tokens"), Some(&json!(256)));
    assert!(
        params.get("num_predict").is_none(),
        "num_predict must be renamed to max_tokens, got {:?}",
        params
    );
}

// --- num_ctx → context_length ---

#[test]
fn num_ctx_renamed_to_context_length() {
    // endpoints.mdx /api/v0/* uses `context_length`; Ollama uses `num_ctx`
    let options = json!({ "num_ctx": 4096 });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(params.get("context_length"), Some(&json!(4096)));
    assert!(
        params.get("num_ctx").is_none(),
        "num_ctx must be renamed; got {:?}",
        params
    );
}

// --- repeat_penalty mapping ---

#[test]
fn repeat_penalty_alone_forwarded_as_repeat_penalty() {
    // README.md option mapping: repeat_penalty alone -> repeat_penalty
    let options = json!({ "repeat_penalty": 1.1 });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(params.get("repeat_penalty"), Some(&json!(1.1)));
    assert!(params.get("frequency_penalty").is_none());
}

#[test]
fn repeat_penalty_routes_to_frequency_when_presence_set() {
    // README.md option mapping: when presence_penalty is already set,
    // repeat_penalty fills the frequency_penalty slot
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
    // README.md option mapping: when both OpenAI penalties are set,
    // repeat_penalty has no slot and is dropped (current code behavior)
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

// --- format mapping ---

#[test]
fn format_string_json_becomes_json_object() {
    // structured-output.md: response_format = {"type":"json_object"} for free-form JSON
    let options = json!({ "format": "json" });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(
        params.get("response_format"),
        Some(&json!({ "type": "json_object" }))
    );
}

#[test]
fn format_string_text_becomes_text() {
    // structured-output.md: "text" is the unstructured default; surfaced as {"type":"text"}
    let options = json!({ "format": "text" });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(
        params.get("response_format"),
        Some(&json!({ "type": "text" }))
    );
}

#[test]
fn format_object_becomes_json_schema() {
    // structured-output.md: JSON Schema object goes under response_format.json_schema.schema
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
    // chat-completions.md: response_format is optional; omit when caller did not request it
    let options = json!({ "temperature": 0.7 });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert!(
        params.get("response_format").is_none(),
        "response_format must be absent when format not provided: {:?}",
        params
    );
}

// --- structured_format precedence over options.format ---

#[test]
fn structured_format_arg_overrides_options_format() {
    // map_format_params: explicit structured_format argument wins over options.format
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

// --- unsupported keys ---

#[test]
fn unsupported_keys_present_in_collect_absent_from_mapped_params() {
    // ollama.md §Modelfile parameters lists these; LM Studio has no equivalent (README.md)
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

// --- build_lm_studio_request shape ---

#[test]
fn build_chat_request_contains_model_messages_stream() {
    // chat-completions.md: required fields are model + messages; stream is the streaming toggle
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
    // chat-completions.md: tools array forwarded for function/tool-calling support
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
    // chat-completions.md: empty tools array carries no callable functions; omit the field
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
    // completions.mdx: required fields are model + prompt; stream toggles streaming
    let request = build_lm_studio_request(
        "my-model",
        LMStudioRequestType::Completion {
            prompt: Cow::Borrowed("once upon a time"),
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
    // endpoints.mdx /api/v0/embeddings: model + input are the only required fields; no stream
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

// --- apply_top_level_params: None case (think branch covered by inline tests) ---

#[test]
fn top_level_none_inserts_no_reasoning_logprobs_keys() {
    // When top_level is None, build must not invent reasoning/logprobs/top_logprobs fields
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
    // TopLevelParams { think: None, logprobs: None, top_logprobs: None } must add zero keys
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

// --- prepare_request_body ---

#[test]
fn prepare_request_body_with_json_returns_serialized_bytes_marked_json() {
    // Some(value) => serialize to bytes and tag is_json=true (downstream sets Content-Type)
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
    // None + non-empty original => forward verbatim and tag is_json=false
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
    // None + empty original => no body to send at all
    let prepared = prepare_request_body(None, b"").expect("prepare must not fail");
    assert!(
        prepared.bytes.is_none(),
        "bytes must be None for empty input"
    );
    assert!(!prepared.is_json);
}
