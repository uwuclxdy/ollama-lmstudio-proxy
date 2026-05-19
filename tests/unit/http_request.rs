use super::*;
use serde_json::json;

#[test]
fn forwards_presence_penalty() {
    let options = json!({ "presence_penalty": 0.5 });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(params.get("presence_penalty"), Some(&json!(0.5)));
}

#[test]
fn forwards_frequency_penalty() {
    let options = json!({ "frequency_penalty": 0.3 });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(params.get("frequency_penalty"), Some(&json!(0.3)));
}

#[test]
fn forwards_min_p() {
    let options = json!({ "min_p": 0.05 });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(params.get("min_p"), Some(&json!(0.05)));
}

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
fn num_ctx_maps_to_context_length() {
    let options = json!({ "num_ctx": 8192 });
    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(params.get("context_length"), Some(&json!(8192)));
    assert!(params.get("num_ctx").is_none());
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
