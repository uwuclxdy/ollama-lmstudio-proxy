use std::time::Instant;

use serde_json::{Value, json};

use crate::lmstudio::native_chat::{
    NativeChatRequestParams, build_native_chat_request, convert_native_to_ollama_chat,
    native_done_reason, native_tool_call_to_openai,
};

fn build(messages: &Value, options: Option<&Value>, think: Option<&Value>) -> Value {
    build_native_chat_request(NativeChatRequestParams {
        model_lm_studio_id: "qwen/qwen3-vl-4b",
        messages,
        system_prompt: Some("be terse"),
        ollama_options: options,
        think,
        stream: false,
        integrations: None,
    })
}

#[test]
fn request_builds_text_input_and_system_prompt() {
    let messages = json!([
        { "role": "user", "content": "hello" },
        { "role": "assistant", "content": "hi" },
    ]);
    let body = build(&messages, None, None);

    assert_eq!(body["model"], json!("qwen/qwen3-vl-4b"));
    assert_eq!(body["system_prompt"], json!("be terse"));
    assert_eq!(body["stream"], json!(false));

    let input = body["input"].as_array().expect("input array");
    assert_eq!(input.len(), 2);
    assert_eq!(
        input[0],
        json!({ "type": "message", "role": "user", "content": "hello" })
    );
    assert_eq!(
        input[1],
        json!({ "type": "message", "role": "assistant", "content": "hi" })
    );
}

#[test]
fn request_emits_image_input_entries() {
    let messages = json!([
        {
            "role": "user",
            "content": "Describe this image in two sentences",
            "images": ["iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9"],
        }
    ]);
    let body = build(&messages, None, None);
    let input = body["input"].as_array().expect("input array");

    assert_eq!(input.len(), 2);
    assert_eq!(input[0]["type"], json!("message"));
    assert_eq!(input[1]["type"], json!("image"));
    // PNG magic prefix sniffed into the data URL mime.
    assert_eq!(
        input[1]["data_url"],
        json!("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9")
    );
}

#[test]
fn request_passes_through_full_data_url_unchanged() {
    let messages = json!([
        {
            "role": "user",
            "content": "x",
            "images": ["data:image/jpeg;base64,/9j/AAAA"],
        }
    ]);
    let body = build(&messages, None, None);
    let input = body["input"].as_array().expect("input array");
    assert_eq!(
        input[1]["data_url"],
        json!("data:image/jpeg;base64,/9j/AAAA")
    );
}

#[test]
fn request_maps_sampling_and_drops_num_ctx() {
    let options = json!({
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 40,
        "min_p": 0.05,
        "repeat_penalty": 1.1,
        "num_predict": 256,
        "num_ctx": 8000,
    });
    let messages = json!([{ "role": "user", "content": "hi" }]);
    let body = build(&messages, Some(&options), None);

    assert_eq!(body["temperature"], json!(0.2));
    assert_eq!(body["top_p"], json!(0.9));
    assert_eq!(body["top_k"], json!(40));
    assert_eq!(body["min_p"], json!(0.05));
    assert_eq!(body["repeat_penalty"], json!(1.1));
    // Native uses max_output_tokens, sourced from num_predict.
    assert_eq!(body["max_output_tokens"], json!(256));
    // `num_ctx` is load-time only (the chat body ignores `context_length`); it
    // is honored out-of-band by `ensure_context_length`, never in the body.
    assert!(
        body.get("context_length").is_none(),
        "num_ctx must not leak into the native chat body"
    );
}

#[test]
fn request_prefers_max_tokens_over_num_predict() {
    let options = json!({ "max_tokens": 100, "num_predict": 256 });
    let messages = json!([{ "role": "user", "content": "hi" }]);
    let body = build(&messages, Some(&options), None);
    assert_eq!(body["max_output_tokens"], json!(100));
}

#[test]
fn request_normalizes_reasoning_levels() {
    let messages = json!([{ "role": "user", "content": "hi" }]);

    let on = build(&messages, None, Some(&json!(true)));
    assert_eq!(on["reasoning"], json!("on"));

    let off = build(&messages, None, Some(&json!(false)));
    assert_eq!(off["reasoning"], json!("off"));

    let none = build(&messages, None, Some(&json!("none")));
    assert_eq!(none["reasoning"], json!("off"));

    let high = build(&messages, None, Some(&json!("high")));
    assert_eq!(high["reasoning"], json!("high"));
}

#[test]
fn request_omits_reasoning_and_system_when_absent() {
    let messages = json!([{ "role": "user", "content": "hi" }]);
    let body = build_native_chat_request(NativeChatRequestParams {
        model_lm_studio_id: "m",
        messages: &messages,
        system_prompt: None,
        ollama_options: None,
        think: None,
        stream: true,
        integrations: None,
    });
    assert!(body.get("reasoning").is_none());
    assert!(body.get("system_prompt").is_none());
    assert_eq!(body["stream"], json!(true));
}

#[test]
fn request_skips_empty_content_messages() {
    let messages = json!([
        { "role": "user", "content": "", "images": ["iVBORw0KGgo"] },
    ]);
    let body = build(&messages, None, None);
    let input = body["input"].as_array().expect("input array");
    // Empty text content produces no message entry, just the image.
    assert_eq!(input.len(), 1);
    assert_eq!(input[0]["type"], json!("image"));
}

#[test]
fn convert_maps_message_and_reasoning_output() {
    // chat.end-style aggregate from the streaming-events doc.
    let native = json!({
        "model_instance_id": "openai/gpt-oss-20b",
        "output": [
            { "type": "reasoning", "content": "Need to call function." },
            { "type": "message", "content": "The current top-trending model is..." }
        ],
        "stats": {
            "input_tokens": 329,
            "total_output_tokens": 268,
            "reasoning_output_tokens": 5,
            "tokens_per_second": 43.73,
            "time_to_first_token_seconds": 0.781
        },
        "response_id": "resp_02b2017dbc06c12bfc353a2ed6c2b802f8cc682884bb5716"
    });

    let out = convert_native_to_ollama_chat(&native, "ollama-model", Instant::now());

    assert_eq!(out["model"], json!("ollama-model"));
    assert_eq!(out["done"], json!(true));
    assert_eq!(out["done_reason"], json!("stop"));
    assert_eq!(
        out["message"]["content"],
        json!("The current top-trending model is...")
    );
    assert_eq!(out["message"]["thinking"], json!("Need to call function."));
    assert_eq!(out["message"]["role"], json!("assistant"));
    // The non-Ollama `response_id` is dropped even when the backend sends one.
    assert!(out.get("response_id").is_none());
    // Timing mapped from the native stats block.
    assert_eq!(out["prompt_eval_count"], json!(329));
    assert_eq!(out["eval_count"], json!(268));
}

#[test]
fn convert_maps_tool_call_output() {
    let native = json!({
        "model_instance_id": "ibm/granite-4-micro",
        "output": [
            {
                "type": "tool_call",
                "tool": "model_search",
                "arguments": { "sort": "trendingScore", "query": "", "limit": 1 },
                "output": "...",
                "provider_info": { "server_label": "huggingface", "type": "ephemeral_mcp" }
            },
            { "type": "message", "content": "done" }
        ],
        "stats": {
            "input_tokens": 646,
            "total_output_tokens": 586,
            "reasoning_output_tokens": 0,
            "tokens_per_second": 29.75,
            "time_to_first_token_seconds": 1.088,
            "model_load_time_seconds": 2.656
        }
    });

    let out = convert_native_to_ollama_chat(&native, "m", Instant::now());

    assert_eq!(out["message"]["content"], json!("done"));
    let tool_calls = out["message"]["tool_calls"]
        .as_array()
        .expect("tool_calls array");
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0]["function"]["name"], json!("model_search"));
    // Native arguments are already an object; converter passes them through.
    assert_eq!(
        tool_calls[0]["function"]["arguments"],
        json!({ "sort": "trendingScore", "query": "", "limit": 1 })
    );
}

#[test]
fn convert_omits_empty_thinking_and_tool_calls() {
    let native = json!({
        "output": [{ "type": "message", "content": "hi" }],
        "stats": {}
    });
    let out = convert_native_to_ollama_chat(&native, "m", Instant::now());
    assert!(out["message"].get("thinking").is_none());
    assert!(out["message"].get("tool_calls").is_none());
    // No response_id when absent.
    assert!(out.get("response_id").is_none());
}

#[test]
fn native_tool_call_shape_matches_converter_input() {
    let item = json!({
        "type": "tool_call",
        "tool": "browser_navigate",
        "arguments": { "url": "https://lmstudio.ai" }
    });
    let shaped = native_tool_call_to_openai(&item);
    assert_eq!(shaped["function"]["name"], json!("browser_navigate"));
    assert_eq!(
        shaped["function"]["arguments"],
        json!({ "url": "https://lmstudio.ai" })
    );
}

#[test]
fn done_reason_is_stop() {
    assert_eq!(native_done_reason(), "stop");
}

#[test]
fn request_drops_tools_format_tool_choice() {
    // The native request schema has no tools/tool_choice/response_format field,
    // so the builder must never emit them (guards against a future regression).
    let messages = json!([{ "role": "user", "content": "hi" }]);
    let options = json!({
        "temperature": 0.2,
        "tools": [{ "type": "function", "function": { "name": "get_weather" } }],
        "tool_choice": "auto",
        "format": "json",
    });
    let body = build(&messages, Some(&options), None);

    assert!(body.get("tools").is_none());
    assert!(body.get("tool_choice").is_none());
    assert!(body.get("response_format").is_none());
    assert!(body.get("format").is_none());
}

#[test]
fn request_forwards_integrations_array() {
    let messages = json!([{ "role": "user", "content": "hi" }]);
    let integrations = json!([
        "huggingface",
        { "type": "plugin", "id": "browser", "allowed_tools": ["browser_navigate"] },
        {
            "type": "ephemeral_mcp",
            "server_label": "hf",
            "server_url": "https://hf.co/mcp",
            "allowed_tools": ["model_search"]
        }
    ]);
    let body = build_native_chat_request(NativeChatRequestParams {
        model_lm_studio_id: "m",
        messages: &messages,
        system_prompt: None,
        ollama_options: None,
        think: None,
        stream: false,
        integrations: Some(&integrations),
    });

    let fwd = body["integrations"].as_array().expect("integrations array");
    assert_eq!(fwd.len(), 3);
    assert_eq!(fwd[0], json!("huggingface"));
    assert_eq!(fwd[1]["type"], json!("plugin"));
    assert_eq!(fwd[2]["type"], json!("ephemeral_mcp"));
}

#[test]
fn request_omits_integrations_when_absent() {
    let messages = json!([{ "role": "user", "content": "hi" }]);
    let body = build_native_chat_request(NativeChatRequestParams {
        model_lm_studio_id: "m",
        messages: &messages,
        system_prompt: None,
        ollama_options: None,
        think: None,
        stream: false,
        integrations: None,
    });
    assert!(body.get("integrations").is_none());
}

#[test]
fn request_omits_integrations_when_not_array() {
    let messages = json!([{ "role": "user", "content": "hi" }]);
    // A non-array value must be silently dropped.
    let non_array = json!("plugin-id-string");
    let body = build_native_chat_request(NativeChatRequestParams {
        model_lm_studio_id: "m",
        messages: &messages,
        system_prompt: None,
        ollama_options: None,
        think: None,
        stream: false,
        integrations: Some(&non_array),
    });
    assert!(body.get("integrations").is_none());
}
