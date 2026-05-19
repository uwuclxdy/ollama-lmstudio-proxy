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
    let result = ResponseTransformer::convert_to_ollama_chat(&lm, "m", 2, Instant::now(), false);
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
    let result =
        ResponseTransformer::convert_to_ollama_generate(&lm, "mymodel", "q", Instant::now(), false);
    assert!(result.get("thinking").is_none());
}
