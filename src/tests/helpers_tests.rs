use crate::handlers::helpers::*;
use serde_json::{json, Value};

#[test]
fn completion_with_images_becomes_chat_payload() {
    let images = json!(["img-data"]);
    let request = build_lm_studio_request(
        "test-model",
        LMStudioRequestType::Completion {
            prompt: "describe the image",
            stream: true,
            images: Some(&images),
        },
        None,
        None,
        None,
    );

    assert!(request.get("messages").is_some());
    assert!(request.get("prompt").is_none());
}

#[test]
fn chat_request_includes_tools_and_schema_format() {
    let messages = json!([
        {"role": "user", "content": "Say hi"}
    ]);
    let tools = json!([
        {"type": "function", "function": {"name": "hi", "parameters": {"type": "object"}}}
    ]);
    let schema = json!({"type": "object", "properties": {"answer": {"type": "string"}}});

    let request = build_lm_studio_request(
        "test-model",
        LMStudioRequestType::Chat {
            messages: &messages,
            stream: false,
        },
        None,
        Some(&tools),
        Some(&schema),
    );

    assert_eq!(request.get("tools"), Some(&tools));
    let response_format = request["response_format"].clone();
    assert_eq!(response_format["json_schema"]["schema"], schema);
}

#[test]
fn json_mode_format_creates_json_object_response_format() {
    let messages = json!([
        {"role": "user", "content": "respond with json"}
    ]);
    let format_value = Value::String("json".into());

    let request = build_lm_studio_request(
        "test-model",
        LMStudioRequestType::Chat {
            messages: &messages,
            stream: true,
        },
        None,
        None,
        Some(&format_value),
    );

    assert_eq!(request["response_format"], json!({"type": "json_object"}));
    // ensure original format value remains unchanged for caller reuse
    assert_eq!(format_value, Value::String("json".into()));
}
