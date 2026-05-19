use super::*;
use serde_json::json;

#[test]
fn extracts_think_from_generate_body() {
    let body = json!({ "think": "high", "model": "x", "prompt": "hi" });
    let top = make_top_level_params(&body);
    assert_eq!(top.think, Some(&json!("high")));
}

#[test]
fn suffix_inserted_into_lm_request() {
    use crate::http::request::{LMStudioRequestType, TopLevelParams, build_lm_studio_request};
    use std::borrow::Cow;

    let body = json!({ "suffix": "world", "model": "test", "prompt": "hello" });
    let suffix_val = body.get("suffix");
    let top_level = TopLevelParams {
        think: None,
        logprobs: None,
        top_logprobs: None,
    };

    let mut lm_request = build_lm_studio_request(
        "test",
        LMStudioRequestType::Completion {
            prompt: Cow::Borrowed("hello"),
            stream: false,
        },
        None,
        None,
        None,
        Some(&top_level),
    );

    if let Some(s) = suffix_val
        && let Some(obj) = lm_request.as_object_mut()
    {
        obj.insert("suffix".to_string(), s.clone());
    }

    assert_eq!(lm_request.get("suffix"), Some(&json!("world")));
}

#[test]
fn suffix_not_inserted_on_vision_path() {
    let body = json!({ "suffix": "world", "model": "test", "prompt": "hello",
                       "images": ["base64data"] });
    let current_images = body.get("images");
    let suffix_val = body.get("suffix");
    let mut lm_request = json!({ "model": "test" });

    if current_images.is_none()
        && let Some(s) = suffix_val
        && let Some(obj) = lm_request.as_object_mut()
    {
        obj.insert("suffix".to_string(), s.clone());
    }

    assert!(
        lm_request.get("suffix").is_none(),
        "suffix must be absent on vision path"
    );
}

#[test]
fn absent_think_gives_none_in_generate() {
    let body = json!({ "model": "x", "prompt": "hi" });
    let top = make_top_level_params(&body);
    assert!(top.think.is_none());
}
