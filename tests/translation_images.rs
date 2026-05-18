//! Tests for image translation between Ollama and LM Studio (OpenAI-compat).
//!
//! Ollama /api/chat spec (api_docs/ollama.md lines 512-513, 979-980):
//!   `images` is a PER-MESSAGE optional field on each message object, a list of
//!   base64-encoded images for multimodal models. Example:
//!     {"role": "user", "content": "what is in this image?", "images": ["..."]}
//!
//! Ollama /api/generate spec (api_docs/ollama.md line 48, 316):
//!   `images` is a TOP-LEVEL optional field (no per-message structure exists in
//!   completion-style requests).
//!
//! LM Studio's OpenAI-compat /v1/chat/completions accepts OpenAI-style content
//! parts: content becomes an array of {"type":"text","text":"..."} and
//! {"type":"image_url","image_url":{"url":"data:image/...;base64,..."}}.

#[path = "../src/handlers/ollama/images.rs"]
#[allow(dead_code)]
mod images;

use images::{convert_per_message_images, inject_images_into_messages};
use serde_json::json;

/// /api/chat: each message can carry its own `images` array; the converter
/// must move those images into OpenAI content parts on the SAME message.
#[test]
fn per_message_images_become_content_parts_on_same_message() {
    let messages = json!([
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "what is in this image?", "images": ["iVBORw0KGgo"]}
    ]);
    let result = convert_per_message_images(messages);
    let arr = result.as_array().unwrap();
    // system unchanged
    assert_eq!(arr[0]["role"], json!("system"));
    assert_eq!(arr[0]["content"], json!("be helpful"));
    // user message: content becomes an array, images sibling removed
    assert!(
        arr[1].get("images").is_none(),
        "per-message images must be stripped"
    );
    let content = arr[1]["content"].as_array().expect("content must be array");
    assert_eq!(content[0]["type"], json!("text"));
    assert_eq!(content[0]["text"], json!("what is in this image?"));
    assert_eq!(content[1]["type"], json!("image_url"));
}

/// A message without images stays untouched (no array conversion, no images key).
#[test]
fn messages_without_images_unchanged() {
    let messages = json!([
        {"role": "user", "content": "hi"}
    ]);
    let result = convert_per_message_images(messages.clone());
    assert_eq!(result, messages);
}

/// /api/generate top-level images go onto the last USER message, not the last
/// message regardless of role. Reference: api_docs/ollama.md §"Generate a chat
/// completion" — images travel with the user turn.
#[test]
fn top_level_images_target_last_user_message_not_last_message() {
    // Last message is `assistant`; images must land on the prior `user` message.
    let messages = json!([
        {"role": "user", "content": "what is in this?"},
        {"role": "assistant", "content": "let me think"}
    ]);
    let images = json!(["iVBORw0KGgo"]);
    let result = inject_images_into_messages(messages, &images);
    let arr = result.as_array().unwrap();

    // assistant must NOT have been turned into an image-carrying message.
    let assistant = &arr[1];
    assert_eq!(assistant["role"], json!("assistant"));
    assert!(
        assistant["content"].is_string(),
        "assistant content must remain a plain string, got {}",
        assistant["content"]
    );

    // The user message must now have a content-parts array.
    let user = &arr[0];
    let content = user["content"]
        .as_array()
        .expect("user content must be array");
    assert_eq!(content[0]["type"], json!("text"));
    assert_eq!(content[0]["text"], json!("what is in this?"));
    assert_eq!(content[1]["type"], json!("image_url"));
}

/// Top-level images with a user message as the trailing one still go there.
#[test]
fn top_level_images_target_last_user_message_when_last() {
    let messages = json!([
        {"role": "system", "content": "s"},
        {"role": "user", "content": "describe"}
    ]);
    let images = json!(["iVBORw0KGgo"]);
    let result = inject_images_into_messages(messages, &images);
    let arr = result.as_array().unwrap();
    let user = &arr[1];
    let content = user["content"].as_array().unwrap();
    assert_eq!(content[1]["type"], json!("image_url"));
}

/// When there is no user message at all, top-level images cannot be injected.
/// The proxy must NOT silently attach them to a system/assistant message.
#[test]
fn top_level_images_with_no_user_message_skipped() {
    let messages = json!([
        {"role": "system", "content": "s"}
    ]);
    let images = json!(["iVBORw0KGgo"]);
    let result = inject_images_into_messages(messages.clone(), &images);
    // Either unchanged or with images-on-system not produced.
    let arr = result.as_array().unwrap();
    assert_eq!(arr[0]["role"], json!("system"));
    assert!(
        arr[0]["content"].is_string(),
        "system message must not be turned into a vision content array, got {}",
        arr[0]["content"]
    );
}
