use super::*;

#[test]
fn detects_png_magic() {
    assert_eq!(detect_image_mime("iVBORw0KGgoAAAANSUhEUgAA"), "image/png");
}

#[test]
fn detects_gif_magic() {
    assert_eq!(detect_image_mime("R0lGODlhAQABAAAAACw="), "image/gif");
}

#[test]
fn detects_webp_magic() {
    assert_eq!(detect_image_mime("UklGRiIAAABXRUJQ"), "image/webp");
}

#[test]
fn falls_back_to_jpeg() {
    assert_eq!(detect_image_mime("/9j/4AAQSkZJRgABAQ"), "image/jpeg");
}

#[test]
fn strips_existing_data_prefix_before_sniffing() {
    let url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA";
    assert_eq!(detect_image_mime(url), "image/png");
}

#[test]
fn vision_messages_embed_image_in_content_array() {
    let images = json!(["iVBORw0KGgoAAAA"]);
    let messages = build_vision_chat_messages(None, "describe", Some(&images));
    let user = &messages.as_array().unwrap()[0];
    let content = user.get("content").unwrap().as_array().unwrap();
    assert_eq!(content[0]["type"], json!("text"));
    assert_eq!(content[0]["text"], json!("describe"));
    assert_eq!(content[1]["type"], json!("image_url"));
    let url = content[1]["image_url"]["url"].as_str().unwrap();
    assert!(url.starts_with("data:image/png;base64,"), "got {url}");
    assert!(user.get("images").is_none());
}

#[test]
fn vision_messages_without_images_keep_plain_string_content() {
    let messages = build_vision_chat_messages(Some("be helpful"), "hi", None);
    let arr = messages.as_array().unwrap();
    assert_eq!(arr[0]["role"], json!("system"));
    assert_eq!(arr[1]["role"], json!("user"));
    assert_eq!(arr[1]["content"], json!("hi"));
}

#[test]
fn inject_uses_detected_mime() {
    let messages = json!([{"role": "user", "content": "what's in here"}]);
    let images = json!(["UklGRiIAAABXRUJQ"]);
    let updated = inject_images_into_messages(messages, &images);
    let parts = updated.as_array().unwrap()[0]["content"]
        .as_array()
        .unwrap();
    let url = parts[1]["image_url"]["url"].as_str().unwrap();
    assert!(url.starts_with("data:image/webp;base64,"), "got {url}");
}

// =========================================================================
// Tests from translation_images
// =========================================================================

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
/// message regardless of role.
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
    // assistant message should be untouched
    assert_eq!(arr[1]["role"], json!("assistant"));
    assert!(arr[1]["content"].is_string(), "assistant must keep string content");
    // user message should have image content parts
    let content = arr[0]["content"].as_array().unwrap();
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
#[test]
fn top_level_images_with_no_user_message_skipped() {
    let messages = json!([
        {"role": "system", "content": "s"}
    ]);
    let images = json!(["iVBORw0KGgo"]);
    let result = inject_images_into_messages(messages.clone(), &images);
    let arr = result.as_array().unwrap();
    assert_eq!(arr[0]["role"], json!("system"));
    assert!(
        arr[0]["content"].is_string(),
        "system message must not be turned into a vision content array, got {}",
        arr[0]["content"]
    );
}

// =========================================================================
// Tests from translation_misc_helpers — images section
// =========================================================================

/// With a system prompt and no images, the user `content` stays a plain string.
#[test]
fn vision_messages_system_plus_no_images_keeps_string_content() {
    let messages = build_vision_chat_messages(Some("be brief"), "hello", None);
    let arr = messages.as_array().unwrap();
    assert_eq!(arr.len(), 2);
    assert_eq!(arr[0]["role"], json!("system"));
    assert_eq!(arr[0]["content"], json!("be brief"));
    assert_eq!(arr[1]["role"], json!("user"));
    assert_eq!(
        arr[1]["content"],
        json!("hello"),
        "no images → content must remain a plain string"
    );
}

/// With a system prompt AND images, the user `content` becomes typed content-parts.
#[test]
fn vision_messages_system_plus_images_yields_typed_parts() {
    let images_val = json!(["iVBORw0KGgoAAA"]);
    let messages =
        build_vision_chat_messages(Some("be brief"), "describe", Some(&images_val));
    let arr = messages.as_array().unwrap();
    assert_eq!(arr.len(), 2);
    assert_eq!(arr[0]["role"], json!("system"));
    let parts = arr[1]["content"].as_array().expect("content must be array");
    assert_eq!(parts[0]["type"], json!("text"));
    assert_eq!(parts[0]["text"], json!("describe"));
    assert_eq!(parts[1]["type"], json!("image_url"));
    let url = parts[1]["image_url"]["url"].as_str().unwrap();
    assert!(url.starts_with("data:image/png;base64,"), "got {url}");
}

/// No system prompt → output is exactly one user message.
#[test]
fn vision_messages_no_system_yields_only_user_message() {
    let messages = build_vision_chat_messages(None, "hi", None);
    let arr = messages.as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["role"], json!("user"));
    assert_eq!(arr[0]["content"], json!("hi"));
}

/// images is an empty array → still a plain-string user content.
#[test]
fn vision_messages_empty_images_array_keeps_string_content() {
    let images_val = json!([]);
    let messages = build_vision_chat_messages(None, "hi", Some(&images_val));
    let arr = messages.as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(
        arr[0]["content"],
        json!("hi"),
        "empty images must not promote content to an array"
    );
}

/// images contains non-string entries → they are skipped silently.
#[test]
fn vision_messages_non_string_images_are_skipped() {
    let images_val = json!([42, { "url": "ignored" }]);
    let messages = build_vision_chat_messages(None, "hi", Some(&images_val));
    let arr = messages.as_array().unwrap();
    assert_eq!(
        arr[0]["content"],
        json!("hi"),
        "non-string image entries must be dropped, leaving plain content"
    );
}

/// convert_per_message_images: non-array input is returned unchanged.
#[test]
fn convert_per_message_images_non_array_passthrough() {
    let messages = json!({ "not": "an array" });
    let out = convert_per_message_images(messages.clone());
    assert_eq!(out, messages);
}

/// inject_images_into_messages: empty images array → messages returned as-is.
#[test]
fn inject_empty_images_returns_messages_unchanged() {
    let messages = json!([{ "role": "user", "content": "hi" }]);
    let images_val = json!([]);
    let out = inject_images_into_messages(messages.clone(), &images_val);
    assert_eq!(out, messages);
}
