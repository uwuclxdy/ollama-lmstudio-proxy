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
