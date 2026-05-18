use std::borrow::Cow;

use serde_json::{Value, json};

fn image_data_url(base64_data: &str) -> String {
    format!(
        "data:{};base64,{}",
        detect_image_mime(base64_data),
        base64_data
    )
}

/// Sniff the base64 magic prefix to infer an image MIME type.
///
/// LM Studio's OpenAI-compat endpoint forwards the data URL to the model
/// runtime; some runtimes require an accurate MIME instead of always-`image/jpeg`.
/// We decode just enough of the first base64 group to inspect the magic bytes.
fn detect_image_mime(base64_data: &str) -> &'static str {
    // Strip any leading "data:...;base64," prefix the caller may have included.
    let payload = base64_data
        .find("base64,")
        .map_or(base64_data, |i| &base64_data[i + 7..]);
    let prefix: String = payload
        .chars()
        .take(16)
        .filter(|c| !c.is_whitespace())
        .collect();
    if prefix.starts_with("iVBORw0KGgo") {
        "image/png"
    } else if prefix.starts_with("R0lGODdh") || prefix.starts_with("R0lGODlh") {
        "image/gif"
    } else if prefix.starts_with("UklGR") {
        "image/webp"
    } else if prefix.starts_with("Qk") {
        "image/bmp"
    } else {
        "image/jpeg"
    }
}

fn build_image_parts(images: &Value) -> Vec<Value> {
    let Some(image_array) = images.as_array() else {
        return Vec::new();
    };
    image_array
        .iter()
        .filter_map(|img| {
            img.as_str().map(|base64_data| {
                json!({
                    "type": "image_url",
                    "image_url": { "url": image_data_url(base64_data) }
                })
            })
        })
        .collect()
}

fn content_to_text_part(content: &Value) -> Value {
    let text = content
        .as_str()
        .map(Cow::Borrowed)
        .unwrap_or(Cow::Owned(content.to_string()));
    json!({ "type": "text", "text": text })
}

/// Injects base64 encoded images into chat messages
pub fn inject_images_into_messages(messages: Value, images: &Value) -> Value {
    let image_parts = build_image_parts(images);
    if image_parts.is_empty() {
        return messages;
    }
    let Some(msg_array) = messages.as_array() else {
        return messages;
    };

    let mut updated = msg_array.clone();
    if let Some(last_msg) = updated.last_mut()
        && let Some(obj) = last_msg.as_object_mut()
        && let Some(content) = obj.get("content")
    {
        let mut parts = vec![content_to_text_part(content)];
        parts.extend(image_parts);
        obj.insert("content".to_string(), Value::Array(parts));
    }

    Value::Array(updated)
}

/// Builds vision chat messages from prompt and images
pub fn build_vision_chat_messages(
    system_prompt: Option<&str>,
    prompt: &str,
    images: Option<&Value>,
) -> Value {
    let mut message_list = Vec::new();
    if let Some(system_text) = system_prompt {
        message_list.push(json!({
            "role": "system",
            "content": system_text,
        }));
    }

    let image_parts = images.map(build_image_parts).unwrap_or_default();
    let user_content: Value = if image_parts.is_empty() {
        Value::String(prompt.to_string())
    } else {
        let mut parts = vec![json!({ "type": "text", "text": prompt })];
        parts.extend(image_parts);
        Value::Array(parts)
    };

    message_list.push(json!({
        "role": "user",
        "content": user_content,
    }));

    Value::Array(message_list)
}

#[cfg(test)]
mod tests {
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
        // The legacy `images` sibling must not be present.
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
}
