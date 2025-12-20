use std::borrow::Cow;

use serde_json::{Value, json};

/// Injects base64 encoded images into chat messages
pub fn inject_images_into_messages(messages: Value, images: &Value) -> Value {
    let Some(image_array) = images.as_array() else {
        return messages;
    };
    if image_array.is_empty() {
        return messages;
    }

    let Some(msg_array) = messages.as_array() else {
        return messages;
    };

    let image_parts: Vec<Value> = image_array
        .iter()
        .filter_map(|img| {
            img.as_str().map(|base64_data| {
                json!({
                    "type": "image_url",
                    "image_url": {
                        "url": format!("data:image/jpeg;base64,{}", base64_data)
                    }
                })
            })
        })
        .collect();

    if image_parts.is_empty() {
        return messages;
    }

    let mut updated = msg_array.clone();
    if let Some(last_msg) = updated.last_mut()
        && let Some(obj) = last_msg.as_object_mut()
        && let Some(content) = obj.get("content")
    {
        let text_part = json!({
            "type": "text",
            "text": content.as_str().map(Cow::Borrowed).unwrap_or(Cow::Owned(content.to_string()))
        });
        let mut parts = vec![text_part];
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

    let mut user_message = json!({
        "role": "user",
        "content": prompt,
    });
    let obj = user_message.as_object_mut();
    if let Some(obj) = obj
        && let Some(img_value) = images {
            obj.insert("images".to_string(), img_value.clone());
        }
    message_list.push(user_message);

    Value::Array(message_list)
}
