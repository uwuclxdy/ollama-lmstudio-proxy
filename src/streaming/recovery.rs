use serde_json::{Value, json};

pub fn recover_json_from_chunk(chunk_data: &str) -> Option<Value> {
    if let Some(start_brace) = chunk_data.find('{')
        && let Some(end_brace) = chunk_data.rfind('}')
        && start_brace < end_brace
    {
        let potential_json = &chunk_data[start_brace..=end_brace];
        if let Ok(parsed) = serde_json::from_str::<Value>(potential_json) {
            return Some(parsed);
        }
    }

    if let Some(start_bracket) = chunk_data.find('[')
        && let Some(end_bracket) = chunk_data.rfind(']')
        && start_bracket < end_bracket
    {
        let potential_json = &chunk_data[start_bracket..=end_bracket];
        if let Ok(parsed) = serde_json::from_str::<Value>(potential_json) {
            return Some(parsed);
        }
    }

    let cleaned_data = chunk_data
        .replace(",\n}", "\n}")
        .replace(",\n]", "\n]")
        .replace(":\n", ": \"\"");

    if let Ok(parsed) = serde_json::from_str::<Value>(&cleaned_data) {
        return Some(parsed);
    }

    if let Some(choices_start) = chunk_data.find("\"choices\":")
        && let Some(array_start) = chunk_data[choices_start..].find('[')
    {
        let choices_start_pos = choices_start + array_start;
        if let Some(array_end) = chunk_data[choices_start_pos..].rfind(']') {
            let choices_json = &chunk_data[choices_start_pos..=choices_start_pos + array_end];
            if let Ok(parsed) = serde_json::from_str::<Value>(choices_json) {
                let mut result = json!({});
                result["choices"] = parsed;
                return Some(result);
            }
        }
    }

    None
}
