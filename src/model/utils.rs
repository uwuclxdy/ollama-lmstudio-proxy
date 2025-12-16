pub fn clean_model_name(name: &str) -> &str {
    if name.is_empty() {
        return name;
    }
    let after_latest = if let Some(pos) = name.rfind(":latest") {
        &name[..pos]
    } else {
        name
    };
    if let Some(colon_pos) = after_latest.rfind(':') {
        let suffix = &after_latest[colon_pos + 1..];
        if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) && colon_pos > 0 {
            return &after_latest[..colon_pos];
        }
    }
    after_latest
}
