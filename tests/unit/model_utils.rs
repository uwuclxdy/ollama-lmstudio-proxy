use super::*;

// ─── clean_model_name ────────────────────────────────────────────────────────

#[test]
fn clean_model_name_passthrough_plain() {
    assert_eq!(clean_model_name("llama3"), "llama3");
}

#[test]
fn clean_model_name_strips_latest_suffix() {
    assert_eq!(clean_model_name("llama3:latest"), "llama3");
}

#[test]
fn clean_model_name_preserves_size_tag() {
    // Per api_docs/ollama.md "Model names": tags identify specific versions;
    // only ":latest" is the default and gets stripped.
    assert_eq!(clean_model_name("llama3.1:8b"), "llama3.1:8b");
}

#[test]
fn clean_model_name_preserves_multi_digit_size_tag() {
    assert_eq!(clean_model_name("mistral:70b"), "mistral:70b");
}

#[test]
fn clean_model_name_keeps_non_numeric_tag() {
    // ":instruct" is not all digits — keep as-is
    assert_eq!(clean_model_name("llama3:instruct"), "llama3:instruct");
}

#[test]
fn clean_model_name_keeps_mixed_alphanumeric_tag() {
    // "q4_0" contains letters, so the colon-based strip should NOT fire
    assert_eq!(clean_model_name("llama3:q4_0"), "llama3:q4_0");
}

#[test]
fn clean_model_name_empty_string_is_identity() {
    assert_eq!(clean_model_name(""), "");
}

#[test]
fn clean_model_name_no_colon_at_all() {
    assert_eq!(clean_model_name("meta-llama-3-8b-instruct"), "meta-llama-3-8b-instruct");
}

#[test]
fn clean_model_name_latest_takes_precedence_over_numeric_colon() {
    // "llama3:latest" — rfind(":latest") fires first, colon scan never runs
    assert_eq!(clean_model_name("llama3:latest"), "llama3");
}

#[test]
fn clean_model_name_namespace_slash_preserves_size_tag() {
    // Per spec: only ":latest" is stripped — size tags like ":7b" identify a version.
    assert_eq!(clean_model_name("user/model:7b"), "user/model:7b");
}

#[test]
fn clean_model_name_namespace_slash_latest() {
    assert_eq!(clean_model_name("user/model:latest"), "user/model");
}

#[test]
fn clean_model_name_only_colon_is_not_numeric() {
    // edge: colon at position 0 — colon_pos == 0, guard `colon_pos > 0` blocks strip
    assert_eq!(clean_model_name(":8b"), ":8b");
}

#[test]
fn clean_model_name_very_long_name_no_tag() {
    let long = "a".repeat(512);
    assert_eq!(clean_model_name(&long), long.as_str());
}

#[test]
fn clean_model_name_very_long_name_with_size_tag_preserved() {
    let mut name = "a".repeat(512);
    name.push_str(":8b");
    assert_eq!(clean_model_name(&name), name.as_str());
}

#[test]
fn clean_model_name_special_chars_in_basename() {
    // Dots and dashes in the base are preserved
    assert_eq!(
        clean_model_name("meta-llama-3.1-8b-instruct:latest"),
        "meta-llama-3.1-8b-instruct"
    );
}

#[test]
fn clean_model_name_only_digits_no_colon() {
    assert_eq!(clean_model_name("123"), "123");
}

#[test]
fn clean_model_name_multiple_colons_preserve_non_latest_tag() {
    // Per spec: only ":latest" is stripped — ":70b" is a version tag and stays.
    assert_eq!(clean_model_name("a:b:70b"), "a:b:70b");
}

#[test]
fn clean_model_name_multiple_colons_last_is_non_numeric() {
    // "a:70b:instruct" — last suffix "instruct" is not digits, no strip
    assert_eq!(clean_model_name("a:70b:instruct"), "a:70b:instruct");
}

#[test]
fn clean_model_name_strips_latest_after_size_tag() {
    // "model:7b:latest" — only ":latest" strips; the residual ":7b" stays.
    assert_eq!(clean_model_name("model:7b:latest"), "model:7b");
}

// ─── extract_required_model_name ─────────────────────────────────────────────

use serde_json::json;

#[test]
fn extract_required_model_name_returns_value() {
    let body = json!({ "model": "llama3" });
    assert_eq!(extract_required_model_name(&body).unwrap(), "llama3");
}

#[test]
fn extract_required_model_name_missing_field_is_err() {
    let body = json!({ "prompt": "hello" });
    let err = extract_required_model_name(&body).unwrap_err();
    assert_eq!(err.status_code, 400);
}

#[test]
fn extract_required_model_name_null_field_is_err() {
    let body = json!({ "model": null });
    let err = extract_required_model_name(&body).unwrap_err();
    assert_eq!(err.status_code, 400);
}

#[test]
fn extract_required_model_name_empty_string_is_err() {
    let body = json!({ "model": "" });
    let err = extract_required_model_name(&body).unwrap_err();
    assert_eq!(err.status_code, 400);
}

#[test]
fn extract_required_model_name_wrong_type_number_is_err() {
    let body = json!({ "model": 42 });
    let err = extract_required_model_name(&body).unwrap_err();
    assert_eq!(err.status_code, 400);
}

#[test]
fn extract_required_model_name_wrong_type_bool_is_err() {
    let body = json!({ "model": true });
    let err = extract_required_model_name(&body).unwrap_err();
    assert_eq!(err.status_code, 400);
}

#[test]
fn extract_required_model_name_wrong_type_array_is_err() {
    let body = json!({ "model": ["llama3"] });
    let err = extract_required_model_name(&body).unwrap_err();
    assert_eq!(err.status_code, 400);
}

#[test]
fn extract_required_model_name_with_tag_returned_verbatim() {
    // The function does not clean — raw value is returned.
    let body = json!({ "model": "llama3.1:8b" });
    assert_eq!(extract_required_model_name(&body).unwrap(), "llama3.1:8b");
}

#[test]
fn extract_required_model_name_with_latest_tag_returned_verbatim() {
    let body = json!({ "model": "mistral:latest" });
    assert_eq!(extract_required_model_name(&body).unwrap(), "mistral:latest");
}

#[test]
fn extract_required_model_name_whitespace_only_is_not_empty_passes() {
    // filter only checks is_empty(); a whitespace-only string is non-empty.
    let body = json!({ "model": "   " });
    assert_eq!(extract_required_model_name(&body).unwrap(), "   ");
}

#[test]
fn extract_required_model_name_special_chars_pass_through() {
    let body = json!({ "model": "user/model-name:latest" });
    assert_eq!(
        extract_required_model_name(&body).unwrap(),
        "user/model-name:latest"
    );
}

#[test]
fn extract_required_model_name_error_message_contains_missing_model() {
    use crate::constants::ERROR_MISSING_MODEL;
    let body = json!({});
    let err = extract_required_model_name(&body).unwrap_err();
    assert!(
        err.message.contains(ERROR_MISSING_MODEL),
        "error message should reference the constant: got '{}'",
        err.message
    );
}
