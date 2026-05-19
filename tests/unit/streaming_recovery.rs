use super::*;

// ════════════════════════════════════════════════════════════════════════════
// recover_json_from_chunk — object extraction
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn valid_json_object_returned_directly() {
    let input = r#"{"choices":[{"delta":{"content":"hi"}}]}"#;
    let result = recover_json_from_chunk(input);
    assert!(result.is_some());
    assert!(result.unwrap().get("choices").is_some());
}

#[test]
fn valid_json_array_returned_directly() {
    let input = r#"[{"delta":{"content":"chunk"}}]"#;
    let result = recover_json_from_chunk(input);
    assert!(result.is_some());
    assert!(result.unwrap().is_array());
}

#[test]
fn empty_string_returns_none() {
    assert!(recover_json_from_chunk("").is_none());
}

#[test]
fn whitespace_only_returns_none() {
    assert!(recover_json_from_chunk("   \n\t  ").is_none());
}

#[test]
fn completely_invalid_input_returns_none() {
    assert!(recover_json_from_chunk("not json at all").is_none());
}

#[test]
fn object_embedded_in_garbage_prefix_extracted() {
    let input = r#"some garbage before {"id":"x","object":"y"} and after"#;
    let result = recover_json_from_chunk(input);
    assert!(result.is_some(), "must extract embedded JSON object");
    let v = result.unwrap();
    assert_eq!(v.get("id").and_then(|v| v.as_str()), Some("x"));
}

#[test]
fn trailing_comma_before_closing_brace_recovered() {
    // ,\n} is a common malformation from partial SSE chunks
    let input = "{\n\"key\": \"val\",\n}";
    // cleaned_data path replaces ",\n}" with "\n}"
    let result = recover_json_from_chunk(input);
    assert!(result.is_some(), "trailing comma before }} must be cleaned");
}

#[test]
fn trailing_comma_before_closing_bracket_recovered() {
    let input = "[1, 2,\n]";
    let result = recover_json_from_chunk(input);
    assert!(result.is_some(), "trailing comma before ] must be cleaned");
}

#[test]
fn choices_array_extracted_when_top_level_broken() {
    // A chunk where the outer object is broken but "choices" key is present
    let input = r#"BROKEN{"choices":[{"delta":{"content":"ok"}}]BROKEN"#;
    let result = recover_json_from_chunk(input);
    // Either brace extraction finds it or choices-key path handles it
    assert!(
        result.is_some(),
        "must recover choices array from broken wrapper"
    );
    let v = result.unwrap();
    // The recovered value has a choices key (either at root or the parsed array itself)
    let has_choices = v.get("choices").is_some() || v.is_array();
    assert!(
        has_choices,
        "recovered value must contain choices data; got {v}"
    );
}

#[test]
fn object_with_nested_braces_recovered_correctly() {
    let input = r#"{"outer":{"inner":"value"},"num":42}"#;
    let result = recover_json_from_chunk(input);
    assert!(result.is_some());
    let v = result.unwrap();
    assert_eq!(v.get("num").and_then(|v| v.as_i64()), Some(42));
}

#[test]
fn start_brace_after_end_brace_returns_none() {
    // start > end, so brace path is skipped; also not valid JSON
    let input = "} some text {";
    // No valid JSON recoverable
    let result = recover_json_from_chunk(input);
    assert!(result.is_none());
}

#[test]
fn array_with_nested_brackets_extracted() {
    let input = r#"noise [{"a":[1,2,3]},{"b":"x"}] noise"#;
    let result = recover_json_from_chunk(input);
    assert!(result.is_some());
    let v = result.unwrap();
    assert!(v.is_array());
    assert_eq!(v.as_array().unwrap().len(), 2);
}

#[test]
fn multiple_recovery_attempts_do_not_panic_on_weird_input() {
    let inputs = [
        "{{{",
        "}}}",
        "[[[",
        "]]]",
        r#"{"unterminated"#,
        r#"{"a":}"#,
        "\0\0\0",
        "data: [DONE]",
    ];
    for input in inputs {
        let _ = recover_json_from_chunk(input);
    }
}
