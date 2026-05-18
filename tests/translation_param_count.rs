//! Tests for parameter-count parsing used to populate Ollama's
//! `general.parameter_count` field on /api/show.
//!
//! Reference: api_docs/ollama.md line 1485:
//!   "general.parameter_count": 8030261248
//! (8030261248 ≈ 8 billion — paired with parameter_size: "8.0B").

#[path = "../src/model/param_count.rs"]
#[allow(dead_code)]
mod param_count;

use param_count::parse_parameter_count;

#[test]
fn parses_billions() {
    assert_eq!(parse_parameter_count("7B"), Some(7_000_000_000));
    assert_eq!(parse_parameter_count("70B"), Some(70_000_000_000));
}

#[test]
fn parses_decimal_billions() {
    assert_eq!(parse_parameter_count("1.5B"), Some(1_500_000_000));
    assert_eq!(parse_parameter_count("0.5B"), Some(500_000_000));
    assert_eq!(parse_parameter_count("8.0B"), Some(8_000_000_000));
}

#[test]
fn parses_millions() {
    assert_eq!(parse_parameter_count("500M"), Some(500_000_000));
    assert_eq!(parse_parameter_count("125M"), Some(125_000_000));
}

#[test]
fn case_insensitive_suffix() {
    assert_eq!(parse_parameter_count("7b"), Some(7_000_000_000));
    assert_eq!(parse_parameter_count("500m"), Some(500_000_000));
}

#[test]
fn unknown_returns_none() {
    assert_eq!(parse_parameter_count("unknown"), None);
    assert_eq!(parse_parameter_count(""), None);
    assert_eq!(parse_parameter_count("   "), None);
}

#[test]
fn garbage_returns_none() {
    assert_eq!(parse_parameter_count("abc"), None);
    assert_eq!(parse_parameter_count("xyz123"), None);
}

#[test]
fn bare_number_treated_as_count() {
    // LM Studio sometimes already gives the raw count.
    assert_eq!(parse_parameter_count("1234567"), Some(1_234_567));
}
