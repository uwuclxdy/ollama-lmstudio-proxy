//! Tests for keep_alive translation.
//!
//! Ollama keep_alive semantics (api_docs/ollama.md, Modelfile keep_alive section):
//!   - positive int N seconds        → keep loaded for N seconds
//!   - 0                              → unload immediately
//!   - negative (e.g. -1, "-1s")      → keep loaded forever (no TTL)
//!   - duration string ("10m", "5h", "500ms")
//!
//! LM Studio /api/v0/* `ttl` field expects a non-negative number of seconds.
//! Therefore negative Ollama values must NOT be forwarded as a negative ttl;
//! the proxy should omit the field (LM Studio default = unbounded) instead.

#[path = "../src/constants.rs"]
#[allow(dead_code)]
mod constants;

#[path = "../src/error.rs"]
#[allow(dead_code)]
mod error;

#[path = "../src/handlers/ollama/keep_alive_parse.rs"]
#[allow(dead_code)]
mod keep_alive_parse;

use keep_alive_parse::{apply_keep_alive_ttl, parse_keep_alive_seconds};
use serde_json::{Value, json};

fn parse(v: Value) -> Option<i64> {
    parse_keep_alive_seconds(Some(&v)).expect("keep_alive parse should not fail")
}

#[test]
fn positive_integer_passes_through() {
    assert_eq!(parse(json!(300)), Some(300));
}

#[test]
fn zero_unloads_immediately() {
    assert_eq!(parse(json!(0)), Some(0));
}

#[test]
fn negative_integer_normalizes_to_minus_one() {
    // Ollama treats any negative as "stay loaded forever".
    assert_eq!(parse(json!(-1)), Some(-1));
    assert_eq!(parse(json!(-3600)), Some(-1));
}

#[test]
fn duration_string_minutes() {
    assert_eq!(parse(json!("5m")), Some(300));
}

#[test]
fn duration_string_compound() {
    assert_eq!(parse(json!("1h30m")), Some(3600 + 30 * 60));
}

#[test]
fn sub_second_duration_rounds_up_to_one() {
    // "500ms" must NOT round down to 0 (which would trigger unload).
    assert_eq!(parse(json!("500ms")), Some(1));
}

#[test]
fn negative_string_normalizes_to_minus_one() {
    // bare "-1" parsed as integer
    assert_eq!(parse(json!("-1")), Some(-1));
}

#[test]
fn null_returns_none() {
    let v = Value::Null;
    assert!(parse_keep_alive_seconds(Some(&v)).unwrap().is_none());
}

#[test]
fn missing_returns_none() {
    assert!(parse_keep_alive_seconds(None).unwrap().is_none());
}

#[test]
fn apply_ttl_negative_omits_field() {
    // Negative keep_alive means "forever" — must NOT be forwarded as a negative
    // ttl (LM Studio would reject or misinterpret).
    let mut target = json!({"model": "x"});
    apply_keep_alive_ttl(&mut target, Some(-1));
    assert!(
        target.get("ttl").is_none(),
        "ttl must be omitted when keep_alive is negative, got {}",
        target
    );
}

#[test]
fn apply_ttl_positive_sets_field() {
    let mut target = json!({"model": "x"});
    apply_keep_alive_ttl(&mut target, Some(300));
    assert_eq!(target.get("ttl"), Some(&json!(300)));
}

#[test]
fn apply_ttl_zero_sets_field() {
    let mut target = json!({"model": "x"});
    apply_keep_alive_ttl(&mut target, Some(0));
    assert_eq!(target.get("ttl"), Some(&json!(0)));
}

#[test]
fn apply_ttl_none_omits_field() {
    let mut target = json!({"model": "x"});
    apply_keep_alive_ttl(&mut target, None);
    assert!(target.get("ttl").is_none());
}
