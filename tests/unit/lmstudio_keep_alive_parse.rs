use super::*;
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
fn duration_string_fractional_units() {
    // Go's time.ParseDuration — what Ollama parses this string with — accepts a
    // decimal part and a leading plus sign.
    assert_eq!(parse(json!("1.5h")), Some(5400));
    assert_eq!(parse(json!("+1h")), Some(3600));
}

#[test]
fn duration_string_micro_second_spellings() {
    // Go accepts both the micro sign and the Greek mu for microseconds; the
    // sub-second clamp keeps the backend from unloading the model.
    assert_eq!(parse(json!("1\u{b5}s")), Some(1));
    assert_eq!(parse(json!("1\u{3bc}s")), Some(1));
}

#[test]
fn duration_string_rejects_units_go_does_not_have() {
    // Ollama parses this string with Go's time.ParseDuration, which has no day,
    // week, month or year units, and no spaces between number and unit.
    for raw in ["1d", "1 hour", "2weeks", "1y"] {
        assert!(
            parse_keep_alive_seconds(Some(&json!(raw))).is_err(),
            "{raw:?} must be refused, as Ollama refuses it"
        );
    }
}

#[test]
fn duration_string_rejects_overflow_past_go_range() {
    // A Go Duration is int64 nanoseconds (~2562047h); beyond that it errors.
    assert!(
        parse_keep_alive_seconds(Some(&json!("2562048h"))).is_err(),
        "a duration past Go's int64-nanosecond range must be refused"
    );
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
fn apply_ttl_none_does_nothing() {
    let mut target = json!({"model": "x"});
    apply_keep_alive_ttl(&mut target, None);
    assert!(target.get("ttl").is_none());
}

#[test]
fn keep_alive_requests_unload_true_for_zero() {
    assert!(keep_alive_requests_unload(Some(0)));
}

#[test]
fn keep_alive_requests_unload_false_for_positive() {
    assert!(!keep_alive_requests_unload(Some(300)));
}

#[test]
fn keep_alive_requests_unload_false_for_negative() {
    assert!(!keep_alive_requests_unload(Some(-1)));
}

#[test]
fn keep_alive_requests_unload_false_for_none() {
    assert!(!keep_alive_requests_unload(None));
}

#[test]
fn integral_float_truncates_to_seconds() {
    // Home Assistant / Python clients send 30.0, not 30.
    assert_eq!(parse(json!(30.0)), Some(30));
}

#[test]
fn fractional_float_truncates() {
    assert_eq!(parse(json!(299.9)), Some(299));
}

#[test]
fn sub_second_float_clamps_to_one() {
    // 0.5 must NOT truncate to 0 (which would trigger unload).
    assert_eq!(parse(json!(0.5)), Some(1));
}

#[test]
fn zero_float_unloads() {
    assert_eq!(parse(json!(0.0)), Some(0));
}

#[test]
fn negative_float_normalizes_to_minus_one() {
    assert_eq!(parse(json!(-1.0)), Some(-1));
    assert_eq!(parse(json!(-3600.5)), Some(-1));
}

// ─── Go time.ParseDuration grammar boundaries ──────────────────────────────

#[test]
fn duration_string_zero_alone_is_zero() {
    // Go's ParseDuration special-cases a bare "0" (after the sign) as zero.
    assert_eq!(parse(json!("0")), Some(0));
}

#[test]
fn duration_string_accepts_go_max_range() {
    // 2562047h is the largest whole-hour duration inside Go's int64
    // nanosecond range; one hour more is refused (pinned above).
    assert_eq!(parse(json!("2562047h")), Some(9_223_369_200));
}

#[test]
fn duration_string_fraction_dot_on_one_side() {
    // Go allows a decimal point with digits on only one side: "1.s" is 1s and
    // ".5s" is 500ms (clamped up to 1 so it cannot trigger unload).
    assert_eq!(parse(json!("1.s")), Some(1));
    assert_eq!(parse(json!(".5s")), Some(1));
}

#[test]
fn duration_string_rejects_dot_without_digits() {
    // A "." with no digits on either side of the element is invalid.
    for raw in ["1.", ".s", "1..s", "."] {
        assert!(
            parse_keep_alive_seconds(Some(&json!(raw))).is_err(),
            "{raw:?} must be refused, as Ollama refuses it"
        );
    }
}

#[test]
fn duration_string_rejects_trailing_digits_without_unit() {
    // "1h2" ends with a bare number after its element; Go requires a unit.
    assert!(
        parse_keep_alive_seconds(Some(&json!("1h2"))).is_err(),
        "\"1h2\" must be refused, as Ollama refuses it"
    );
}

#[test]
fn duration_string_compound_fractions() {
    // Fractions combine across elements: 1.5h + 30m, and a fraction-only
    // ".5m" element after a whole one.
    assert_eq!(parse(json!("1.5h30m")), Some(7200));
    assert_eq!(parse(json!("1h.5m")), Some(3630));
}

#[test]
fn duration_string_long_fraction_does_not_overflow() {
    // Go parses any number of fractional digits — its scale is a float64 — and
    // an all-zero fraction is zero. An integer scale must saturate instead of
    // wrapping (which panics here) or reaching zero (which divides by it).
    let raw = format!("0.{}s", "0".repeat(200));
    assert_eq!(parse(json!(raw)), Some(0));
}

#[test]
fn duration_string_rejects_a_second_sign() {
    // Go consumes at most one leading sign and Ollama parses this field with
    // Go's parser; the caller strips the first '-', so a second sign is invalid.
    for raw in ["--1h", "-+1h", "--30m", "-+1.5h", "-+1"] {
        assert!(
            parse_keep_alive_seconds(Some(&json!(raw))).is_err(),
            "{raw:?} must be refused, as Ollama refuses it"
        );
    }
    // The guard must still let a valid single-sign negative duration through.
    assert_eq!(parse(json!("-1h")), Some(-1));
    assert_eq!(parse(json!("-30m")), Some(-1));
}
