#[test]
fn modified_at_fallback_is_deterministic() {
    let first = super::model_modified_at_fallback();
    std::thread::sleep(std::time::Duration::from_millis(15));
    let second = super::model_modified_at_fallback();
    assert_eq!(
        first, second,
        "modified_at fallback must be stable; got first={first}, second={second}"
    );
}

#[test]
fn modified_at_fallback_is_rfc3339_epoch() {
    let ts = super::model_modified_at_fallback();
    assert_eq!(ts, "1970-01-01T00:00:00Z");
    // RFC3339 requires a `T` separator and a timezone suffix.
    assert!(ts.contains('T'), "expected RFC3339 timestamp, got {ts}");
    assert!(
        ts.ends_with('Z') || ts.contains('+') || ts.matches('-').count() >= 3,
        "expected RFC3339 timezone suffix, got {ts}"
    );
}
