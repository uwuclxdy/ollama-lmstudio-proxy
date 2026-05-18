//! /api/tags `modified_at` must be stable across calls within a single proxy run.
//!
//! Per Ollama spec, `modified_at` is the on-disk modification time of the model.
//! LM Studio does not expose that, but emitting a fresh timestamp every call
//! defeats clients that use the field for cache invalidation. The proxy must
//! therefore return a process-stable value.

#[path = "../src/model/timestamps.rs"]
#[allow(dead_code)]
mod timestamps;

use timestamps::process_start_modified_at;

#[test]
fn modified_at_is_stable_across_calls() {
    let first = process_start_modified_at();
    std::thread::sleep(std::time::Duration::from_millis(15));
    let second = process_start_modified_at();
    assert_eq!(
        first, second,
        "modified_at must be process-stable; got first={first}, second={second}"
    );
}

#[test]
fn modified_at_is_rfc3339() {
    let ts = process_start_modified_at();
    // RFC3339 requires a `T` separator and a timezone suffix.
    assert!(ts.contains('T'), "expected RFC3339 timestamp, got {ts}");
    assert!(
        ts.ends_with('Z') || ts.contains('+') || ts.matches('-').count() >= 3,
        "expected RFC3339 timezone suffix, got {ts}"
    );
}
