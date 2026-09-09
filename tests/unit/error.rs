use crate::error::is_anthropic_surface;

// ── is_anthropic_surface: which paths wear the anthropic envelope ───────────

#[test]
fn surface_predicate_bounds_anthropic_to_the_messages_endpoint() {
    // The predicate answers "is this the anthropic surface?", which means
    // exactly `/v1/messages` and its subpaths — not any same-prefix stranger.
    assert!(is_anthropic_surface("/v1/messages"));
    assert!(is_anthropic_surface("/v1/messages/count_tokens"));
    assert!(is_anthropic_surface("/v1/messages/abc/def"));
    assert!(!is_anthropic_surface("/v1/messagesXYZ"));
    assert!(!is_anthropic_surface("/v1/messages-thing"));
    assert!(!is_anthropic_surface("/v1/chat/completions"));
    assert!(!is_anthropic_surface("/api/chat"));
    assert!(!is_anthropic_surface("/"));
}
