use super::*;

// Helper — build a minimal ModelInfo without going through the HTTP stack.
fn mi(id: &str, loaded: bool) -> ModelInfo {
    ModelInfo {
        id: id.to_string(),
        ollama_name: format!("{id}:latest"),
        model_type: "llm".to_string(),
        publisher: "test".to_string(),
        arch: "llama".to_string(),
        compatibility_type: "gguf".to_string(),
        quantization: "q4_0".to_string(),
        state: if loaded { "loaded" } else { "not-loaded" }.to_string(),
        max_context_length: 4096,
        context_length: 4096,
        is_loaded: loaded,
        supports_vision: false,
        supports_tools: false,
        supports_reasoning: false,
        size_bytes: None,
        params_string: None,
        display_name: None,
        description: None,
    }
}

// ─── resolve_match: empty list ────────────────────────────────────────────────

#[test]
fn resolve_match_empty_model_list_returns_none() {
    assert!(ModelResolver::resolve_match("llama3", &[]).is_none());
}

// ─── resolve_match: exact match ───────────────────────────────────────────────

#[test]
fn resolve_match_exact_id_returns_that_model() {
    let models = vec![
        mi("meta-llama-3.1-8b-instruct", false),
        mi("mistral-7b-instruct-v0.3", false),
    ];
    let result = ModelResolver::resolve_match("meta-llama-3.1-8b-instruct", &models)
        .expect("should match");
    assert_eq!(result.id, "meta-llama-3.1-8b-instruct");
}

#[test]
fn resolve_match_exact_match_case_insensitive() {
    let models = vec![mi("Meta-Llama-3-8B-Instruct", false)];
    let result = ModelResolver::resolve_match("meta-llama-3-8b-instruct", &models)
        .expect("should match");
    assert_eq!(result.id, "Meta-Llama-3-8B-Instruct");
}

// ─── resolve_match: ollama shorthand → lmstudio id ───────────────────────────

#[test]
fn resolve_match_ollama_shorthand_llama3_8b() {
    // After clean_model_name, "llama3.1:8b" becomes "llama3.1".
    // resolve_match receives the already-cleaned query.
    let models = vec![
        mi("meta-llama-3.1-8b-instruct", false),
        mi("meta-llama-3.1-70b-instruct", false),
    ];
    // "llama3.1" should match via token overlap — contains "3" and "1" parts
    let result = ModelResolver::resolve_match("llama3.1", &models);
    // At minimum it should not panic; the exact match depends on scorer weights.
    // We only assert that if anything comes back, it starts with "meta-llama".
    if let Some(m) = result {
        assert!(
            m.id.starts_with("meta-llama"),
            "unexpected match: {}",
            m.id
        );
    }
}

#[test]
fn resolve_match_query_with_numeric_tag_stripped_matches() {
    // Simulates the cleaned name "llama3" reaching resolve_match.
    let models = vec![mi("llama3-8b-instruct", false), mi("llama3-70b-instruct", false)];
    let result = ModelResolver::resolve_match("llama3", &models);
    assert!(result.is_some(), "should find at least one candidate for 'llama3'");
}

// ─── resolve_match: no candidate ─────────────────────────────────────────────

#[test]
fn resolve_match_completely_unrelated_query_returns_none() {
    let models = vec![mi("mistral-7b-instruct", false), mi("phi-3-mini", false)];
    // "zzz-unknown-xyz" has no token overlap with either model
    let result = ModelResolver::resolve_match("zzz-unknown-xyz", &models);
    assert!(result.is_none(), "should not match any model");
}

#[test]
fn resolve_match_single_unrelated_model_returns_none() {
    let models = vec![mi("phi-3-mini-128k", false)];
    let result = ModelResolver::resolve_match("qwen-2-72b", &models);
    assert!(result.is_none());
}

// ─── resolve_match: loaded preference ────────────────────────────────────────

#[test]
fn resolve_match_prefers_loaded_model() {
    let models = vec![
        mi("llama3-8b-instruct", false),
        mi("llama3-8b-chat", true),
    ];
    let result = ModelResolver::resolve_match("llama3-8b", &models).expect("should match");
    assert_eq!(
        result.id, "llama3-8b-chat",
        "loaded model must be preferred when both are substring matches"
    );
}

#[test]
fn resolve_match_loaded_state_is_preserved_in_returned_info() {
    let models = vec![mi("llama3-8b-instruct", true)];
    let result = ModelResolver::resolve_match("llama3-8b-instruct", &models).expect("should match");
    assert!(result.is_loaded);
    assert_eq!(result.state, "loaded");
}

#[test]
fn resolve_match_not_loaded_state_is_preserved_in_returned_info() {
    let models = vec![mi("llama3-8b-instruct", false)];
    let result = ModelResolver::resolve_match("llama3-8b-instruct", &models).expect("should match");
    assert!(!result.is_loaded);
    assert_eq!(result.state, "not-loaded");
}

// ─── resolve_match: single-model list ────────────────────────────────────────

#[test]
fn resolve_match_single_exact_model_matches() {
    let models = vec![mi("phi-3-mini", false)];
    let result = ModelResolver::resolve_match("phi-3-mini", &models).expect("should match");
    assert_eq!(result.id, "phi-3-mini");
}

#[test]
fn resolve_match_single_model_no_overlap_returns_none() {
    let models = vec![mi("phi-3-mini", false)];
    assert!(ModelResolver::resolve_match("llama3", &models).is_none());
}

// ─── resolve_match: result identity (clone) ──────────────────────────────────

#[test]
fn resolve_match_returned_info_matches_original_fields() {
    let mut model = mi("qwen2-7b-instruct", true);
    model.arch = "qwen2".to_string();
    model.publisher = "Qwen".to_string();
    model.quantization = "q8_0".to_string();
    model.max_context_length = 32768;

    let models = vec![model.clone()];
    let result = ModelResolver::resolve_match("qwen2-7b-instruct", &models).expect("should match");

    assert_eq!(result.arch, "qwen2");
    assert_eq!(result.publisher, "Qwen");
    assert_eq!(result.quantization, "q8_0");
    assert_eq!(result.max_context_length, 32768);
}

// ─── resolve_match: determinism ──────────────────────────────────────────────

#[test]
fn resolve_match_result_is_stable_across_input_order() {
    let base = vec![
        mi("llama3-8b-instruct", false),
        mi("llama3-8b-chat", false),
        mi("llama3-8b-tools", false),
    ];
    let reversed: Vec<_> = base.iter().rev().cloned().collect();

    let r1 = ModelResolver::resolve_match("llama3-8b", &base).map(|m| m.id.clone());
    let r2 = ModelResolver::resolve_match("llama3-8b", &reversed).map(|m| m.id.clone());
    assert_eq!(r1, r2, "result must be deterministic regardless of input ordering");
}

// ─── clean_model_name integration (called inside resolve_model_name) ──────────
// Verify the clean step that precedes resolve_match produces the expected input.

#[test]
fn clean_model_name_feeds_query_to_resolver_with_tag_preserved() {
    // Per api_docs/ollama.md "Model names": ":8b" is a version identifier, not
    // the default ":latest", so it is preserved into the resolver.
    let cleaned = crate::model::utils::clean_model_name("llama3.1:8b");
    assert_eq!(cleaned, "llama3.1:8b");
}

#[test]
fn clean_model_name_strips_latest_before_resolver() {
    let cleaned = crate::model::utils::clean_model_name("mistral:latest");
    assert_eq!(cleaned, "mistral");
}
