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
        reasoning_capability: None,
        size_bytes: None,
        params_string: None,
        display_name: None,
        description: None,
        decision_type: None,
        has_vision_adapter: None,
        has_backend_capabilities: false,
        loaded_flash_attention: None,
        loaded_eval_batch_size: None,
        loaded_parallel: None,
        loaded_offload_kv_cache: None,
        variants: Vec::new(),
        selected_variant: None,
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
    let result =
        ModelResolver::resolve_match("meta-llama-3.1-8b-instruct", &models).expect("should match");
    assert_eq!(result.id, "meta-llama-3.1-8b-instruct");
}

#[test]
fn resolve_match_exact_match_case_insensitive() {
    let models = vec![mi("Meta-Llama-3-8B-Instruct", false)];
    let result =
        ModelResolver::resolve_match("meta-llama-3-8b-instruct", &models).expect("should match");
    assert_eq!(result.id, "Meta-Llama-3-8B-Instruct");
}

// ─── resolve_match: ollama shorthand → lmstudio id ───────────────────────────

#[test]
fn resolve_match_query_with_numeric_tag_stripped_matches() {
    // Simulates the cleaned name "llama3" reaching resolve_match.
    let models = vec![
        mi("llama3-8b-instruct", false),
        mi("llama3-70b-instruct", false),
    ];
    let result = ModelResolver::resolve_match("llama3", &models);
    assert!(
        result.is_some(),
        "should find at least one candidate for 'llama3'"
    );
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
    let models = vec![mi("llama3-8b-instruct", false), mi("llama3-8b-chat", true)];
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
    assert_eq!(
        r1, r2,
        "result must be deterministic regardless of input ordering"
    );
}

// ─── available-models decode: stamped as a decode failure ────────────────────

#[tokio::test]
async fn available_models_decode_failure_is_stamped_as_decode() {
    // A 200 whose body is not JSON is an ANSWERED response, so the retry layer
    // must return it verbatim: without the marker its message ("invalid JSON
    // from /api/v1/models: ...") satisfies the two-word load classifier and the
    // request detours through an explicit load plus the load-timeout sleep.
    let mock = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("GET"))
        .and(wiremock::matchers::path(LM_STUDIO_NATIVE_MODELS))
        .respond_with(wiremock::ResponseTemplate::new(200).set_body_string("<html>not json</html>"))
        .mount(&mock)
        .await;

    let resolver = ModelResolver::new(
        mock.uri(),
        moka::future::Cache::builder().max_capacity(8).build(),
    );
    let err = resolver
        .resolve_model_name(
            "some-model",
            &reqwest::Client::new(),
            CancellationToken::new(),
        )
        .await
        .expect_err("a 200 body that is not JSON must fail decoding");

    assert_eq!(
        err.status_code, 500,
        "a decode fault on a 2xx stays a genuine 500"
    );
    assert!(
        err.message.starts_with("invalid JSON from /api/v1/models"),
        "client-visible decode wording must stay; got {}",
        err.message
    );
    assert!(
        err.is_decode_failure(),
        "the resolver's decode failure must carry the marker the retry layer reads"
    );
}
