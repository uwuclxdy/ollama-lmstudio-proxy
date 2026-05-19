use super::*;
// handlers/ollama/model_resolution.rs
//
// Public surface:
//   - ModelResolutionContext (struct with pub fields)
//   - resolve_model_target   — async, needs RequestContext + ModelResolverType (network)
//   - resolve_model_with_context — async, network
//   - fetch_model_info_for_id    — async, network
//
// The private helper merge_option_maps drives the effective_options/format
// merging logic inside resolve_model_with_context.  We cannot call it directly
// but we can verify ModelResolutionContext can be constructed and its fields
// are accessible, and we document the merge semantics via the pub struct.

#[test]
fn model_resolution_context_fields_accessible() {
    use serde_json::json;

    let ctx = ModelResolutionContext {
        lm_studio_model_id: "llama-3-8b".to_string(),
        effective_options: Some(json!({"temperature": 0.5})),
        effective_format: None,
        system_prompt: Some("be helpful".to_string()),
    };

    assert_eq!(ctx.lm_studio_model_id, "llama-3-8b");
    assert!(ctx.effective_options.is_some());
    assert!(ctx.effective_format.is_none());
    assert_eq!(ctx.system_prompt.as_deref(), Some("be helpful"));
}

#[test]
fn model_resolution_context_all_none() {
    let ctx = ModelResolutionContext {
        lm_studio_model_id: String::new(),
        effective_options: None,
        effective_format: None,
        system_prompt: None,
    };

    assert!(ctx.lm_studio_model_id.is_empty());
    assert!(ctx.effective_options.is_none());
    assert!(ctx.system_prompt.is_none());
}
