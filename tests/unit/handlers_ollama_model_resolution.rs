// handlers/ollama/model_resolution.rs
//
// Public async functions (resolve_model_target, resolve_model_with_context,
// fetch_model_info_for_id) all require live network access (ModelResolver +
// RequestContext). Their behaviour is covered by the integration test suite.
// The private merge_option_maps helper is exercised indirectly through those
// integration paths.
