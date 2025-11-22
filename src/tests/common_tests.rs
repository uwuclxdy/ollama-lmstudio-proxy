use crate::common::*;
use serde_json::json;

#[test]
fn prefers_explicit_max_tokens_over_num_predict() {
    let options = json!({
        "num_predict": 256,
        "max_tokens": 42
    });

    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(params.get("max_tokens"), Some(&json!(42)));
}

#[test]
fn forwards_logit_bias_when_present() {
    let options = json!({
        "logit_bias": {"123": -10},
        "temperature": 0.2
    });

    let params = map_ollama_to_lmstudio_params(Some(&options), None);
    assert_eq!(params.get("logit_bias"), Some(&json!({"123": -10})));
}

#[test]
fn converts_schema_format_into_response_format() {
    let schema = json!({"type": "object", "properties": {"answer": {"type": "string"}}});
    let params = map_ollama_to_lmstudio_params(None, Some(&schema));

    let response_format = params
        .get("response_format")
        .expect("missing response_format");
    assert_eq!(
        response_format.get("type").and_then(|v| v.as_str()),
        Some("json_schema")
    );
    assert_eq!(response_format["json_schema"]["schema"], schema);
}

#[test]
fn maps_json_mode_string_to_json_object_response_format() {
    let params =
        map_ollama_to_lmstudio_params(None, Some(&serde_json::Value::String("json".into())));
    assert_eq!(
        params.get("response_format"),
        Some(&json!({"type": "json_object"}))
    );
}
