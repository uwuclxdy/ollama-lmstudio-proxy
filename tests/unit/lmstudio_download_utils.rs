use super::*;
use chrono::Utc;
use serde_json::json;

fn make_virtual_entry(parameters: Option<serde_json::Value>) -> VirtualModelEntry {
    VirtualModelEntry {
        name: "alias".to_string(),
        source_model: "llama3".to_string(),
        target_model_id: "llama-3-8b".to_string(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        metadata: crate::storage::virtual_models::VirtualModelMetadata {
            parameters,
            ..Default::default()
        },
    }
}

// --- looks_like_remote_identifier ---

#[test]
fn http_url_is_remote() {
    assert!(looks_like_remote_identifier(
        "http://example.com/model.gguf"
    ));
}

#[test]
fn https_url_is_remote() {
    assert!(looks_like_remote_identifier(
        "https://huggingface.co/org/model"
    ));
}

#[test]
fn hf_protocol_is_remote() {
    assert!(looks_like_remote_identifier("hf://org/model"));
}

#[test]
fn s3_protocol_is_remote() {
    assert!(looks_like_remote_identifier("s3://bucket/model.gguf"));
}

#[test]
fn gs_protocol_is_remote() {
    assert!(looks_like_remote_identifier("gs://bucket/model.gguf"));
}

#[test]
fn plain_model_name_is_not_remote() {
    assert!(!looks_like_remote_identifier("llama3"));
}

#[test]
fn model_with_tag_is_not_remote() {
    assert!(!looks_like_remote_identifier("llama3:latest"));
}

#[test]
fn slash_separated_model_is_not_remote() {
    assert!(!looks_like_remote_identifier("org/model:tag"));
}

#[test]
fn case_insensitive_http_scheme() {
    assert!(looks_like_remote_identifier("HTTP://example.com/model"));
    assert!(looks_like_remote_identifier("HTTPS://example.com/model"));
    assert!(looks_like_remote_identifier("HF://org/model"));
}

// --- extract_virtual_download_source ---

#[test]
fn extract_download_source_present() {
    let entry = make_virtual_entry(Some(json!({
        "download_source": "https://huggingface.co/org/model"
    })));
    let source = extract_virtual_download_source(&entry);
    assert_eq!(source.as_deref(), Some("https://huggingface.co/org/model"));
}

#[test]
fn extract_download_source_absent() {
    let entry = make_virtual_entry(Some(json!({"temperature": 0.7})));
    assert!(extract_virtual_download_source(&entry).is_none());
}

#[test]
fn extract_download_source_no_parameters() {
    let entry = make_virtual_entry(None);
    assert!(extract_virtual_download_source(&entry).is_none());
}

#[test]
fn extract_download_source_non_string_value_returns_none() {
    let entry = make_virtual_entry(Some(json!({"download_source": 42})));
    assert!(extract_virtual_download_source(&entry).is_none());
}

// --- publisher_prefers_hf_link ---

#[test]
fn lmstudio_community_prefers_hf() {
    assert!(publisher_prefers_hf_link("lmstudio-community"));
}

#[test]
fn huggingface_prefers_hf() {
    assert!(publisher_prefers_hf_link("huggingface"));
}

#[test]
fn case_insensitive_lmstudio_community() {
    assert!(publisher_prefers_hf_link("LMStudio-Community"));
    assert!(publisher_prefers_hf_link("LMSTUDIO-COMMUNITY"));
}

#[test]
fn arbitrary_publisher_does_not_prefer_hf() {
    assert!(!publisher_prefers_hf_link("meta-llama"));
    assert!(!publisher_prefers_hf_link("mistralai"));
    assert!(!publisher_prefers_hf_link(""));
}

// --- build_hf_download_url ---

#[test]
fn hf_url_basic() {
    let url = build_hf_download_url("org", "model");
    assert_eq!(url, "https://huggingface.co/org/model");
}

#[test]
fn hf_url_trims_trailing_slash_from_publisher() {
    let url = build_hf_download_url("org/", "model");
    assert_eq!(url, "https://huggingface.co/org/model");
}

#[test]
fn hf_url_trims_leading_slash_from_model_id() {
    let url = build_hf_download_url("org", "/model");
    assert_eq!(url, "https://huggingface.co/org/model");
}

#[test]
fn hf_url_trims_both_sides() {
    let url = build_hf_download_url("org/", "/model-name");
    assert_eq!(url, "https://huggingface.co/org/model-name");
}

#[test]
fn hf_url_with_variant() {
    let url = build_hf_download_url("lmstudio-community", "Meta-Llama-3-8B-Instruct-GGUF");
    assert_eq!(
        url,
        "https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF"
    );
}

// --- build_catalog_identifier ---

#[test]
fn catalog_id_basic() {
    let id = build_catalog_identifier("org", "model");
    assert_eq!(id, Some("org/model".to_string()));
}

#[test]
fn catalog_id_empty_publisher_returns_none() {
    assert!(build_catalog_identifier("", "model").is_none());
    assert!(build_catalog_identifier("   ", "model").is_none());
}

#[test]
fn catalog_id_trims_trailing_slash_from_publisher() {
    let id = build_catalog_identifier("org/", "model");
    assert_eq!(id, Some("org/model".to_string()));
}

#[test]
fn catalog_id_trims_leading_slash_from_model() {
    let id = build_catalog_identifier("org", "/model");
    assert_eq!(id, Some("org/model".to_string()));
}
