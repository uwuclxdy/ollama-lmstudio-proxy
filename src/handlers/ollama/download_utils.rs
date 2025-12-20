use crate::handlers::RequestContext;
use crate::model::clean_model_name;
use crate::server::ModelResolverType;
use crate::storage::VirtualModelEntry;

pub fn looks_like_remote_identifier(identifier: &str) -> bool {
    let lowered = identifier.to_ascii_lowercase();
    lowered.starts_with("http://")
        || lowered.starts_with("https://")
        || lowered.starts_with("hf://")
        || lowered.starts_with("s3://")
        || lowered.starts_with("gs://")
}

pub fn extract_virtual_download_source(entry: &VirtualModelEntry) -> Option<String> {
    entry
        .metadata
        .parameters
        .as_ref()
        .and_then(|params| params.get("download_source"))
        .and_then(|value| value.as_str())
        .map(|value| value.to_string())
}

pub fn publisher_prefers_hf_link(publisher: &str) -> bool {
    matches!(
        publisher.to_ascii_lowercase().as_str(),
        "lmstudio-community" | "huggingface"
    )
}

pub fn build_hf_download_url(publisher: &str, model_id: &str) -> String {
    format!(
        "https://huggingface.co/{}/{}",
        publisher.trim().trim_end_matches('/'),
        model_id.trim_start_matches('/')
    )
}

pub fn build_catalog_identifier(publisher: &str, model_id: &str) -> Option<String> {
    let trimmed = publisher.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(format!(
        "{}/{}",
        trimmed.trim_end_matches('/'),
        model_id.trim_start_matches('/')
    ))
}

pub async fn determine_download_identifier(
    context: &RequestContext<'_>,
    model_resolver: &ModelResolverType,
    requested_model: &str,
    source_override: Option<&str>,
    resolved_model: Option<(String, Option<VirtualModelEntry>)>,
    cancellation_token: tokio_util::sync::CancellationToken,
) -> Result<String, crate::error::ProxyError> {
    if let Some(source) = source_override {
        return Ok(source.to_string());
    }

    if looks_like_remote_identifier(requested_model) {
        return Ok(requested_model.to_string());
    }

    if let Some((resolved_model_id, virtual_entry)) = resolved_model {
        if let Some(source) = virtual_entry
            .as_ref()
            .and_then(extract_virtual_download_source)
        {
            return Ok(source);
        }

        if looks_like_remote_identifier(&resolved_model_id) {
            return Ok(resolved_model_id);
        }

        if resolved_model_id.contains('/') && !resolved_model_id.contains(' ') {
            return Ok(resolved_model_id);
        }

        if let Some(model_info) =
            crate::handlers::ollama::model_resolution::fetch_model_info_for_id(
                context,
                model_resolver,
                &resolved_model_id,
                cancellation_token,
            )
            .await?
        {
            let cleaned_id = clean_model_name(&model_info.id).to_string();
            if publisher_prefers_hf_link(&model_info.publisher) {
                return Ok(build_hf_download_url(&model_info.publisher, &cleaned_id));
            }

            if let Some(identifier) = build_catalog_identifier(&model_info.publisher, &cleaned_id) {
                return Ok(identifier);
            }
        }

        return Ok(resolved_model_id);
    }

    Ok(requested_model.to_string())
}
