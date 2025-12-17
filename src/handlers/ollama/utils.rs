use bytes::Bytes;
use humantime::parse_duration;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::constants::ERROR_MISSING_MODEL;
use crate::error::ProxyError;
use crate::handlers::RequestContext;
use crate::handlers::transform::extract_system_prompt;
use crate::model::{ModelInfo, clean_model_name};
use crate::server::ModelResolverType;
use crate::storage::{VirtualModelEntry, VirtualModelMetadata};
use crate::streaming::create_ndjson_stream_response;

/// Unified model resolution result containing the resolved LM Studio model ID
/// and merged metadata from virtual model aliases and request parameters.
///
/// Fields are merged with request parameters taking precedence:
/// - `effective_options`: Virtual model parameters merged with request `options`
/// - `effective_format`: Virtual model format merged with request `format`
/// - `system_prompt`: Request system prompt or virtual model system prompt
pub struct ModelResolutionContext {
    pub lm_studio_model_id: String,
    pub effective_options: Option<Value>,
    pub effective_format: Option<Value>,
    pub system_prompt: Option<String>,
}

pub fn extract_model_name<'a>(body: &'a Value, field_name: &str) -> Result<&'a str, ProxyError> {
    body.get(field_name)
        .and_then(|m| m.as_str())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| match field_name {
            "model" => ProxyError::bad_request(ERROR_MISSING_MODEL),
            _ => ProxyError::bad_request("missing required field"),
        })
}

pub async fn resolve_model_target<'a>(
    context: &RequestContext<'a>,
    model_resolver: &ModelResolverType,
    requested_model: &str,
    cancellation_token: CancellationToken,
) -> Result<(String, Option<VirtualModelEntry>), ProxyError> {
    if let Some(entry) = context.virtual_models.get(requested_model).await {
        return Ok((entry.target_model_id.clone(), Some(entry)));
    }

    match model_resolver {
        ModelResolverType::Native(resolver) => resolver
            .resolve_model_name(requested_model, context.client, cancellation_token)
            .await
            .map(|id| (id, None)),
    }
}

pub async fn resolve_model_with_context<'a>(
    context: &RequestContext<'a>,
    model_resolver: &ModelResolverType,
    requested_model: &str,
    request_body: &Value,
    cancellation_token: CancellationToken,
) -> Result<ModelResolutionContext, ProxyError> {
    let (lm_studio_model_id, virtual_entry) =
        resolve_model_target(context, model_resolver, requested_model, cancellation_token).await?;

    let request_options = request_body.get("options");
    let request_format = request_body.get("format");

    let effective_options = merge_option_maps(
        virtual_entry
            .as_ref()
            .and_then(|entry| entry.metadata.parameters.as_ref()),
        request_options,
    );

    let effective_format = virtual_entry
        .as_ref()
        .and_then(|entry| entry.metadata.parameters.as_ref())
        .and_then(|params| params.get("format"))
        .cloned()
        .or_else(|| request_format.cloned());

    let system_from_body = extract_system_prompt(request_body);
    let system_from_virtual = virtual_entry
        .as_ref()
        .and_then(|entry| entry.metadata.system_prompt.clone());
    let system_prompt = system_from_body.or(system_from_virtual);

    Ok(ModelResolutionContext {
        lm_studio_model_id,
        effective_options,
        effective_format,
        system_prompt,
    })
}

pub fn parse_keep_alive_seconds(raw_value: Option<&Value>) -> Result<Option<i64>, ProxyError> {
    let Some(value) = raw_value else {
        return Ok(None);
    };

    match value {
        Value::Null => Ok(None),
        Value::Number(num) => {
            if let Some(signed) = num.as_i64() {
                Ok(Some(signed))
            } else if let Some(unsigned) = num.as_u64() {
                if unsigned <= i64::MAX as u64 {
                    Ok(Some(unsigned as i64))
                } else {
                    Err(ProxyError::bad_request(
                        "keep_alive value exceeds supported range",
                    ))
                }
            } else {
                Err(ProxyError::bad_request("keep_alive must be integral"))
            }
        }
        Value::String(text) => {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                return Ok(None);
            }

            if let Ok(duration) = parse_duration(trimmed) {
                if duration.as_secs() <= i64::MAX as u64 {
                    Ok(Some(duration.as_secs() as i64))
                } else {
                    Err(ProxyError::bad_request(
                        "keep_alive duration exceeds supported range",
                    ))
                }
            } else {
                trimmed.parse::<i64>().map(Some).map_err(|_| {
                    ProxyError::bad_request(
                        "invalid keep_alive value. Use numeric seconds or durations like '5m'",
                    )
                })
            }
        }
        _ => Err(ProxyError::bad_request(
            "invalid keep_alive value. Use numeric seconds or durations like '5m'",
        )),
    }
}

pub fn apply_keep_alive_ttl(target: &mut Value, keep_alive_seconds: Option<i64>) {
    if let Some(ttl) = keep_alive_seconds
        && let Some(obj) = target.as_object_mut()
    {
        obj.insert("ttl".to_string(), Value::from(ttl));
    }
}

pub fn keep_alive_requests_unload(ttl: Option<i64>) -> bool {
    matches!(ttl, Some(value) if value == 0)
}

pub fn merge_option_maps(base: Option<&Value>, overrides: Option<&Value>) -> Option<Value> {
    match (base, overrides) {
        (None, None) => None,
        (Some(b), None) => Some(b.clone()),
        (None, Some(o)) => Some(o.clone()),
        (Some(b), Some(o)) => match (b.as_object(), o.as_object()) {
            (Some(base_obj), Some(override_obj)) => {
                let mut combined = serde_json::Map::new();
                for (k, v) in base_obj {
                    combined.insert(k.clone(), v.clone());
                }
                for (k, v) in override_obj {
                    combined.insert(k.clone(), v.clone());
                }
                Some(Value::Object(combined))
            }
            _ => Some(o.clone()),
        },
    }
}

pub fn normalize_chat_messages(messages: &[Value], system_prompt: Option<&str>) -> Value {
    if let Some(system_text) = system_prompt {
        let already_has_system = messages.iter().any(|message| {
            message
                .get("role")
                .and_then(|role| role.as_str())
                .map(|role| role.eq_ignore_ascii_case("system"))
                .unwrap_or(false)
        });

        if already_has_system {
            json!(messages)
        } else {
            let mut combined = Vec::with_capacity(messages.len() + 1);
            combined.push(json!({
                "role": "system",
                "content": system_text,
            }));
            combined.extend(messages.iter().cloned());
            Value::Array(combined)
        }
    } else {
        json!(messages)
    }
}

pub fn build_virtual_metadata(
    body: &Value,
    base: Option<VirtualModelMetadata>,
) -> VirtualModelMetadata {
    let mut metadata = base.unwrap_or_default();

    if let Some(system_prompt) = body.get("system").and_then(|v| v.as_str()) {
        metadata.system_prompt = Some(system_prompt.to_string());
    }

    if let Some(template) = body.get("template").and_then(|v| v.as_str()) {
        metadata.template = Some(template.to_string());
    }

    if let Some(parameters) = body.get("parameters") {
        metadata.parameters = Some(parameters.clone());
    }

    if let Some(license) = body.get("license") {
        metadata.license = Some(license.clone());
    }

    if let Some(adapters) = body.get("adapters") {
        metadata.adapters = Some(adapters.clone());
    }

    if let Some(messages) = body.get("messages").and_then(|m| m.as_array()).cloned() {
        metadata.messages = Some(messages);
    }

    metadata
}

pub fn build_virtual_model_response(entry: &VirtualModelEntry) -> Value {
    json!({
        "status": "success",
        "model": entry.name,
        "virtual": true,
        "source_model": entry.source_model,
        "target_model_id": entry.target_model_id,
        "created_at": entry.created_at.to_rfc3339(),
        "updated_at": entry.updated_at.to_rfc3339(),
    })
}

pub fn stream_status_messages(
    chunks: Vec<Value>,
    error_label: &str,
) -> Result<warp::reply::Response, ProxyError> {
    let (tx, rx) = mpsc::unbounded_channel();
    for chunk in chunks {
        if !send_status_chunk(&tx, &chunk) {
            break;
        }
    }
    drop(tx);
    create_ndjson_stream_response(rx, error_label)
}

pub fn send_status_chunk(
    tx: &mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
    chunk: &Value,
) -> bool {
    match serde_json::to_string(chunk) {
        Ok(serialized) => tx
            .send(Ok(Bytes::from(format!("{}\n", serialized))))
            .is_ok(),
        Err(e) => {
            log::warn!("pull chunk: serialization failed: {}", e);
            false
        }
    }
}

pub fn send_status_error_chunk(
    tx: &mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
    model: &str,
    message: &str,
) {
    let chunk = json!({
        "status": "failed",
        "model": model,
        "error": message
    });
    let _ = send_status_chunk(tx, &chunk);
}

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

pub async fn fetch_model_info_for_id(
    context: &RequestContext<'_>,
    model_resolver: &ModelResolverType,
    target_model_id: &str,
    cancellation_token: CancellationToken,
) -> Result<Option<ModelInfo>, ProxyError> {
    match model_resolver {
        ModelResolverType::Native(resolver) => {
            let models = resolver
                .get_all_models(context.client, cancellation_token)
                .await?;
            Ok(models.into_iter().find(|model| model.id == target_model_id))
        }
    }
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
    cancellation_token: CancellationToken,
) -> Result<String, ProxyError> {
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

        if let Some(model_info) = fetch_model_info_for_id(
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

pub async fn create_virtual_model_alias(
    context: &RequestContext<'_>,
    model_resolver: &ModelResolverType,
    alias_name: &str,
    source_name: &str,
    body: &Value,
    cancellation_token: CancellationToken,
) -> Result<VirtualModelEntry, ProxyError> {
    if let Some(files) = body.get("files") {
        let has_content = match files {
            Value::Object(map) => !map.is_empty(),
            Value::Array(arr) => !arr.is_empty(),
            Value::Null => false,
            _ => true,
        };
        if has_content {
            return Err(ProxyError::not_implemented(
                "creating models from custom blobs is not supported via LM Studio proxy",
            ));
        }
    }

    if body.get("quantize").is_some() {
        return Err(ProxyError::not_implemented(
            "quantization is not available via LM Studio proxy",
        ));
    }

    let (resolved_id, source_virtual_entry) =
        resolve_model_target(context, model_resolver, source_name, cancellation_token).await?;

    let base_metadata = source_virtual_entry.map(|entry| entry.metadata);
    let metadata = build_virtual_metadata(body, base_metadata);

    context
        .virtual_models
        .create_alias(alias_name, source_name.to_string(), resolved_id, metadata)
        .await
}
