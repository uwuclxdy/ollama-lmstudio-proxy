use std::time::Instant;

use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::api::RequestContext;
use crate::api::retry::trigger_model_loading_for_ollama;
use crate::constants::LOG_PREFIX_SUCCESS;
use crate::error::ProxyError;
use crate::http::json_response;
use crate::logging::{LogConfig, log_request, log_timed};
use crate::model::ModelResolver;
use std::sync::Arc;

use super::resolution::resolve_model_target;
use crate::lmstudio::keep_alive::{keep_alive_requests_unload, parse_keep_alive_seconds};
use crate::logging::log_handler_io;
use crate::model::naming::extract_required_model_name;
use crate::model::types::ModelInfo;

pub async fn handle_ollama_show(
    context: RequestContext<'_>,
    model_resolver: Arc<ModelResolver>,
    body: Value,
    cancellation_token: CancellationToken,
) -> Result<axum::response::Response, ProxyError> {
    let start_time = Instant::now();
    if LogConfig::get().debug_enabled {
        log::debug!(
            "show request: {}",
            serde_json::to_string_pretty(&body).unwrap_or_default()
        );
    }

    let ollama_model_name = extract_required_model_name(&body)?;
    let keep_alive_seconds = parse_keep_alive_seconds(body.get("keep_alive"))?;

    log_request("POST", "/api/show", Some(ollama_model_name));

    if !keep_alive_requests_unload(keep_alive_seconds) {
        trigger_model_loading_for_ollama(&context, ollama_model_name, cancellation_token.clone())
            .await?;
    }

    let (resolved_id, virtual_entry) = resolve_model_target(
        &context,
        &model_resolver,
        ollama_model_name,
        cancellation_token.clone(),
    )
    .await?;

    let models = model_resolver
        .get_all_models(context.client, cancellation_token)
        .await?;

    let verbose = body
        .get("verbose")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let base_model = models.iter().find(|m| m.id == resolved_id);

    let Some(model) = base_model else {
        return Err(ProxyError::not_found(&format!(
            "model '{}' not found",
            ollama_model_name
        )));
    };

    let mut response = model.to_show_response_verbose(verbose);

    if let Some(entry) = virtual_entry
        && let Some(obj) = response.as_object_mut()
    {
        obj.insert("virtual".to_string(), json!(true));
        obj.insert("alias_name".to_string(), json!(entry.name));
        obj.insert("source_model".to_string(), json!(entry.source_model));
        obj.insert("target_model_id".to_string(), json!(entry.target_model_id));

        if let Some(system) = &entry.metadata.system_prompt {
            obj.insert("system".to_string(), json!(system));
        }
        if let Some(template) = &entry.metadata.template {
            obj.insert("template".to_string(), json!(template));
        }
        if let Some(license) = &entry.metadata.license {
            obj.insert("license".to_string(), license.clone());
        }
    }

    log_timed(LOG_PREFIX_SUCCESS, "Ollama show", start_time);
    log_handler_io("show", None, Some(&response));
    Ok(json_response(&response))
}

pub async fn handle_ollama_ps(
    context: RequestContext<'_>,
    model_resolver: Arc<ModelResolver>,
    cancellation_token: CancellationToken,
) -> Result<axum::response::Response, ProxyError> {
    let start_time = Instant::now();

    let loaded_models = model_resolver
        .get_loaded_models(context.client, cancellation_token)
        .await?;

    let virtual_entries = context.virtual_models.list().await;
    let loaded_virtuals: Vec<_> = virtual_entries
        .into_iter()
        .filter(|entry| loaded_models.iter().any(|m| m.id == entry.target_model_id))
        .collect();

    let ollama_models = ModelInfo::merge_with_virtuals(&loaded_models, &loaded_virtuals, |m| {
        m.to_ollama_ps_model()
    });

    let response = json!({ "models": ollama_models });
    log_timed(LOG_PREFIX_SUCCESS, "Ollama ps", start_time);
    log_handler_io("ps", None, Some(&response));
    Ok(json_response(&response))
}

pub async fn handle_ollama_tags(
    context: RequestContext<'_>,
    model_resolver: Arc<ModelResolver>,
    cancellation_token: CancellationToken,
) -> Result<axum::response::Response, ProxyError> {
    let start_time = Instant::now();

    let models = model_resolver
        .get_all_models(context.client, cancellation_token)
        .await?;

    let virtual_entries = context.virtual_models.list().await;
    let mut ollama_models =
        ModelInfo::merge_with_virtuals(&models, &virtual_entries, |m| m.to_ollama_tags_model());

    for entry in &virtual_entries {
        if models.iter().all(|m| m.id != entry.target_model_id) {
            ollama_models.push(json!({
                "name": entry.name,
                "model": entry.name,
                "modified_at": entry.updated_at.to_rfc3339(),
                "size": 0,
                "digest": format!("{:x}", md5::compute(entry.name.as_bytes())),
                "details": {
                    "parent_model": entry.source_model,
                    "format": "virtual",
                    "family": "unknown",
                    "families": ["unknown"],
                    "parameter_size": "unknown",
                    "quantization_level": "unknown"
                }
            }));
        }
    }

    let response = json!({ "models": ollama_models });
    log_timed(LOG_PREFIX_SUCCESS, "Ollama tags", start_time);
    log_handler_io("tags", None, Some(&response));
    Ok(json_response(&response))
}
