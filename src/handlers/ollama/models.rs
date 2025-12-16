use std::time::Instant;

use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::constants::LOG_PREFIX_SUCCESS;
use crate::error::ProxyError;
use crate::handlers::RequestContext;
use crate::handlers::retry::trigger_model_loading_for_ollama;
use crate::http::json_response;
use crate::logging::{LogConfig, log_request, log_timed};
use crate::server::ModelResolverType;

use super::utils::{
    extract_model_name, keep_alive_requests_unload, parse_keep_alive_seconds, resolve_model_target,
};

pub async fn handle_ollama_show(
    context: RequestContext<'_>,
    model_resolver: ModelResolverType,
    body: Value,
    cancellation_token: CancellationToken,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();
    if LogConfig::get().debug_enabled {
        log::debug!(
            "show request: {}",
            serde_json::to_string_pretty(&body).unwrap_or_default()
        );
    }

    let ollama_model_name = extract_model_name(&body, "model")?;
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

    let models = match &model_resolver {
        ModelResolverType::Native(resolver) => {
            resolver
                .get_all_models(context.client, cancellation_token)
                .await?
        }
    };

    let base_model = models.iter().find(|m| m.id == resolved_id);

    let mut response = if let Some(model) = base_model {
        model.to_show_response()
    } else {
        json!({
            "error": format!("Model '{}' not found in LM Studio", resolved_id),
            "requested": ollama_model_name,
            "resolved_to": resolved_id
        })
    };

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
    }

    log_timed(LOG_PREFIX_SUCCESS, "Ollama show", start_time);

    if LogConfig::get().debug_enabled {
        log::debug!(
            "show response: {}",
            serde_json::to_string_pretty(&response).unwrap_or_default()
        );
    }

    Ok(json_response(&response))
}

pub async fn handle_ollama_ps(
    context: RequestContext<'_>,
    model_resolver: ModelResolverType,
    cancellation_token: CancellationToken,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();

    let loaded_models = match &model_resolver {
        ModelResolverType::Native(resolver) => {
            resolver
                .get_loaded_models(context.client, cancellation_token)
                .await?
        }
    };

    let virtual_entries = context.virtual_models.list().await;

    let mut ollama_models: Vec<_> = loaded_models
        .iter()
        .map(|m| m.to_ollama_ps_model())
        .collect();

    for entry in &virtual_entries {
        if loaded_models.iter().any(|m| m.id == entry.target_model_id)
            && let Some(base_model) = loaded_models.iter().find(|m| m.id == entry.target_model_id)
        {
            let aliased = base_model.with_alias_name(&entry.name);
            ollama_models.push(aliased.to_ollama_ps_model());
        }
    }

    let response = json!({ "models": ollama_models });

    log_timed(LOG_PREFIX_SUCCESS, "Ollama ps", start_time);

    if LogConfig::get().debug_enabled {
        log::debug!(
            "ps response: {}",
            serde_json::to_string_pretty(&response).unwrap_or_default()
        );
    }

    Ok(json_response(&response))
}
