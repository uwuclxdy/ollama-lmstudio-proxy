use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::error::ProxyError;
use crate::handlers::RequestContext;
use crate::handlers::ollama::utils::extract_system_prompt;
use crate::model::ModelInfo;
use crate::server::ModelResolverType;
use crate::storage::VirtualModelEntry;

pub struct ModelResolutionContext {
    pub lm_studio_model_id: String,
    pub effective_options: Option<Value>,
    pub effective_format: Option<Value>,
    pub system_prompt: Option<String>,
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

fn merge_option_maps(base: Option<&Value>, overrides: Option<&Value>) -> Option<Value> {
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
