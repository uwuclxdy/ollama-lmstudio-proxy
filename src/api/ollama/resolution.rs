use std::sync::Arc;

use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::api::RequestContext;
use crate::error::ProxyError;
use crate::lmstudio::request::TopLevelParams;
use crate::model::ModelInfo;
use crate::model::ModelResolver;
use crate::storage::VirtualModelEntry;

/// Pull the three Ollama top-level forwarded keys (`think`, `logprobs`,
/// `top_logprobs`) out of a request body. Both /api/chat and /api/generate
/// forward these to LM Studio (`reasoning`, `logprobs`, `top_logprobs`).
///
/// `reasoning_effort` (OpenAI alias) is accepted as a fallback when `think` is
/// absent. When both are present, `think` takes precedence.
pub fn make_top_level_params(body: &Value) -> TopLevelParams<'_> {
    TopLevelParams {
        think: body.get("think").or_else(|| body.get("reasoning_effort")),
        logprobs: body.get("logprobs"),
        top_logprobs: body.get("top_logprobs"),
        // Caller fills this from the resolved model's capability after building
        // the params (the body alone can't say whether the model reasons).
        model_is_thinking: false,
    }
}

pub fn extract_system_prompt(body: &Value) -> Option<String> {
    body.get("system")
        .and_then(|value| value.as_str())
        .map(|s| s.to_string())
        .or_else(|| {
            body.get("options")
                .and_then(|opts| opts.get("system"))
                .and_then(|value| value.as_str())
                .map(|s| s.to_string())
        })
}

pub struct ModelResolutionContext {
    pub lm_studio_model_id: String,
    pub effective_options: Option<Value>,
    pub effective_format: Option<Value>,
    pub system_prompt: Option<String>,
    /// Whether the resolved model is reasoning-capable (`ModelInfo::is_thinking_model`).
    /// Drives the default-`reasoning:on` behavior when the caller omits `think`.
    pub model_supports_thinking: bool,
}

pub async fn resolve_model_target<'a>(
    context: &RequestContext<'a>,
    model_resolver: &Arc<ModelResolver>,
    requested_model: &str,
    cancellation_token: CancellationToken,
) -> Result<(String, Option<VirtualModelEntry>), ProxyError> {
    if let Some(entry) = context.virtual_models.get(requested_model).await {
        return Ok((entry.target_model_id.clone(), Some(entry)));
    }

    model_resolver
        .resolve_model_name(requested_model, context.client, cancellation_token)
        .await
        .map(|id| (id, None))
}

pub async fn resolve_model_with_context<'a>(
    context: &RequestContext<'a>,
    model_resolver: &Arc<ModelResolver>,
    requested_model: &str,
    request_body: &Value,
    wants_thinking_default: bool,
    cancellation_token: CancellationToken,
) -> Result<ModelResolutionContext, ProxyError> {
    let (lm_studio_model_id, virtual_entry) = resolve_model_target(
        context,
        model_resolver,
        requested_model,
        cancellation_token.clone(),
    )
    .await?;

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

    // Resolve the model's reasoning capability so the inference path can default
    // `reasoning:on` for thinking models when the caller omitted `think`
    // (matching real Ollama). The lookup costs a `GET /api/v1/models`, so skip it
    // unless it can change the outcome: only the chat/generate paths
    // (`wants_thinking_default = true`) reason, and only when no explicit
    // `think`/`reasoning_effort` is set (an explicit value always wins
    // downstream). Embeddings pass `false` — they never reason, and the legacy
    // `/api/embeddings` shape carries a `prompt` that must NOT be mistaken for an
    // inference body. Best-effort: an unknown / unfetchable model is non-thinking.
    let think_absent =
        request_body.get("think").is_none() && request_body.get("reasoning_effort").is_none();
    let model_supports_thinking = if wants_thinking_default && think_absent {
        let model_info = fetch_model_info_for_id(
            context,
            model_resolver,
            &lm_studio_model_id,
            cancellation_token,
        )
        .await?;
        model_info
            .map(|info| info.is_thinking_model())
            .unwrap_or(false)
    } else {
        false
    };

    Ok(ModelResolutionContext {
        lm_studio_model_id,
        effective_options,
        effective_format,
        system_prompt,
        model_supports_thinking,
    })
}

pub async fn fetch_model_info_for_id(
    context: &RequestContext<'_>,
    model_resolver: &Arc<ModelResolver>,
    target_model_id: &str,
    cancellation_token: CancellationToken,
) -> Result<Option<ModelInfo>, ProxyError> {
    let models = model_resolver
        .get_all_models(context.client, cancellation_token)
        .await?;
    Ok(models.into_iter().find(|model| model.id == target_model_id))
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

#[cfg(test)]
#[path = "../../../tests/unit/handlers_ollama_resolution.rs"]
mod tests;
