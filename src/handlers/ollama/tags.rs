use std::time::Instant;

use serde_json::json;
use tokio_util::sync::CancellationToken;

use crate::constants::LOG_PREFIX_SUCCESS;
use crate::error::ProxyError;
use crate::handlers::RequestContext;
use crate::http::json_response;
use crate::logging::log_timed;
use crate::server::ModelResolverType;

use crate::logging::log_handler_io;
use crate::model::types::ModelInfo;

pub async fn handle_ollama_tags(
    context: RequestContext<'_>,
    model_resolver: ModelResolverType,
    cancellation_token: CancellationToken,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();

    let models = match &model_resolver {
        ModelResolverType::Native(resolver) => {
            resolver
                .get_all_models(context.client, cancellation_token)
                .await?
        }
    };

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
    log_handler_io("tags", None, Some(&response), false);
    Ok(json_response(&response))
}
