use std::time::Instant;

use reqwest::Method;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::api::RequestContext;
use crate::constants::{
    ERROR_LM_STUDIO_UNAVAILABLE, LM_STUDIO_NATIVE_MODELS, LOG_PREFIX_ERROR, LOG_PREFIX_SUCCESS,
};
use crate::error::ProxyError;
use crate::http::CancellableRequest;
use crate::logging::{LogConfig, log_timed};

pub async fn handle_ollama_root() -> Result<axum::response::Response, ProxyError> {
    use axum::body::Body;
    Ok(axum::response::Response::builder()
        .status(http::StatusCode::OK)
        .header("Content-Type", "text/plain; charset=utf-8")
        .body(Body::from("Ollama is running"))
        .unwrap_or_else(|_| {
            axum::response::Response::builder()
                .status(http::StatusCode::INTERNAL_SERVER_ERROR)
                .body(Body::from("Internal Server Error"))
                .unwrap()
        }))
}

pub async fn handle_ollama_version(version: &str) -> Result<axum::response::Response, ProxyError> {
    if LogConfig::get().debug_enabled {
        log::debug!("version request");
    }
    let response = json!({
        "version": version
    });
    if LogConfig::get().debug_enabled {
        log::debug!(
            "version response: {}",
            serde_json::to_string_pretty(&response).unwrap_or_default()
        );
    }
    Ok(crate::http::json_response(&response))
}

pub async fn handle_health_check(
    context: RequestContext<'_>,
    cancellation_token: CancellationToken,
) -> Result<Value, ProxyError> {
    let start_time = Instant::now();
    if LogConfig::get().debug_enabled {
        log::debug!("health check request");
    }
    let url = context.endpoint_url(LM_STUDIO_NATIVE_MODELS);
    let request = CancellableRequest::new(context.client, cancellation_token.clone());

    match request.make_request(Method::GET, &url, None::<Value>).await {
        Ok(response) => {
            let status = response.status();
            let is_healthy = status.is_success();
            let mut model_count = 0;

            if is_healthy && let Ok(models_response) = response.json::<Value>().await {
                model_count = models_response
                    .get("models")
                    .or_else(|| models_response.get("data"))
                    .and_then(|d| d.as_array())
                    .map(|arr| arr.len())
                    .unwrap_or(0);
            }

            log_timed(
                if is_healthy {
                    LOG_PREFIX_SUCCESS
                } else {
                    LOG_PREFIX_ERROR
                },
                &format!("health check - {} models", model_count),
                start_time,
            );

            let response = json!({
                "status": if is_healthy { "healthy" } else { "unhealthy" },
                "lmstudio_url": context.lmstudio_url,
                "http_status": status.as_u16(),
                "models_known_to_lmstudio": model_count,
                "response_time_ms": start_time.elapsed().as_millis(),
                "timestamp": chrono::Utc::now().to_rfc3339(),
                "proxy_version": crate::VERSION
            });
            if LogConfig::get().debug_enabled {
                log::debug!(
                    "health check response: {}",
                    serde_json::to_string_pretty(&response).unwrap_or_default()
                );
            }
            Ok(response)
        }
        Err(e) if e.is_cancelled() => Err(ProxyError::request_cancelled()),
        Err(e) => {
            log_timed(
                LOG_PREFIX_ERROR,
                &format!("health check failed: {}", e.message),
                start_time,
            );
            let response = json!({
                "status": "unreachable",
                "lmstudio_url": context.lmstudio_url,
                "error_message": e.message,
                "error_details": ERROR_LM_STUDIO_UNAVAILABLE,
                "response_time_ms": start_time.elapsed().as_millis(),
                "timestamp": chrono::Utc::now().to_rfc3339(),
                "proxy_version": crate::VERSION
            });
            if LogConfig::get().debug_enabled {
                log::debug!(
                    "health check response (error): {}",
                    serde_json::to_string_pretty(&response).unwrap_or_default()
                );
            }
            Ok(response)
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/handlers_ollama_health.rs"]
mod tests;
