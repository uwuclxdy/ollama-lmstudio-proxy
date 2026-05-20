use std::sync::Arc;
use std::time::Instant;

use axum::body::Body;
use axum::response::Response;
use bytes::Bytes;
use http::{HeaderName, HeaderValue};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::api::RequestContext;
use crate::api::retry::with_retry_and_cancellation;
use crate::constants::{LOG_PREFIX_INFO, LOG_PREFIX_SUCCESS};
use crate::error::ProxyError;
use crate::http::body::{parse_json_body_template, prepare_request_body};
use crate::http::{build_forward_headers, client::CancellableRequest};
use crate::logging::{LogConfig, format_duration, log_request, log_timed};
use crate::model::ModelResolver;
use crate::streaming::{handle_passthrough_streaming_response, is_streaming_request};

pub struct LmStudioPassthroughRequest {
    pub method: http::Method,
    pub endpoint: String,
    pub body: Bytes,
    pub headers: http::HeaderMap,
    pub query: Option<String>,
}

pub async fn handle_lmstudio_passthrough(
    context: RequestContext<'_>,
    model_resolver: Arc<ModelResolver>,
    request: LmStudioPassthroughRequest,
    cancellation_token: CancellationToken,
    load_timeout_seconds: u64,
) -> Result<axum::response::Response, ProxyError> {
    let start_time = Instant::now();
    let LmStudioPassthroughRequest {
        method,
        endpoint,
        body,
        headers,
        query,
    } = request;

    if LogConfig::get().debug_enabled {
        log::debug!("passthrough request: {} {}", method, endpoint);
        if let Ok(body_str) = std::str::from_utf8(&body) {
            log::debug!("passthrough body: {}", body_str);
        }
    }

    let json_body_template = parse_json_body_template(&headers, &body)?;
    let original_model_name = json_body_template
        .as_ref()
        .and_then(|value: &Value| value.get("model"))
        .and_then(|value: &Value| value.as_str())
        .map(|s| s.to_string());

    let operation = {
        let context = context.clone();
        let model_resolver = model_resolver.clone();
        let method_clone = method.clone();
        let endpoint_clone = endpoint.clone();
        let body_clone = body.clone();
        let headers_clone = headers.clone();
        let query_clone = query.clone();
        let json_template_clone = json_body_template.clone();
        let cancellation_clone = cancellation_token.clone();
        let original_model_name_clone = original_model_name.clone();

        move || {
            let context = context.clone();
            let model_resolver = model_resolver.clone();
            let method = method_clone.clone();
            let endpoint = endpoint_clone.clone();
            let body_bytes = body_clone.clone();
            let headers = headers_clone.clone();
            let query = query_clone.clone();
            let json_template = json_template_clone.clone();
            let cancellation_token = cancellation_clone.clone();
            let original_model_name = original_model_name_clone.clone();

            async move {
                let mut current_body = json_template.clone();
                let mut resolved_model_name: Option<String> = None;

                if let Some(ref mut body_json) = current_body
                    && let Some(model_name) =
                        body_json.get("model").and_then(|m: &Value| m.as_str())
                {
                    let resolved_model = model_resolver
                        .resolve_model_name(model_name, context.client, cancellation_token.clone())
                        .await?;
                    resolved_model_name = Some(resolved_model.clone());
                    if let Some(obj) = body_json.as_object_mut() {
                        obj.insert("model".to_string(), Value::String(resolved_model));
                    }
                }

                let final_endpoint_url = context.append_query_params(
                    determine_passthrough_endpoint_url(
                        context.lmstudio_url,
                        &endpoint,
                        &model_resolver,
                    ),
                    query.as_deref(),
                );

                let log_model = resolved_model_name
                    .as_deref()
                    .or(original_model_name.as_deref());
                log_request(method.as_str(), &final_endpoint_url, log_model);

                if let Some(body_json) = current_body {
                    forward_json_body_request(ForwardJsonRequest {
                        client: context.client,
                        method,
                        endpoint_url: &final_endpoint_url,
                        headers: &headers,
                        body_json,
                        body_bytes: &body_bytes,
                        endpoint: &endpoint,
                        original_model_name: original_model_name.as_deref(),
                        cancellation_token,
                    })
                    .await
                } else {
                    forward_raw_body_request(
                        context.client,
                        method,
                        &final_endpoint_url,
                        &headers,
                        &body_bytes,
                        cancellation_token,
                    )
                    .await
                }
            }
        }
    };

    let result = match original_model_name.as_deref() {
        Some(model) => {
            with_retry_and_cancellation(
                &context,
                model,
                load_timeout_seconds,
                operation,
                cancellation_token,
            )
            .await?
        }
        None => {
            crate::check_cancelled!(cancellation_token);
            operation().await?
        }
    };

    log_timed(LOG_PREFIX_SUCCESS, "LM Studio passthrough", start_time);
    Ok(result)
}

struct ForwardJsonRequest<'a> {
    client: &'a reqwest::Client,
    method: http::Method,
    endpoint_url: &'a str,
    headers: &'a http::HeaderMap,
    body_json: Value,
    body_bytes: &'a Bytes,
    endpoint: &'a str,
    original_model_name: Option<&'a str>,
    cancellation_token: CancellationToken,
}

async fn forward_json_body_request(
    req: ForwardJsonRequest<'_>,
) -> Result<axum::response::Response, ProxyError> {
    let ForwardJsonRequest {
        client,
        method,
        endpoint_url,
        headers,
        body_json,
        body_bytes,
        endpoint,
        original_model_name,
        cancellation_token,
    } = req;
    let is_streaming = is_streaming_request(&body_json);
    let prepared_body = prepare_request_body(Some(body_json), body_bytes)?;

    let forward_headers = build_forward_headers(headers, prepared_body.is_json);

    let lm_studio_request_start = Instant::now();
    let cancellable_request = CancellableRequest::new(client, cancellation_token.clone());
    let response = cancellable_request
        .make_raw_request(method, endpoint_url, forward_headers, prepared_body.bytes)
        .await?;

    if original_model_name.is_some()
        && (endpoint.contains("completion") || endpoint.contains("chat"))
    {
        log_timed(
            LOG_PREFIX_INFO,
            &format!(
                "LM Studio responded | {}",
                format_duration(lm_studio_request_start.elapsed())
            ),
            lm_studio_request_start,
        );
    }

    route_response(response, is_streaming, cancellation_token).await
}

async fn forward_raw_body_request(
    client: &reqwest::Client,
    method: http::Method,
    endpoint_url: &str,
    headers: &http::HeaderMap,
    body_bytes: &Bytes,
    cancellation_token: CancellationToken,
) -> Result<axum::response::Response, ProxyError> {
    let forward_headers = build_forward_headers(headers, false);
    let prepared_body = prepare_request_body(None, body_bytes)?;

    let cancellable_request = CancellableRequest::new(client, cancellation_token.clone());
    let response = cancellable_request
        .make_raw_request(method, endpoint_url, forward_headers, prepared_body.bytes)
        .await?;

    forward_raw_response(response).await
}

async fn route_response(
    response: reqwest::Response,
    is_streaming: bool,
    cancellation_token: CancellationToken,
) -> Result<axum::response::Response, ProxyError> {
    if is_streaming {
        if LogConfig::get().debug_enabled {
            log::debug!("passthrough response: (streaming)");
        }
        handle_passthrough_streaming_response(response, cancellation_token, 60).await
    } else {
        route_non_streaming_response(response, cancellation_token).await
    }
}

async fn route_non_streaming_response(
    response: reqwest::Response,
    _cancellation_token: CancellationToken,
) -> Result<axum::response::Response, ProxyError> {
    if LogConfig::get().debug_enabled {
        log::debug!("passthrough response: (verbatim)");
    }
    forward_raw_response(response).await
}

fn determine_passthrough_endpoint_url(
    lmstudio_base_url: &str,
    requested_endpoint: &str,
    _model_resolver: &Arc<ModelResolver>,
) -> String {
    format!("{}{}", lmstudio_base_url, requested_endpoint)
}

async fn forward_raw_response(response: reqwest::Response) -> Result<Response, ProxyError> {
    let status = http::StatusCode::from_u16(response.status().as_u16())
        .map_err(|_| ProxyError::internal_server_error("invalid status code from LM Studio"))?;
    let headers = response.headers().clone();
    let body = Body::from_stream(response.bytes_stream());

    let mut builder = Response::builder().status(status);
    for (name, value) in headers.iter() {
        if let (Ok(hn), Ok(hv)) = (
            HeaderName::from_bytes(name.as_str().as_bytes()),
            HeaderValue::from_bytes(value.as_bytes()),
        ) {
            builder = builder.header(hn, hv);
        }
    }

    builder
        .body(body)
        .map_err(|_| ProxyError::internal_server_error("failed to build passthrough response"))
}

#[cfg(test)]
#[path = "../../tests/unit/handlers_lmstudio.rs"]
mod tests;
