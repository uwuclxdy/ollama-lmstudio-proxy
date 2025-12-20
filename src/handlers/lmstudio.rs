use std::time::Instant;

use bytes::Bytes;
use futures_util::StreamExt;
use http_body_util::StreamBody;
use reqwest::header::{self};
use serde_json::Value;
use tokio_util::sync::CancellationToken;
use warp::http::{self, Response as WarpResponse};

use crate::constants::{LOG_PREFIX_INFO, LOG_PREFIX_SUCCESS};
use crate::error::ProxyError;
use crate::handlers::RequestContext;
use crate::handlers::retry::{with_retry_and_cancellation, with_simple_retry};
use crate::http::{
    build_forward_headers,
    client::{CancellableRequest, handle_json_response},
    json_response,
    parsing::parse_json_body_template,
    request::prepare_request_body,
};
use crate::logging::{LogConfig, format_duration, log_request, log_timed};
use crate::server::ModelResolverType;
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
    model_resolver: ModelResolverType,
    request: LmStudioPassthroughRequest,
    cancellation_token: CancellationToken,
    load_timeout_seconds: u64,
) -> Result<warp::reply::Response, ProxyError> {
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

    let json_body_template = parse_json_body_template(&headers, &body)
        .map_err(|e| ProxyError::bad_request(&format!("Failed to parse JSON body: {}", e)))?;
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
                    let resolved_model = match &model_resolver {
                        ModelResolverType::Native(resolver) => {
                            resolver
                                .resolve_model_name(
                                    model_name,
                                    context.client,
                                    cancellation_token.clone(),
                                )
                                .await?
                        }
                    };
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

                if let Some(ref body_json) = current_body {
                    let is_streaming = is_streaming_request(body_json);
                    let prepared_body = prepare_request_body(Some(body_json.clone()), &body_bytes)
                        .map_err(|e| {
                            ProxyError::bad_request(&format!(
                                "Failed to prepare request body: {}",
                                e
                            ))
                        })?;

                    let forward_headers = build_forward_headers(&headers, prepared_body.is_json);

                    let lm_studio_request_start = Instant::now();
                    let cancellable_request =
                        CancellableRequest::new(context.client, cancellation_token.clone());
                    let response = cancellable_request
                        .make_raw_request(
                            method.clone(),
                            &final_endpoint_url,
                            forward_headers,
                            prepared_body.bytes,
                        )
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

                    if is_streaming {
                        if LogConfig::get().debug_enabled {
                            log::debug!("passthrough response: (streaming)");
                        }
                        handle_passthrough_streaming_response(response, cancellation_token, 60)
                            .await
                    } else {
                        let is_json_response = crate::http::response::is_json_response(&response);
                        if is_json_response {
                            let json_body =
                                handle_json_response(response, cancellation_token).await?;
                            if LogConfig::get().debug_enabled {
                                log::debug!(
                                    "passthrough response: {}",
                                    serde_json::to_string_pretty(&json_body).unwrap_or_default()
                                );
                            }
                            Ok(json_response(&json_body))
                        } else {
                            if LogConfig::get().debug_enabled {
                                log::debug!("passthrough response: (raw/non-json)");
                            }
                            forward_raw_response(response).await
                        }
                    }
                } else {
                    // Handle non-JSON body case
                    let forward_headers = build_forward_headers(&headers, false);
                    let prepared_body = prepare_request_body(None, &body_bytes).map_err(|e| {
                        ProxyError::bad_request(&format!("Failed to prepare request body: {}", e))
                    })?;

                    let cancellable_request =
                        CancellableRequest::new(context.client, cancellation_token.clone());
                    let response = cancellable_request
                        .make_raw_request(
                            method.clone(),
                            &final_endpoint_url,
                            forward_headers,
                            prepared_body.bytes,
                        )
                        .await?;

                    forward_raw_response(response).await
                }
            }
        }
    };

    let result = if let Some(ref model) = original_model_name {
        with_retry_and_cancellation(
            &context,
            model.as_str(),
            load_timeout_seconds,
            operation,
            cancellation_token,
        )
        .await?
    } else {
        with_simple_retry(operation, cancellation_token).await?
    };

    log_timed(LOG_PREFIX_SUCCESS, "LM Studio passthrough", start_time);
    Ok(result)
}

fn determine_passthrough_endpoint_url(
    lmstudio_base_url: &str,
    requested_endpoint: &str,
    _model_resolver: &ModelResolverType,
) -> String {
    format!("{}{}", lmstudio_base_url, requested_endpoint)
}

async fn forward_raw_response(
    response: reqwest::Response,
) -> Result<warp::reply::Response, ProxyError> {
    let status = http::StatusCode::from_u16(response.status().as_u16())
        .map_err(|_| ProxyError::internal_server_error("invalid status code from LM Studio"))?;
    let headers = response.headers().clone();
    let stream = response.bytes_stream();

    // Create a body using the same pattern as warp's internal wrap_stream
    let mapped_stream = stream.map(|item: Result<Bytes, _>| {
        item.map(warp::hyper::body::Frame::data)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
    });

    let body_impl = StreamBody::new(mapped_stream);
    let boxed_body = http_body_util::BodyExt::boxed(body_impl);

    let mut builder = WarpResponse::builder().status(status);
    for (name, value) in headers.iter() {
        if let (Ok(warp_name), Ok(warp_value)) = (
            header::HeaderName::from_bytes(name.as_str().as_bytes()),
            header::HeaderValue::from_bytes(value.as_bytes()),
        ) {
            builder = builder.header(warp_name, warp_value);
        }
    }

    let temp_response = builder
        .body(boxed_body)
        .map_err(|_| ProxyError::internal_server_error("failed to build passthrough response"))?;

    Ok(unsafe {
        std::mem::transmute::<
            http::Response<
                http_body_util::combinators::BoxBody<
                    Bytes,
                    Box<dyn std::error::Error + Send + Sync>,
                >,
            >,
            warp::reply::Response,
        >(temp_response)
    })
}
