use std::time::Instant;

use bytes::Bytes;
use futures_util::StreamExt;
use http_body_util::StreamBody;
use reqwest::header::{
    self, HeaderMap as ReqHeaderMap, HeaderName as ReqHeaderName, HeaderValue as ReqHeaderValue,
};
use serde_json::Value;
use tokio_util::sync::CancellationToken;
use warp::http::{self, Response as WarpResponse};

use crate::constants::{
    CONTENT_TYPE_JSON, ERROR_LM_STUDIO_UNAVAILABLE, ERROR_TIMEOUT, LOG_PREFIX_INFO,
    LOG_PREFIX_SUCCESS,
};
use crate::error::ProxyError;
use crate::handlers::RequestContext;
use crate::handlers::retry::{with_retry_and_cancellation, with_simple_retry};
use crate::http::{client::handle_json_response, json_response};
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

    let json_body_template = parse_json_body_template(&headers, &body)?;
    let original_model_name = json_body_template
        .as_ref()
        .and_then(|value| value.get("model"))
        .and_then(|value| value.as_str())
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
                    && let Some(model_name) = body_json.get("model").and_then(|m| m.as_str())
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

                let final_endpoint_url = append_query_params(
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

                let is_streaming = current_body.as_ref().is_some_and(is_streaming_request);

                let prepared_body = prepare_request_body(current_body, &body_bytes)?;
                let forward_headers = build_forward_headers(&headers, prepared_body.is_json);

                let lm_studio_request_start = Instant::now();
                let response = send_raw_request(
                    context.client,
                    method.clone(),
                    &final_endpoint_url,
                    forward_headers,
                    prepared_body.bytes,
                    cancellation_token.clone(),
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
                    handle_passthrough_streaming_response(response, cancellation_token, 60).await
                } else {
                    let is_json_response = response_is_json(&response);
                    if is_json_response {
                        let json_body = handle_json_response(response, cancellation_token).await?;
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
            }
        }
    };

    let result = if let Some(ref model) = original_model_name {
        with_retry_and_cancellation(
            &context,
            model,
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

fn append_query_params(mut base: String, query: Option<&str>) -> String {
    if let Some(qs) = query
        && !qs.is_empty()
    {
        if base.contains('?') {
            base.push('&');
        } else {
            base.push('?');
        }
        base.push_str(qs);
    }
    base
}

fn parse_json_body_template(
    headers: &http::HeaderMap,
    body: &Bytes,
) -> Result<Option<Value>, ProxyError> {
    if body.is_empty() {
        return Ok(None);
    }

    if !should_parse_as_json(headers, body) {
        return Ok(None);
    }

    serde_json::from_slice::<Value>(body)
        .map(Some)
        .map_err(|e| ProxyError::bad_request(&format!("invalid JSON payload: {}", e)))
}

fn should_parse_as_json(headers: &http::HeaderMap, body: &Bytes) -> bool {
    contains_json_content_type(headers) || body_looks_like_json(body)
}

fn contains_json_content_type(headers: &http::HeaderMap) -> bool {
    headers
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .map(|ct| ct.to_ascii_lowercase().contains("json"))
        .unwrap_or(false)
}

fn body_looks_like_json(body: &Bytes) -> bool {
    body.iter()
        .copied()
        .find(|byte| !byte.is_ascii_whitespace())
        .map(|byte| matches!(byte, b'{' | b'['))
        .unwrap_or(false)
}

struct PreparedBody {
    bytes: Option<Vec<u8>>,
    is_json: bool,
}

fn prepare_request_body(
    json_body: Option<Value>,
    original_bytes: &Bytes,
) -> Result<PreparedBody, ProxyError> {
    if let Some(value) = json_body {
        let serialized = serde_json::to_vec(&value).map_err(|e| {
            ProxyError::internal_server_error(&format!(
                "failed to serialize LM Studio request body: {}",
                e
            ))
        })?;
        Ok(PreparedBody {
            bytes: Some(serialized),
            is_json: true,
        })
    } else if !original_bytes.is_empty() {
        Ok(PreparedBody {
            bytes: Some(original_bytes.to_vec()),
            is_json: false,
        })
    } else {
        Ok(PreparedBody {
            bytes: None,
            is_json: false,
        })
    }
}

fn build_forward_headers(original: &http::HeaderMap, force_json: bool) -> ReqHeaderMap {
    let mut filtered = ReqHeaderMap::new();

    for (name, value) in original.iter() {
        let name_str = name.as_str();
        if name_str.eq_ignore_ascii_case("host")
            || name_str.eq_ignore_ascii_case("content-length")
            || name_str.eq_ignore_ascii_case("transfer-encoding")
        {
            continue;
        }
        if force_json && name_str.eq_ignore_ascii_case("content-type") {
            continue;
        }

        if let (Ok(req_name), Ok(req_value)) = (
            name_str.parse::<ReqHeaderName>(),
            ReqHeaderValue::from_bytes(value.as_bytes()),
        ) {
            filtered.append(req_name, req_value);
        }
    }

    if force_json {
        filtered.insert(
            header::CONTENT_TYPE,
            ReqHeaderValue::from_static(CONTENT_TYPE_JSON),
        );
    }

    filtered
}

async fn send_raw_request(
    client: &reqwest::Client,
    method: http::Method,
    url: &str,
    headers: ReqHeaderMap,
    body: Option<Vec<u8>>,
    cancellation_token: CancellationToken,
) -> Result<reqwest::Response, ProxyError> {
    let req_method = reqwest::Method::from_bytes(method.as_str().as_bytes())
        .map_err(|_| ProxyError::bad_request("unsupported HTTP method"))?;
    let mut builder = client.request(req_method, url);

    if !headers.is_empty() {
        builder = builder.headers(headers);
    }

    if let Some(payload) = body {
        builder = builder.body(payload);
    }

    tokio::select! {
        resp = builder.send() => resp.map_err(map_reqwest_error),
        _ = cancellation_token.cancelled() => Err(ProxyError::request_cancelled()),
    }
}

fn map_reqwest_error(err: reqwest::Error) -> ProxyError {
    if err.is_connect() {
        ProxyError::lm_studio_unavailable(ERROR_LM_STUDIO_UNAVAILABLE)
    } else if err.is_timeout() {
        ProxyError::lm_studio_unavailable(ERROR_TIMEOUT)
    } else {
        log::error!("LM Studio passthrough: {}", err);
        ProxyError::internal_server_error(&format!("LM Studio request failed: {}", err))
    }
}

fn response_is_json(response: &reqwest::Response) -> bool {
    response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(|ct| ct.to_ascii_lowercase().contains("json"))
        .unwrap_or(false)
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
            http::header::HeaderName::from_bytes(name.as_str().as_bytes()),
            http::header::HeaderValue::from_bytes(value.as_bytes()),
        ) {
            builder = builder.header(warp_name, warp_value);
        }
    }

    let temp_response = builder
        .body(boxed_body)
        .map_err(|_| ProxyError::internal_server_error("failed to build passthrough response"))?;

    Ok(unsafe {
        std::mem::transmute::<
            warp::http::Response<
                http_body_util::combinators::BoxBody<
                    bytes::Bytes,
                    Box<dyn std::error::Error + Send + Sync>,
                >,
            >,
            warp::reply::Response,
        >(temp_response)
    })
}
