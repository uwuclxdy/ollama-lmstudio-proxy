use futures_util::StreamExt;
use http_body_util::StreamBody;
use serde_json::Value;
use tokio::sync::mpsc;

use crate::constants::{
    CONTENT_TYPE_SSE, HEADER_ACCESS_CONTROL_ALLOW_HEADERS, HEADER_ACCESS_CONTROL_ALLOW_METHODS,
    HEADER_ACCESS_CONTROL_ALLOW_ORIGIN, HEADER_CACHE_CONTROL, HEADER_CONNECTION,
};
use crate::error::ProxyError;

pub enum StreamContentType {
    Ndjson,
    Sse,
}

pub fn is_streaming_request(body: &Value) -> bool {
    body.get("stream")
        .and_then(|s| s.as_bool())
        .unwrap_or(false)
}

fn create_generic_streaming_response(
    rx: mpsc::UnboundedReceiver<Result<bytes::Bytes, std::io::Error>>,
    content_type: &str,
    error_message_on_build_fail: &str,
) -> Result<warp::reply::Response, ProxyError> {
    use bytes::Bytes;

    let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);
    // Create a body using the same pattern as warp's internal wrap_stream
    let mapped_stream = stream.map(|item: Result<Bytes, std::io::Error>| {
        item.map(warp::hyper::body::Frame::data)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
    });

    let body_impl = StreamBody::new(mapped_stream);
    let boxed_body = http_body_util::BodyExt::boxed(body_impl);

    let temp_response = warp::http::Response::builder()
        .status(warp::http::StatusCode::OK)
        .header("content-type", content_type)
        .header("cache-control", HEADER_CACHE_CONTROL)
        .header("connection", HEADER_CONNECTION)
        .header(
            "access-control-allow-origin",
            HEADER_ACCESS_CONTROL_ALLOW_ORIGIN,
        )
        .header(
            "access-control-allow-methods",
            HEADER_ACCESS_CONTROL_ALLOW_METHODS,
        )
        .header(
            "access-control-allow-headers",
            HEADER_ACCESS_CONTROL_ALLOW_HEADERS,
        )
        .body(boxed_body)
        .map_err(|_| ProxyError::internal_server_error(error_message_on_build_fail))?;

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

pub fn create_streaming_response(
    rx: mpsc::UnboundedReceiver<Result<bytes::Bytes, std::io::Error>>,
    content_type: StreamContentType,
) -> Result<warp::reply::Response, ProxyError> {
    let (content_type_str, error_message) = match content_type {
        StreamContentType::Ndjson => (
            "application/x-ndjson; charset=utf-8",
            "failed to create NDJSON streaming response",
        ),
        StreamContentType::Sse => (CONTENT_TYPE_SSE, "failed to create SSE streaming response"),
    };

    create_generic_streaming_response(rx, content_type_str, error_message)
}

pub fn create_ndjson_stream_response(
    rx: mpsc::UnboundedReceiver<Result<bytes::Bytes, std::io::Error>>,
    error_message_on_build_fail: &str,
) -> Result<warp::reply::Response, ProxyError> {
    let result = create_streaming_response(rx, StreamContentType::Ndjson);
    if result.is_err() {
        return Err(ProxyError::internal_server_error(
            error_message_on_build_fail,
        ));
    }
    result
}
