use std::pin::Pin;
use std::task::{Context, Poll};

use axum::body::Body;
use axum::response::Response;
use futures_util::{Stream, StreamExt};
use http::StatusCode;
use serde_json::Value;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::constants::{
    CONTENT_TYPE_SSE, HEADER_ACCESS_CONTROL_ALLOW_HEADERS, HEADER_ACCESS_CONTROL_ALLOW_METHODS,
    HEADER_ACCESS_CONTROL_ALLOW_ORIGIN, HEADER_CACHE_CONTROL, HEADER_CONNECTION,
};
use crate::error::ProxyError;

pub enum StreamContentType {
    Ndjson,
    Sse,
}

const NDJSON_MEDIA_TYPE: &str = "application/x-ndjson";
const CONTENT_TYPE_NDJSON: &str = "application/x-ndjson; charset=utf-8";

pub fn is_streaming_request(body: &Value) -> bool {
    body.get("stream")
        .and_then(|s| s.as_bool())
        .unwrap_or(false)
}

fn create_generic_streaming_response(
    rx: mpsc::UnboundedReceiver<Result<bytes::Bytes, std::io::Error>>,
    content_type: &str,
    error_message_on_build_fail: &str,
) -> Result<Response, ProxyError> {
    let body = Body::from_stream(UnboundedReceiverStream::new(rx));

    Response::builder()
        .status(StatusCode::OK)
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
        .body(body)
        .map_err(|_| ProxyError::internal_server_error(error_message_on_build_fail))
}

pub fn create_streaming_response(
    rx: mpsc::UnboundedReceiver<Result<bytes::Bytes, std::io::Error>>,
    content_type: StreamContentType,
) -> Result<Response, ProxyError> {
    let (content_type_str, error_message) = match content_type {
        StreamContentType::Ndjson => (
            CONTENT_TYPE_NDJSON,
            "failed to create NDJSON streaming response",
        ),
        StreamContentType::Sse => (CONTENT_TYPE_SSE, "failed to create SSE streaming response"),
    };

    create_generic_streaming_response(rx, content_type_str, error_message)
}

pub fn create_ndjson_stream_response(
    rx: mpsc::UnboundedReceiver<Result<bytes::Bytes, std::io::Error>>,
    error_message_on_build_fail: &str,
) -> Result<Response, ProxyError> {
    create_streaming_response(rx, StreamContentType::Ndjson)
        .map_err(|_| ProxyError::internal_server_error(error_message_on_build_fail))
}

/// Wraps a response body stream and fires `end_signal` once, when the body
/// reaches its own end (the inner stream yields `None`, i.e. every chunk was
/// delivered) or is dropped (client disconnect) — whichever comes first.
/// keep_alive:0 teardown waits on this signal, so a model stays loaded
/// exactly as long as the response stream is alive.
struct NotifyOnStreamEnd<S> {
    inner: S,
    end_signal: Option<oneshot::Sender<()>>,
}

impl<S> NotifyOnStreamEnd<S> {
    fn new(inner: S, end_signal: oneshot::Sender<()>) -> Self {
        Self {
            inner,
            end_signal: Some(end_signal),
        }
    }

    fn fire(&mut self) {
        if let Some(signal) = self.end_signal.take() {
            // The receiver may already be gone (teardown task aborted); the
            // body's end is a best-effort notification.
            let _ = signal.send(());
        }
    }
}

impl<S: Stream + Unpin> Stream for NotifyOnStreamEnd<S> {
    type Item = S::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<S::Item>> {
        match self.inner.poll_next_unpin(cx) {
            Poll::Ready(None) => {
                self.fire();
                Poll::Ready(None)
            }
            other => other,
        }
    }
}

impl<S> Drop for NotifyOnStreamEnd<S> {
    fn drop(&mut self) {
        self.fire();
    }
}

/// Whether `response` carries a streamed body — one of the two media types
/// `create_streaming_response` emits, compared on the media type alone so a
/// different charset parameter cannot change the teardown timing. Any other
/// body is complete when the handler returns, so its teardown runs
/// immediately.
fn is_streaming_response(response: &Response) -> bool {
    response
        .headers()
        .get(http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|content_type| content_type.split(';').next())
        .map(str::trim)
        .is_some_and(|media_type| media_type == NDJSON_MEDIA_TYPE || media_type == CONTENT_TYPE_SSE)
}

/// Wraps a streaming response's body with `NotifyOnStreamEnd` and returns
/// the receiver that resolves when the body ends or is dropped. Non-streaming
/// responses pass through unchanged with `None` (immediate teardown).
pub fn attach_stream_end_notifier(response: Response) -> (Response, Option<oneshot::Receiver<()>>) {
    if !is_streaming_response(&response) {
        return (response, None);
    }
    let (signal, receiver) = oneshot::channel();
    let (parts, body) = response.into_parts();
    let stream = body.into_data_stream();
    let body = Body::from_stream(NotifyOnStreamEnd::new(stream, signal));
    (Response::from_parts(parts, body), Some(receiver))
}

#[cfg(test)]
#[path = "../../tests/unit/streaming_response.rs"]
mod tests;
