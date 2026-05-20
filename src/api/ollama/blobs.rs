use std::time::Instant;

use axum::body::Body;
use axum::response::Response;
use bytes::Buf;
use futures_util::{Stream, TryStreamExt};
use http::StatusCode;

use crate::api::RequestContext;
use crate::constants::LOG_PREFIX_SUCCESS;
use crate::error::ProxyError;
use crate::logging::{LogConfig, log_request, log_timed};

pub async fn handle_blob_head(
    context: RequestContext<'_>,
    digest: String,
) -> Result<Response, ProxyError> {
    if LogConfig::get().debug_enabled {
        log::debug!("blob head request: {}", digest);
    }
    let exists = context.blob_store.exists(&digest).await?;
    let status = if exists {
        StatusCode::OK
    } else {
        StatusCode::NOT_FOUND
    };

    if LogConfig::get().debug_enabled {
        log::debug!("blob head response: {}", status);
    }

    Response::builder()
        .status(status)
        .body(Body::empty())
        .map_err(|_| ProxyError::internal_server_error("failed to build blob response"))
}

pub async fn handle_blob_upload<S, B>(
    context: RequestContext<'_>,
    digest: String,
    stream: S,
) -> Result<Response, ProxyError>
where
    S: Stream<Item = Result<B, axum::Error>> + Unpin,
    B: Buf,
{
    let start_time = Instant::now();
    log_request("POST", "/api/blobs", Some(&digest));
    if LogConfig::get().debug_enabled {
        log::debug!("blob upload request: {}", digest);
    }

    let byte_stream = stream.map_ok(|mut buf| buf.copy_to_bytes(buf.remaining()));

    context.blob_store.save_stream(&digest, byte_stream).await?;

    log_timed(
        LOG_PREFIX_SUCCESS,
        &format!("stored blob {}", digest),
        start_time,
    );

    if LogConfig::get().debug_enabled {
        log::debug!("blob upload response: created");
    }

    Response::builder()
        .status(StatusCode::CREATED)
        .body(Body::empty())
        .map_err(|_| ProxyError::internal_server_error("failed to build blob upload response"))
}
