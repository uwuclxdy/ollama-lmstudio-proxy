use std::time::Instant;

use bytes::Buf;
use futures_util::{Stream, TryStreamExt};
use http_body_util::Empty;

use crate::constants::LOG_PREFIX_SUCCESS;
use crate::error::ProxyError;
use crate::handlers::RequestContext;
use crate::logging::{LogConfig, log_request, log_timed};

pub async fn handle_blob_head(
    context: RequestContext<'_>,
    digest: String,
) -> Result<warp::reply::Response, ProxyError> {
    if LogConfig::get().debug_enabled {
        log::debug!("blob head request: {}", digest);
    }
    let exists = context.blob_store.exists(&digest).await?;
    let status = if exists {
        warp::http::StatusCode::OK
    } else {
        warp::http::StatusCode::NOT_FOUND
    };

    if LogConfig::get().debug_enabled {
        log::debug!("blob head response: {}", status);
    }

    let body_impl = Empty::<bytes::Bytes>::new();
    let boxed_body = http_body_util::BodyExt::boxed(body_impl);

    let temp_response = warp::http::Response::builder()
        .status(status)
        .body(boxed_body)
        .map_err(|_| ProxyError::internal_server_error("failed to build blob response"))?;

    Ok(unsafe {
        std::mem::transmute::<
            warp::http::Response<
                http_body_util::combinators::BoxBody<bytes::Bytes, std::convert::Infallible>,
            >,
            warp::reply::Response,
        >(temp_response)
    })
}

pub async fn handle_blob_upload<S, B>(
    context: RequestContext<'_>,
    digest: String,
    stream: S,
) -> Result<warp::reply::Response, ProxyError>
where
    S: Stream<Item = Result<B, warp::Error>> + Unpin,
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

    let body_impl = Empty::<bytes::Bytes>::new();
    let boxed_body = http_body_util::BodyExt::boxed(body_impl);

    let temp_response = warp::http::Response::builder()
        .status(warp::http::StatusCode::CREATED)
        .body(boxed_body)
        .map_err(|_| ProxyError::internal_server_error("failed to build blob upload response"))?;

    Ok(unsafe {
        std::mem::transmute::<
            warp::http::Response<
                http_body_util::combinators::BoxBody<bytes::Bytes, std::convert::Infallible>,
            >,
            warp::reply::Response,
        >(temp_response)
    })
}
