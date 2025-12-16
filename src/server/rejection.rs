use std::convert::Infallible;

use serde_json::json;
use warp::http::StatusCode;
use warp::{Rejection, Reply};

use crate::error::ProxyError;

pub async fn handle_rejection(rejection: Rejection) -> Result<impl Reply, Infallible> {
    let (status, message) = if let Some(proxy_err) = rejection.find::<ProxyError>() {
        (
            StatusCode::from_u16(proxy_err.status_code)
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            proxy_err.message.clone(),
        )
    } else if rejection.is_not_found() {
        (StatusCode::NOT_FOUND, "endpoint not found".to_string())
    } else if rejection.find::<warp::reject::MethodNotAllowed>().is_some() {
        (
            StatusCode::METHOD_NOT_ALLOWED,
            "method not allowed".to_string(),
        )
    } else if rejection.find::<warp::reject::InvalidHeader>().is_some() {
        (StatusCode::BAD_REQUEST, "invalid header".to_string())
    } else if rejection.find::<warp::reject::MissingHeader>().is_some() {
        (
            StatusCode::BAD_REQUEST,
            "missing required header".to_string(),
        )
    } else if rejection.find::<warp::reject::PayloadTooLarge>().is_some() {
        (
            StatusCode::PAYLOAD_TOO_LARGE,
            "request body too large".to_string(),
        )
    } else if let Some(body_err) = rejection.find::<warp::body::BodyDeserializeError>() {
        (
            StatusCode::BAD_REQUEST,
            format!("invalid request body: {}", body_err),
        )
    } else {
        log::error!("unhandled rejection: {:?}", rejection);
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal server error".to_string(),
        )
    };

    let error_response = json!({
        "error": message,
        "status": status.as_u16()
    });

    Ok(warp::reply::with_status(
        warp::reply::json(&error_response),
        status,
    ))
}
