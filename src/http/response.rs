use serde_json::Value;

use crate::constants::{
    CONTENT_TYPE_JSON, HEADER_ACCESS_CONTROL_ALLOW_HEADERS, HEADER_ACCESS_CONTROL_ALLOW_METHODS,
    HEADER_ACCESS_CONTROL_ALLOW_ORIGIN, HEADER_CACHE_CONTROL,
};

pub fn json_response(value: &Value) -> warp::reply::Response {
    let json_string = serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string());
    let content_length = json_string.len();

    warp::http::Response::builder()
        .status(warp::http::StatusCode::OK)
        .header("Content-Type", CONTENT_TYPE_JSON)
        .header("Content-Length", content_length.to_string())
        .header("Cache-Control", HEADER_CACHE_CONTROL)
        .header(
            "Access-Control-Allow-Origin",
            HEADER_ACCESS_CONTROL_ALLOW_ORIGIN,
        )
        .header(
            "Access-Control-Allow-Methods",
            HEADER_ACCESS_CONTROL_ALLOW_METHODS,
        )
        .header(
            "Access-Control-Allow-Headers",
            HEADER_ACCESS_CONTROL_ALLOW_HEADERS,
        )
        .body(json_string.into())
        .unwrap_or_else(|_| {
            warp::http::Response::builder()
                .status(warp::http::StatusCode::INTERNAL_SERVER_ERROR)
                .body("Internal Server Error".into())
                .unwrap()
        })
}
