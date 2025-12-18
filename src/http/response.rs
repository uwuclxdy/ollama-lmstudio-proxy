use serde_json::Value;
use warp::http::HeaderMap;

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

/// Build forward headers for requests, filtering out hop-by-hop headers
pub fn build_forward_headers(original: &HeaderMap, force_json: bool) -> reqwest::header::HeaderMap {
    use reqwest::header::{
        HeaderMap as ReqHeaderMap, HeaderName as ReqHeaderName, HeaderValue as ReqHeaderValue,
    };
    use warp::http::header;

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
