pub(crate) use std::error::Error;
use std::fmt;

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

use crate::constants::ERROR_CANCELLED;

/// Error type for the proxy server
#[derive(Debug, Clone)]
pub struct ProxyError {
    pub message: String,
    pub status_code: u16,
}

impl ProxyError {
    pub fn new(message: String, status_code: u16) -> Self {
        Self {
            message,
            status_code,
        }
    }

    pub fn internal_server_error(message: &str) -> Self {
        Self {
            message: message.to_string(),
            status_code: 500,
        }
    }

    pub fn bad_request(message: &str) -> Self {
        Self {
            message: message.to_string(),
            status_code: 400,
        }
    }

    pub fn not_found(message: &str) -> Self {
        Self {
            message: message.to_string(),
            status_code: 404,
        }
    }

    pub fn not_implemented(message: &str) -> Self {
        Self {
            message: message.to_string(),
            status_code: 501,
        }
    }

    pub fn request_cancelled() -> Self {
        Self {
            message: ERROR_CANCELLED.to_string(),
            status_code: 499,
        }
    }

    pub fn lm_studio_unavailable(message: &str) -> Self {
        Self {
            message: message.to_string(),
            status_code: 503,
        }
    }

    pub fn too_many_requests(message: &str) -> Self {
        Self {
            message: message.to_string(),
            status_code: 429,
        }
    }

    pub fn bad_gateway(message: &str) -> Self {
        Self {
            message: message.to_string(),
            status_code: 502,
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.status_code == 499
    }

    pub fn is_lm_studio_unavailable(&self) -> bool {
        self.status_code == 503
    }
}

impl fmt::Display for ProxyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ProxyError {}: {}", self.status_code, self.message)
    }
}

/// Map an HTTP status to Anthropic's documented error `type` literal.
///
/// The Anthropic surface (`/v1/messages*`) expects
/// `{"type":"error","error":{"type":...,"message":...}}` — SDKs branch on the
/// inner `type`, so unknown statuses collapse to the generic `api_error`.
pub fn anthropic_error_type(status_code: u16) -> &'static str {
    match status_code {
        400 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        413 => "request_too_large",
        429 => "rate_limit_error",
        529 => "overloaded_error",
        _ => "api_error",
    }
}

/// Build the Anthropic error envelope for a status + message pair.
pub fn anthropic_error_body(status_code: u16, message: &str) -> serde_json::Value {
    json!({
        "type": "error",
        "error": {
            "type": anthropic_error_type(status_code),
            "message": message,
        }
    })
}

impl ProxyError {
    /// Render this error in Anthropic's error envelope instead of the Ollama
    /// `{"error":msg}` shape, for errors surfacing on `/v1/messages*`.
    pub fn into_anthropic_response(self) -> Response {
        let status =
            StatusCode::from_u16(self.status_code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let body = Json(anthropic_error_body(self.status_code, &self.message));
        (status, body).into_response()
    }
}

impl Error for ProxyError {}

impl IntoResponse for ProxyError {
    fn into_response(self) -> Response {
        let status =
            StatusCode::from_u16(self.status_code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let body = Json(json!({
            "error": self.message,
        }));
        (status, body).into_response()
    }
}

#[macro_export]
macro_rules! check_cancelled {
    ($token:expr) => {
        if $token.is_cancelled() {
            return Err($crate::error::ProxyError::request_cancelled());
        }
    };
}
