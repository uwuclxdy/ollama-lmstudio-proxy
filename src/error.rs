pub(crate) use std::error::Error;
use std::fmt;

use axum::Json;
use axum::extract::rejection::{BytesRejection, JsonRejection, PathRejection, QueryRejection};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

use crate::constants::{ERROR_BODY_TOO_LARGE, ERROR_CANCELLED};

/// Error type for the proxy server
#[derive(Debug, Clone)]
pub struct ProxyError {
    pub message: String,
    pub status_code: u16,
    /// Set at the rejection site when the failing request belongs to the
    /// Anthropic surface, so `IntoResponse` renders the Anthropic envelope
    /// without any outer layer having to re-read the error from response
    /// extensions (which a response-rebuilding middleware would drop).
    anthropic_surface: bool,
}

impl ProxyError {
    pub fn new(message: String, status_code: u16) -> Self {
        Self {
            message,
            status_code,
            anthropic_surface: false,
        }
    }

    /// Stamp the envelope choice from the path the rejection site knows,
    /// leaving the error untouched when the path is not the Anthropic surface.
    pub fn on_anthropic_surface_if(mut self, path: &str) -> Self {
        self.anthropic_surface = is_anthropic_surface(path);
        self
    }

    pub fn internal_server_error(message: &str) -> Self {
        Self::new(message.to_string(), 500)
    }

    pub fn bad_request(message: &str) -> Self {
        Self::new(message.to_string(), 400)
    }

    pub fn not_found(message: &str) -> Self {
        Self::new(message.to_string(), 404)
    }

    pub fn not_implemented(message: &str) -> Self {
        Self::new(message.to_string(), 501)
    }

    pub fn request_cancelled() -> Self {
        Self::new(ERROR_CANCELLED.to_string(), 499)
    }

    pub fn lm_studio_unavailable(message: &str) -> Self {
        Self::new(message.to_string(), 503)
    }

    pub fn too_many_requests(message: &str) -> Self {
        Self::new(message.to_string(), 429)
    }

    pub fn bad_gateway(message: &str) -> Self {
        Self::new(message.to_string(), 502)
    }

    /// Build the error for a request the web framework refused to extract.
    ///
    /// `detail` is the framework's own body text, which names the offending
    /// header, field or path param — and embeds the client's bytes verbatim.
    /// It answers repr'd so those bytes read as a quoted echo of what was
    /// sent, never as proxy wording. The 413 wording stays the proxy's own
    /// because "failed to buffer the request body" describes the server's
    /// machinery rather than what the client sent.
    fn extraction_rejected(status: StatusCode, detail: String) -> Self {
        let message = if status == StatusCode::PAYLOAD_TOO_LARGE {
            ERROR_BODY_TOO_LARGE.to_string()
        } else {
            format!("{:?}", detail)
        };
        Self::new(message, status.as_u16())
    }

    pub fn is_cancelled(&self) -> bool {
        self.status_code == 499
    }

    pub fn is_lm_studio_unavailable(&self) -> bool {
        self.status_code == 503
    }
}

/// Extraction rejections render themselves as framework-native `text/plain`,
/// which reaches the client before any handler runs and so escapes every
/// surface's error envelope. Converting them to `ProxyError` keeps one error
/// type behind every response the proxy emits.
macro_rules! rejection_into_proxy_error {
    ($($rejection:ty),+ $(,)?) => {$(
        impl From<$rejection> for ProxyError {
            fn from(rejection: $rejection) -> Self {
                Self::extraction_rejected(rejection.status(), rejection.body_text())
            }
        }
    )+};
}

rejection_into_proxy_error!(BytesRejection, JsonRejection, PathRejection, QueryRejection);

impl fmt::Display for ProxyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ProxyError {}: {}", self.status_code, self.message)
    }
}

/// Whether a request path belongs to the Anthropic-compatible surface, whose
/// errors wear Anthropic's envelope instead of Ollama's.
pub fn is_anthropic_surface(path: &str) -> bool {
    path == "/v1/messages" || path.starts_with("/v1/messages/")
}

/// Map an HTTP status to Anthropic's documented error `type` literal.
///
/// The Anthropic surface (`/v1/messages*`) expects
/// `{"type":"error","error":{"type":...,"message":...}}` — SDKs branch on the
/// inner `type`, so unknown statuses collapse to the generic `api_error`.
pub fn anthropic_error_type(status_code: u16) -> &'static str {
    match status_code {
        400 | 405 => "invalid_request_error",
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
        if self.anthropic_surface {
            return self.into_anthropic_response();
        }
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

#[cfg(test)]
#[path = "../tests/unit/error.rs"]
mod tests;
