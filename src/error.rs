use std::error::Error;
use std::fmt;

use warp::reject::Reject;

use crate::constants::ERROR_CANCELLED;

/// Error type for the proxy server
#[derive(Debug, Clone)]
pub struct ProxyError {
    pub message: String,
    pub status_code: u16,
    kind: ProxyErrorKind,
}

#[derive(Debug, Clone)]
enum ProxyErrorKind {
    RequestCancelled,
    InternalServerError,
    BadRequest,
    NotFound,
    NotImplemented,
    LMStudioUnavailable,
    Custom,
}

impl ProxyError {
    pub fn new(message: String, status_code: u16) -> Self {
        Self {
            message,
            status_code,
            kind: ProxyErrorKind::Custom,
        }
    }

    pub fn internal_server_error(message: &str) -> Self {
        Self {
            message: message.to_string(),
            status_code: 500,
            kind: ProxyErrorKind::InternalServerError,
        }
    }

    pub fn bad_request(message: &str) -> Self {
        Self {
            message: message.to_string(),
            status_code: 400,
            kind: ProxyErrorKind::BadRequest,
        }
    }

    pub fn not_found(message: &str) -> Self {
        Self {
            message: message.to_string(),
            status_code: 404,
            kind: ProxyErrorKind::NotFound,
        }
    }

    pub fn not_implemented(message: &str) -> Self {
        Self {
            message: message.to_string(),
            status_code: 501,
            kind: ProxyErrorKind::NotImplemented,
        }
    }

    pub fn request_cancelled() -> Self {
        Self {
            message: ERROR_CANCELLED.to_string(),
            status_code: 499,
            kind: ProxyErrorKind::RequestCancelled,
        }
    }

    pub fn lm_studio_unavailable(message: &str) -> Self {
        Self {
            message: message.to_string(),
            status_code: 503,
            kind: ProxyErrorKind::LMStudioUnavailable,
        }
    }

    pub fn is_cancelled(&self) -> bool {
        matches!(self.kind, ProxyErrorKind::RequestCancelled)
    }

    pub fn is_lm_studio_unavailable(&self) -> bool {
        matches!(self.kind, ProxyErrorKind::LMStudioUnavailable)
    }
}

impl fmt::Display for ProxyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ProxyError {}: {}", self.status_code, self.message)
    }
}

impl Error for ProxyError {}

impl Reject for ProxyError {}

pub fn is_model_loading_error(message: &str) -> bool {
    let lower = message.to_lowercase();

    let loading_indicators = [
        "loading model",
        "model loading",
        "model is loading",
        "loading the model",
        "model not loaded",
        "not loaded",
        "model unavailable",
        "model not available",
        "model not found",
        "no model",
        "invalid model",
        "unknown model",
        "failed to load",
        "loading failed",
        "model error",
        "is not embedding",
        "model initialization",
        "initializing model",
        "warming up model",
        "model startup",
        "preparing model",
        "model not ready",
    ];

    if loading_indicators
        .iter()
        .any(|&pattern| lower.contains(pattern))
    {
        return true;
    }

    let has_negative = lower.contains("no")
        || lower.contains("not")
        || lower.contains("missing")
        || lower.contains("invalid")
        || lower.contains("unknown")
        || lower.contains("failed")
        || lower.contains("unavailable")
        || lower.contains("unreachable");

    let has_model_ref = lower.contains("model")
        || lower.contains("load")
        || lower.contains("available")
        || lower.contains("ready")
        || lower.contains("initialize");

    let lm_studio_loading_patterns = [
        "service unavailable",
        "server error",
        "internal error",
        "timeout",
        "connection",
        "503",
        "500",
    ];

    let has_lm_studio_loading = lm_studio_loading_patterns
        .iter()
        .any(|&pattern| lower.contains(pattern));

    (has_negative && has_model_ref) || has_lm_studio_loading
}

#[macro_export]
macro_rules! check_cancelled {
    ($token:expr) => {
        if $token.is_cancelled() {
            return Err($crate::error::ProxyError::request_cancelled());
        }
    };
}
