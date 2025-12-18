use crate::constants::{ERROR_LM_STUDIO_UNAVAILABLE, ERROR_TIMEOUT};
use crate::error::ProxyError;

pub fn map_reqwest_error(err: reqwest::Error) -> ProxyError {
    if err.is_connect() {
        ProxyError::lm_studio_unavailable(ERROR_LM_STUDIO_UNAVAILABLE)
    } else if err.is_timeout() {
        ProxyError::lm_studio_unavailable(ERROR_TIMEOUT)
    } else {
        log::error!("HTTP request failed: {}", err);
        ProxyError::internal_server_error(&format!("LM Studio request failed: {}", err))
    }
}
