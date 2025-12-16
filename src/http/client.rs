use serde::Serialize;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::check_cancelled;
use crate::constants::{CONTENT_TYPE_JSON, ERROR_LM_STUDIO_UNAVAILABLE};
use crate::error::ProxyError;

pub struct CancellableRequest<'a> {
    client: &'a reqwest::Client,
    token: CancellationToken,
}

impl<'a> CancellableRequest<'a> {
    pub fn new(client: &'a reqwest::Client, token: CancellationToken) -> Self {
        Self { client, token }
    }

    pub async fn make_request<B: Serialize>(
        &self,
        method: reqwest::Method,
        url: &str,
        body: Option<B>,
    ) -> Result<reqwest::Response, ProxyError> {
        check_cancelled!(self.token);

        let mut request_builder = self.client.request(method, url);

        if let Some(body_content) = body {
            request_builder = request_builder
                .header("Content-Type", CONTENT_TYPE_JSON)
                .json(&body_content);
        }

        tokio::select! {
            result = request_builder.send() => {
                match result {
                    Ok(response) => Ok(response),
                    Err(err) => {
                        let error_msg = if err.is_connect() {
                            ERROR_LM_STUDIO_UNAVAILABLE
                        } else if err.is_request() {
                            "invalid request"
                        } else {
                            "request failed"
                        };
                        log::error!("HTTP request failed: {}: {:?}", error_msg, err);
                        Err(ProxyError::internal_server_error(error_msg))
                    }
                }
            }
            _ = self.token.cancelled() => {
                Err(ProxyError::request_cancelled())
            }
        }
    }
}

pub async fn handle_json_response(
    response: reqwest::Response,
    cancellation_token: CancellationToken,
) -> Result<Value, ProxyError> {
    check_cancelled!(cancellation_token);

    let status = response.status();
    let is_error = !status.is_success();

    tokio::select! {
        result = response.json::<Value>() => {
            match result {
                Ok(json_value) => {
                    if is_error {
                        let error_message = match json_value.get("error") {
                            Some(Value::Object(obj)) => obj
                                .get("message")
                                .and_then(|m| m.as_str())
                                .map(|s| s.to_string()),
                            Some(Value::String(message)) => Some(message.clone()),
                            _ => None,
                        }
                        .unwrap_or_else(|| format!("LM Studio error: {}", status));
                        Err(ProxyError::new(error_message, status.as_u16()))
                    } else {
                        Ok(json_value)
                    }
                }
                Err(e) => {
                    Err(ProxyError::internal_server_error(&format!(
                        "invalid JSON from LM Studio: {}", e
                    )))
                }
            }
        }
        _ = cancellation_token.cancelled() => {
            Err(ProxyError::request_cancelled())
        }
    }
}
