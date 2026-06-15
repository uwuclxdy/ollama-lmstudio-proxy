use std::future::Future;
use std::time::{Duration, Instant};

use serde::Serialize;
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;

use crate::api::RequestContext;
use crate::check_cancelled;
use crate::config::get_runtime_config;
use crate::constants::{
    ERROR_LM_STUDIO_UNAVAILABLE, LM_STUDIO_MODELS_LOAD, LM_STUDIO_NATIVE_CHAT, LOG_PREFIX_INFO,
    LOG_PREFIX_SUCCESS,
};
use crate::error::ProxyError;
use crate::http::CancellableRequest;
use crate::lmstudio::{build_load_config_body, is_model_loading_error};
use crate::logging::log_timed;

#[derive(Serialize)]
struct MinimalChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct MinimalChatRequestPayload<'a> {
    model: &'a str,
    messages: Vec<MinimalChatMessage<'a>>,
    max_tokens: u32,
    stream: bool,
}

pub async fn trigger_model_loading(
    context: &RequestContext<'_>,
    ollama_model_name: &str,
    cancellation_token: CancellationToken,
) -> Result<bool, ProxyError> {
    let model_for_lm_studio_trigger = ollama_model_name;

    if let Some(load_body) =
        build_load_config_body(model_for_lm_studio_trigger, get_runtime_config(), None)
    {
        let load_url = context.endpoint_url(LM_STUDIO_MODELS_LOAD);
        let load_request = CancellableRequest::new(context.client, cancellation_token.clone());
        match load_request
            .make_request(reqwest::Method::POST, &load_url, Some(load_body))
            .await
        {
            Ok(_) => {}
            Err(e) if e.is_cancelled() => return Err(ProxyError::request_cancelled()),
            Err(e) => {
                log::warn!("model load config: best-effort POST failed: {}", e.message);
            }
        }
    }

    let url = context.endpoint_url(LM_STUDIO_NATIVE_CHAT);
    let minimal_request_body = MinimalChatRequestPayload {
        model: model_for_lm_studio_trigger,
        messages: vec![MinimalChatMessage {
            role: "user",
            content: "ping",
        }],
        max_tokens: 1,
        stream: false,
    };

    let request = CancellableRequest::new(context.client, cancellation_token.clone());

    match request
        .make_request(reqwest::Method::POST, &url, Some(minimal_request_body))
        .await
    {
        Ok(response) => {
            let status = response.status();
            let trigger_considered_successful = status.is_success() || status.is_client_error();

            if !trigger_considered_successful {
                log::warn!("model trigger: status: {}", status);
            }
            Ok(trigger_considered_successful)
        }
        Err(e) if e.is_cancelled() => Err(ProxyError::request_cancelled()),
        Err(e) if e.is_lm_studio_unavailable() => Err(ProxyError::lm_studio_unavailable(
            ERROR_LM_STUDIO_UNAVAILABLE,
        )),
        Err(e) => {
            log::error!("model trigger: {}", e.message);
            Ok(false)
        }
    }
}

pub async fn trigger_model_loading_for_ollama(
    context: &RequestContext<'_>,
    ollama_model_name: &str,
    cancellation_token: CancellationToken,
) -> Result<(), ProxyError> {
    match trigger_model_loading(context, ollama_model_name, cancellation_token).await {
        Ok(true) => Ok(()),
        Ok(false) => {
            log::warn!(
                "load hint: trigger for '{}' failed, proceeding",
                ollama_model_name
            );
            Ok(())
        }
        Err(e) => Err(e),
    }
}

pub async fn with_retry_and_cancellation<F, Fut, T>(
    context: &RequestContext<'_>,
    ollama_model_name: &str,
    load_timeout_seconds: u64,
    operation: F,
    cancellation_token: CancellationToken,
) -> Result<T, ProxyError>
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<T, ProxyError>>,
{
    check_cancelled!(cancellation_token);

    match operation().await {
        Ok(result) => Ok(result),
        Err(e) if e.is_cancelled() => Err(ProxyError::request_cancelled()),
        Err(e) if e.is_lm_studio_unavailable() => {
            log::error!("request failed: LM Studio unavailable - failing fast");
            Err(e)
        }
        Err(e) if (400..500).contains(&e.status_code) => Err(e),
        Err(e) => {
            if is_model_loading_error(&e.message) {
                let model_loading_start = Instant::now();
                log_timed(
                    LOG_PREFIX_INFO,
                    &format!("{} not loaded, triggering", ollama_model_name),
                    model_loading_start,
                );

                match trigger_model_loading(context, ollama_model_name, cancellation_token.clone())
                    .await
                {
                    Ok(true) => {
                        tokio::select! {
                            _ = sleep(Duration::from_secs(load_timeout_seconds)) => {},
                            _ = cancellation_token.cancelled() => {
                                return Err(ProxyError::request_cancelled());
                            }
                        }
                        check_cancelled!(cancellation_token);

                        match operation().await {
                            Ok(result) => {
                                log_timed(
                                    LOG_PREFIX_SUCCESS,
                                    &format!("{} loaded", ollama_model_name),
                                    model_loading_start,
                                );
                                Ok(result)
                            }
                            Err(retry_error) => {
                                log::error!(
                                    "retry failed for {}: {}",
                                    ollama_model_name,
                                    retry_error.message
                                );
                                Err(e)
                            }
                        }
                    }
                    Ok(false) => {
                        log::error!(
                            "model trigger: failed for {} - model may not exist. Original: {}",
                            ollama_model_name,
                            e.message
                        );
                        Err(e)
                    }
                    Err(loading_trigger_error) => {
                        log::error!("model trigger error: {}", loading_trigger_error.message);
                        Err(loading_trigger_error)
                    }
                }
            } else {
                Err(e)
            }
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/handlers_retry.rs"]
mod tests;
