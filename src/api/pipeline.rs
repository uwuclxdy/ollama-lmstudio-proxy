//! Shared scaffold for chat / generate / embeddings handlers.
//!
//! Each handler runs the same outer pattern: extract model + keep_alive, run the
//! request inside the retry-with-cancellation wrapper, then (for chat/generate)
//! kick off a background unload if `keep_alive: 0` was requested, and finally
//! log the elapsed time. Only the closure body between resolution and response
//! transformation differs between handlers. That body stays in each handler;
//! everything around it lives here.

use std::future::Future;
use std::sync::Arc;
use std::time::Instant;

use axum::response::Response;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::constants::{
    DEFAULT_STREAM_TIMEOUT_SECONDS, LM_STUDIO_NATIVE_MODELS, LM_STUDIO_NATIVE_UNLOAD,
    LOG_PREFIX_SUCCESS,
};
use crate::error::ProxyError;
use crate::lmstudio::keep_alive::{proactive_evict_if_unloaded, spawn_model_unload_if_needed};
use crate::logging::log_timed;
use crate::model::ModelResolver;
use crate::model::types::NativeModelsResponse;
use crate::streaming::is_streaming_request;

use super::context::RequestContext;
use super::retry::with_retry_and_cancellation;

pub struct ChatLikeCall<'a> {
    pub context: RequestContext<'a>,
    pub resolver: Arc<ModelResolver>,
    pub body: Value,
    pub cancellation: CancellationToken,
    pub load_timeout_seconds: u64,
    pub ollama_model_name: String,
    pub keep_alive_seconds: Option<i64>,
    pub start_time: Instant,
    pub op_label: &'static str,
    pub spawn_unload: bool,
    /// When true, proactively evict other models before the inference attempt
    /// if the target model is not yet loaded. Mirror of `Config.auto_evict`
    /// (NOT the global RuntimeConfig — that's frozen at first test init).
    pub auto_evict: bool,
}

impl ChatLikeCall<'_> {
    pub async fn run<F, Fut>(self, attempt: F) -> Result<Response, ProxyError>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = Result<Response, ProxyError>>,
    {
        let ChatLikeCall {
            context,
            resolver,
            body,
            cancellation,
            load_timeout_seconds,
            ollama_model_name,
            keep_alive_seconds,
            start_time,
            op_label,
            spawn_unload,
            auto_evict,
        } = self;

        // Proactive auto-evict: ONE GET /api/v1/models before inference.
        // Runs only when the flag is set AND the target isn't already loaded —
        // no thrash on warm requests, zero overhead when flag is off.
        // The unload-only path (keep_alive:0, no prompt/messages) is
        // short-circuited before ChatLikeCall::run is ever called, so no guard
        // needed here.
        if auto_evict {
            let models_url = format!("{}{}", context.lmstudio_url, LM_STUDIO_NATIVE_MODELS);
            let unload_url = format!("{}{}", context.lmstudio_url, LM_STUDIO_NATIVE_UNLOAD);
            match context.client.get(&models_url).send().await {
                Ok(resp) => match resp.json::<NativeModelsResponse>().await {
                    Ok(models) => {
                        match resolver
                            .resolve_model_name(
                                &ollama_model_name,
                                context.client,
                                cancellation.clone(),
                            )
                            .await
                        {
                            Ok(lm_key) => {
                                proactive_evict_if_unloaded(
                                    context.client,
                                    &models,
                                    &lm_key,
                                    &unload_url,
                                )
                                .await;
                            }
                            Err(e) => log::debug!(
                                "auto-evict: could not resolve key for '{}', skipping: {}",
                                ollama_model_name,
                                e.message
                            ),
                        }
                    }
                    Err(e) => {
                        log::warn!("auto-evict: parse models response failed, skipping: {}", e)
                    }
                },
                Err(e) => log::warn!("auto-evict: fetch models failed, skipping: {}", e),
            }
        }

        let result = with_retry_and_cancellation(
            &context,
            &ollama_model_name,
            load_timeout_seconds,
            attempt,
            cancellation.clone(),
        )
        .await?;

        // Refresh the load tracker with this request's resolved keep_alive so
        // /api/ps can report an accurate expires_at. Best-effort: resolution is
        // cached on the shared resolver and any failure is ignored. Skipped for
        // keep_alive:0 (the model is torn down by spawn_unload below).
        if keep_alive_seconds != Some(0) {
            let intent = match keep_alive_seconds {
                Some(s) if s < 0 => crate::model::load_tracker::KeepAlive::Forever,
                Some(s) if s > 0 => crate::model::load_tracker::KeepAlive::Finite(
                    std::time::Duration::from_secs(s as u64),
                ),
                _ => crate::model::load_tracker::KeepAlive::Unknown,
            };
            match resolver
                .resolve_model_name(&ollama_model_name, context.client, cancellation.clone())
                .await
            {
                Ok(lm_key) => context.load_tracker.record(&lm_key, intent),
                Err(e) => log::debug!(
                    "load-tracker: could not resolve key for '{}', skipping: {}",
                    ollama_model_name,
                    e.message
                ),
            }
        }

        if spawn_unload {
            let unload_delay = if is_streaming_request(&body) {
                DEFAULT_STREAM_TIMEOUT_SECONDS
            } else {
                0
            };
            spawn_model_unload_if_needed(
                context.client.clone(),
                context.lmstudio_url.to_string(),
                resolver,
                ollama_model_name,
                keep_alive_seconds,
                unload_delay,
            );
        }

        log_timed(LOG_PREFIX_SUCCESS, op_label, start_time);
        Ok(result)
    }
}
