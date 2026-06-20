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

use crate::constants::{DEFAULT_STREAM_TIMEOUT_SECONDS, LOG_PREFIX_SUCCESS};
use crate::error::ProxyError;
use crate::lmstudio::keep_alive::spawn_model_unload_if_needed;
use crate::logging::log_timed;
use crate::model::ModelResolver;
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
        } = self;

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
            let ttl = keep_alive_seconds
                .filter(|&s| s > 0)
                .map(|s| std::time::Duration::from_secs(s as u64));
            match resolver
                .resolve_model_name(&ollama_model_name, context.client, cancellation.clone())
                .await
            {
                Ok(lm_key) => context.load_tracker.record(&lm_key, ttl),
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
