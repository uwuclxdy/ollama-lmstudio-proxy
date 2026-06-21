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
use crate::lmstudio::keep_alive::unload_other_models;
use crate::lmstudio::{build_load_config_body, is_model_loading_error};
use crate::logging::log_timed;
use crate::model::ModelResolver;

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
    do_explicit_load: bool,
    cancellation_token: CancellationToken,
) -> Result<bool, ProxyError> {
    // `/api/v1/models/load` and the v0 chat-ping below address LM Studio by its
    // bare model key (e.g. "qwen3.5-0.8b"). The Ollama-facing name usually carries
    // a ":latest" tag that LM Studio rejects (`model_not_found` / "no models
    // loaded"), which silently broke every explicit load whenever LM Studio's own
    // JIT was off (and embedders, which only load via this path). Resolve to the
    // key first; best-effort, falling back to the raw name (a truly missing model
    // 404s either way).
    let resolved_key = {
        let cache = moka::future::Cache::builder().max_capacity(64).build();
        let resolver = ModelResolver::new(context.lmstudio_url.to_string(), cache);
        resolver
            .resolve_model_name(
                ollama_model_name,
                context.client,
                cancellation_token.clone(),
            )
            .await
            .unwrap_or_else(|_| ollama_model_name.to_string())
    };
    let model_for_lm_studio_trigger = resolved_key.as_str();

    // Issue a model-only explicit load ONLY when the caller knows the model is
    // not resident (the JIT-on-error path). `build_load_config_body` returns None
    // with no tuning flags, but a bare `{"model": ...}` load still brings up BOTH
    // chat and embedding models — and it is the only thing that loads an embedder
    // (the chat-ping below is invalid for embedders: a client-error, treated as
    // ok). The unconditional warm path (e.g. /api/show) passes `false`: a
    // `/api/v1/models/load` SPAWNS a new instance rather than reconfiguring, so
    // force-loading an already-resident model would stack a duplicate (:2, :3…);
    // the chat-ping JIT-loads chat models idempotently instead.
    if do_explicit_load {
        if get_runtime_config().auto_evict {
            // Best-effort: evict every other model's loaded instances before
            // bringing up the target. Resolution uses a transient resolver (no
            // shared cache) — fine here since the explicit-load path is cold and
            // any failure logs and continues, never aborting the load below.
            let cache = moka::future::Cache::builder().max_capacity(64).build();
            let resolver = ModelResolver::new(context.lmstudio_url.to_string(), cache);
            match resolver
                .resolve_model_name(
                    model_for_lm_studio_trigger,
                    context.client,
                    cancellation_token.clone(),
                )
                .await
            {
                Ok(keep_key) => {
                    if let Err(e) =
                        unload_other_models(context.client, context.lmstudio_url, &keep_key).await
                    {
                        log::warn!("auto-evict: unload failed, continuing: {}", e.message);
                    }
                }
                Err(e) => {
                    log::warn!(
                        "auto-evict: could not resolve keep key for '{}', skipping: {}",
                        model_for_lm_studio_trigger,
                        e.message
                    );
                }
            }
        }

        let load_body =
            build_load_config_body(model_for_lm_studio_trigger, get_runtime_config(), None)
                .unwrap_or_else(|| serde_json::json!({ "model": model_for_lm_studio_trigger }));
        let load_url = context.endpoint_url(LM_STUDIO_MODELS_LOAD);
        let load_request = CancellableRequest::new(context.client, cancellation_token.clone());
        match load_request
            .make_request(reqwest::Method::POST, &load_url, Some(load_body))
            .await
        {
            Ok(_) => {
                // Record the load so /api/ps can report a best-effort expires_at.
                // Keyed on the LM Studio model key (= ModelInfo.id); resolve the
                // Ollama name first so the ps lookup matches. ttl is unknown on
                // the load path (keep_alive lives on the inference request, not
                // here), so None = loaded forever until a keep-alive refresh.
                record_loaded_key(
                    context,
                    model_for_lm_studio_trigger,
                    cancellation_token.clone(),
                )
                .await;
            }
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

            // A successful chat-ping is the actual load for chat models (the
            // explicit-load branch above only runs on the JIT-on-error path).
            // Record it so /api/ps has a real expires_at to report.
            if trigger_considered_successful {
                record_loaded_key(context, ollama_model_name, cancellation_token.clone()).await;
            }
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
    // Unconditional warm (e.g. /api/show): no explicit load — see the duplicate-
    // instance note on `trigger_model_loading`. The chat-ping loads chat models.
    match trigger_model_loading(context, ollama_model_name, false, cancellation_token).await {
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

/// LM Studio's literal "no models loaded" 400 — a model that exists in the
/// catalog but has no resident instance yet. Matched narrowly (rather than via
/// the loose `is_model_loading_error` classifier) so proxy-side validation 400s
/// such as "raw cannot be combined with images ... vision models" don't get
/// mistaken for a load-and-retry signal.
fn is_no_models_loaded(message: &str) -> bool {
    message.to_lowercase().contains("no models loaded")
}

/// Best-effort: resolve `ollama_model_name` to its LM Studio key (= `ModelInfo.id`)
/// and record the load in the proxy's tracker so `/api/ps` can report a real
/// `expires_at`. Resolution uses a transient resolver (no shared cache) — the
/// recorded deadline is approximate anyway and any failure logs and continues,
/// never affecting the load itself. ttl is None (loaded forever) because the
/// load path carries no keep_alive; a subsequent inference request's keep_alive
/// would refresh it if threaded, but that refresh is out of scope here.
async fn record_loaded_key(
    context: &RequestContext<'_>,
    ollama_model_name: &str,
    cancellation_token: CancellationToken,
) {
    let cache = moka::future::Cache::builder().max_capacity(64).build();
    let resolver = ModelResolver::new(context.lmstudio_url.to_string(), cache);
    match resolver
        .resolve_model_name(
            ollama_model_name,
            context.client,
            cancellation_token.clone(),
        )
        .await
    {
        Ok(lm_key) => context
            .load_tracker
            .record(&lm_key, crate::model::load_tracker::KeepAlive::Unknown),
        Err(e) => log::debug!(
            "load-tracker: could not resolve key for '{}', skipping record: {}",
            ollama_model_name,
            e.message
        ),
    }
}

/// Whether an upstream failure should trigger a best-effort model load and retry.
///
/// True for a 400 carrying LM Studio's "no models loaded" (a model that exists
/// but isn't resident) or for a 5xx loading error (still spinning up). Other 4xx
/// return verbatim: a 404 "model not found" means the model truly doesn't exist,
/// so a load would be futile. 429/502 are forwarded by the caller and never
/// reach here, so they are excluded outright.
pub(crate) fn should_trigger_load(status: u16, message: &str) -> bool {
    if status == 429 || status == 502 {
        return false;
    }
    if (400..500).contains(&status) {
        return is_no_models_loaded(message);
    }
    is_model_loading_error(message)
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
        // Most 4xx return immediately. The one exception is a 400 carrying LM
        // Studio's literal "no models loaded" — a model that exists but isn't
        // resident yet — which must fall through to the trigger-and-retry arm.
        // A 404 ("model not found") still returns at once (the model genuinely
        // doesn't exist), and proxy-side validation 400s (e.g. raw + images) are
        // NOT loading errors even when the loose classifier would match them.
        Err(e) if (400..500).contains(&e.status_code) && !is_no_models_loaded(&e.message) => Err(e),
        // 429/502 are passed through verbatim (their messages can match the
        // loading classifier — e.g. a 502 "model unreachable" — so guard them
        // here to avoid a spurious load detour before returning).
        Err(e) if e.status_code == 429 || e.status_code == 502 => Err(e),
        Err(e) => {
            if should_trigger_load(e.status_code, &e.message) {
                let model_loading_start = Instant::now();
                log_timed(
                    LOG_PREFIX_INFO,
                    &format!("{} not loaded, triggering", ollama_model_name),
                    model_loading_start,
                );

                // JIT-on-error: the model is genuinely not resident, so force an
                // explicit load (required to bring up embedding models).
                match trigger_model_loading(
                    context,
                    ollama_model_name,
                    true,
                    cancellation_token.clone(),
                )
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
