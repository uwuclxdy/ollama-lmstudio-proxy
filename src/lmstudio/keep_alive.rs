//! Keep-alive translation and explicit model-unload spawning.
//!
//! Ollama keep_alive accepts:
//!   - integer N: seconds (0 = unload now; negative = stay forever)
//!   - string: Go-style duration ("5m", "1h30m", "500ms", optionally negative)
//!
//! LM Studio's `ttl` field is a non-negative seconds count; we normalize
//! negative values to a `-1` sentinel and omit `ttl` from the request,
//! letting LM Studio's default ("loaded indefinitely") apply.

use std::sync::Arc;
use std::time::Duration;

use serde_json::{Value, json};
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use crate::constants::{LM_STUDIO_NATIVE_MODELS, LM_STUDIO_NATIVE_UNLOAD};
use crate::error::ProxyError;
use crate::model::ModelResolver;
use crate::model::types::NativeModelsResponse;

const FOREVER_SENTINEL: i64 = -1;

pub fn parse_keep_alive_seconds(raw_value: Option<&Value>) -> Result<Option<i64>, ProxyError> {
    let Some(value) = raw_value else {
        return Ok(None);
    };

    match value {
        Value::Null => Ok(None),
        Value::Number(num) => parse_numeric(num),
        Value::String(text) => parse_string(text),
        _ => Err(ProxyError::bad_request(
            "invalid keep_alive value. Use numeric seconds or durations like '5m'",
        )),
    }
}

fn parse_numeric(num: &serde_json::Number) -> Result<Option<i64>, ProxyError> {
    if let Some(signed) = num.as_i64() {
        return Ok(Some(normalize(signed)));
    }
    if let Some(unsigned) = num.as_u64() {
        if unsigned <= i64::MAX as u64 {
            return Ok(Some(unsigned as i64));
        }
        return Err(ProxyError::bad_request(
            "keep_alive value exceeds supported range",
        ));
    }
    // Float-valued keep_alive (e.g. 30.0 from Home Assistant / Python clients).
    // Ollama itself accepts float64; truncate to whole seconds and clamp any
    // positive sub-second value up to 1 so it can't accidentally trigger unload.
    if let Some(float) = num.as_f64() {
        if !float.is_finite() {
            return Err(ProxyError::bad_request(
                "keep_alive must be a finite number",
            ));
        }
        if float > 0.0 && float < 1.0 {
            return Ok(Some(1));
        }
        if float < 0.0 {
            return Ok(Some(FOREVER_SENTINEL));
        }
        if float > i64::MAX as f64 {
            return Err(ProxyError::bad_request(
                "keep_alive value exceeds supported range",
            ));
        }
        return Ok(Some(normalize(float.trunc() as i64)));
    }
    Err(ProxyError::bad_request("keep_alive must be integral"))
}

fn parse_string(text: &str) -> Result<Option<i64>, ProxyError> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }

    // Explicit negative durations: "-1", "-1s", "-30m", ...
    if let Some(rest) = trimmed.strip_prefix('-') {
        // Go's ParseDuration consumes at most one sign, so a rest that still
        // starts with one ("--1h", "-+1h", "-+1") is invalid, never negative.
        if rest.starts_with(['-', '+']) {
            return Err(ProxyError::bad_request(
                "invalid keep_alive value. Use numeric seconds or durations like '5m'",
            ));
        }
        if rest.parse::<i64>().is_ok() || parse_go_duration(rest).is_some() {
            return Ok(Some(FOREVER_SENTINEL));
        }
        return Err(ProxyError::bad_request(
            "invalid keep_alive value. Use numeric seconds or durations like '5m'",
        ));
    }

    if let Some(duration) = parse_go_duration(trimmed) {
        let secs = duration.as_secs();
        // Sub-second durations must NOT round down to 0 (would trigger unload).
        let effective = if duration.is_zero() {
            0
        } else if secs == 0 {
            1
        } else if secs > i64::MAX as u64 {
            return Err(ProxyError::bad_request(
                "keep_alive duration exceeds supported range",
            ));
        } else {
            secs as i64
        };
        return Ok(Some(effective));
    }

    trimmed
        .parse::<i64>()
        .map(|n| Some(normalize(n)))
        .map_err(|_| {
            ProxyError::bad_request(
                "invalid keep_alive value. Use numeric seconds or durations like '5m'",
            )
        })
}

/// Parses the Go duration grammar (Go's `src/time/format.go`, what Ollama uses
/// for the string form of keep_alive): `[-+]?([0-9]*(\.[0-9]*)?[a-z]+)+`, an
/// optional sign then one or more elements of `digits [ "." digits? ] unit`
/// (at least one digit on one side of the point, so `.5s` and `1.s` parse) with
/// units `ns`, `us`/`µs`/`μs`, `ms`, `s`, `m`, `h`; `"0"` after the sign is
/// zero. The sign is consumed but dropped — the caller maps `-` to the forever
/// sentinel before this runs — and a magnitude past Go's `int64` nanosecond
/// range is rejected, so the result fits `i64` seconds.
///
/// The fraction accumulates as an exact integer ratio, not Go's float64
/// `f/scale`; where float64 rounds a contribution up across a nanosecond
/// boundary, this truncates one nanosecond lower (`"1.999999999999999999s"` is
/// 1 s here, 2 s in Go), so second truncation can read one second lower. A
/// 2026-11-17 differential run over 171,238 inputs found 12 divergences, all
/// `N.999999999999999999s`-shaped. A fraction small enough to avoid the
/// numerator freeze through 39+ digits saturates the integer scale,
/// contributing 0 ns — the value Go's float64 scale yields for it.
fn parse_go_duration(text: &str) -> Option<Duration> {
    let bytes = text.as_bytes();
    // Consume [-+]?
    let mut rest = match bytes.split_first() {
        Some((b'-' | b'+', tail)) => tail,
        _ => bytes,
    };
    // Special case: if all that is left is "0", this is zero.
    if rest == b"0" {
        return Some(Duration::ZERO);
    }
    if rest.is_empty() {
        return None;
    }

    let mut total_ns: u128 = 0;
    while !rest.is_empty() {
        // The next character must be [0-9.]
        if !rest[0].is_ascii_digit() && rest[0] != b'.' {
            return None;
        }

        // Consume [0-9]*, bounded like Go's leadingInt: overflowing int64 is
        // an error there, and any element this large would overflow the total
        // anyway.
        let mut int_part = 0u128;
        let mut i = 0;
        while i < rest.len() && rest[i].is_ascii_digit() {
            let digit = u128::from(rest[i] - b'0');
            if int_part > i64::MAX as u128 / 10 {
                return None;
            }
            int_part = int_part * 10 + digit;
            if int_part > (i64::MAX as u128) + 1 {
                return None;
            }
            i += 1;
        }
        let pre = i > 0;
        rest = &rest[i..];

        // Consume (\.[0-9]*)?; like Go's leadingFraction, numerator and scale
        // freeze once the fraction would overflow int64.
        let mut frac = 0u128;
        let mut scale = 1u128;
        let mut post = false;
        let mut overflow = false;
        if let Some(tail) = rest.strip_prefix(b".") {
            rest = tail;
            let mut j = 0;
            while j < rest.len() && rest[j].is_ascii_digit() {
                let digit = u128::from(rest[j] - b'0');
                if !overflow {
                    if frac > i64::MAX as u128 / 10 {
                        overflow = true;
                    } else {
                        let next = frac * 10 + digit;
                        if next > (i64::MAX as u128) + 1 {
                            overflow = true;
                        } else {
                            frac = next;
                            // No numerator freeze is in effect when the scale
                            // would overflow, so saturate: a saturated scale
                            // contributes 0 ns, Go's value for a fraction
                            // that long.
                            scale = scale.saturating_mul(10);
                        }
                    }
                }
                j += 1;
            }
            post = j > 0;
            rest = &rest[j..];
        }

        if !pre && !post {
            // no digits (e.g. ".s" or "-.s")
            return None;
        }

        // Consume the unit: everything up to the next digit or '.'.
        let unit_end = rest
            .iter()
            .position(|&b| b.is_ascii_digit() || b == b'.')
            .unwrap_or(rest.len());
        if unit_end == 0 {
            return None;
        }
        let unit = &rest[..unit_end];
        rest = &rest[unit_end..];
        let unit_ns: u128 = match unit {
            b"ns" => 1,
            b"us" | b"\xC2\xB5s" | b"\xCE\xBCs" => 1_000,
            b"ms" => 1_000_000,
            b"s" => 1_000_000_000,
            b"m" => 60_000_000_000,
            b"h" => 3_600_000_000_000,
            _ => return None,
        };

        total_ns += int_part * unit_ns + frac * unit_ns / scale;
        if total_ns > i64::MAX as u128 {
            return None;
        }
    }

    Some(Duration::from_nanos(u64::try_from(total_ns).ok()?))
}

fn normalize(secs: i64) -> i64 {
    if secs < 0 { FOREVER_SENTINEL } else { secs }
}

pub fn apply_keep_alive_ttl(target: &mut Value, keep_alive_seconds: Option<i64>) {
    let Some(ttl) = keep_alive_seconds else {
        return;
    };
    if ttl < 0 {
        // "stay loaded forever" → omit ttl (LM Studio default applies)
        return;
    }
    if let Some(obj) = target.as_object_mut() {
        obj.insert("ttl".to_string(), Value::from(ttl));
    }
}

pub fn keep_alive_requests_unload(ttl: Option<i64>) -> bool {
    matches!(ttl, Some(value) if value == 0)
}

/// Spawns a background task to explicitly unload the model via LMStudio's native
/// unload endpoint when `keep_alive: 0` is requested.
///
/// `stream_end` defers the unload until a streaming response body reaches its
/// own end (all chunks delivered) or is dropped (client disconnect): the
/// receiver resolves when the body ends, whichever way. `None` unloads
/// immediately, for a non-streaming response that is complete when the handler
/// returns.
pub fn spawn_model_unload_if_needed(
    client: reqwest::Client,
    base_url: String,
    model_resolver: Arc<ModelResolver>,
    ollama_model_name: String,
    keep_alive_seconds: Option<i64>,
    stream_end: Option<oneshot::Receiver<()>>,
) {
    if !matches!(keep_alive_seconds, Some(0)) {
        return;
    }

    tokio::spawn(async move {
        if let Some(stream_end) = stream_end {
            // The sender is dropped with the body, so a client disconnect
            // resolves this too; the payload is empty.
            let _ = stream_end.await;
        }
        if let Err(e) =
            unload_model_instances(&client, &base_url, &model_resolver, &ollama_model_name).await
        {
            log::warn!(
                "model unload failed for '{}': {}",
                ollama_model_name,
                e.message
            );
        }
    });
}

async fn unload_model_instances(
    client: &reqwest::Client,
    base_url: &str,
    model_resolver: &Arc<ModelResolver>,
    ollama_model_name: &str,
) -> Result<(), ProxyError> {
    let lm_studio_id = model_resolver
        .resolve_model_name(ollama_model_name, client, CancellationToken::new())
        .await?;

    let models_url = format!("{}{}", base_url, LM_STUDIO_NATIVE_MODELS);
    let native: NativeModelsResponse = client
        .get(&models_url)
        .send()
        .await
        .map_err(|e| {
            ProxyError::internal_server_error(&format!("unload: fetch models failed: {e}"))
        })?
        .json()
        .await
        .map_err(|e| {
            ProxyError::internal_server_error(&format!("unload: parse models response failed: {e}"))
        })?;

    let unload_url = format!("{}{}", base_url, LM_STUDIO_NATIVE_UNLOAD);
    for model in native.models.iter().filter(|m| m.key == lm_studio_id) {
        for instance in &model.loaded_instances {
            match client
                .post(&unload_url)
                .json(&json!({ "instance_id": instance.id }))
                .send()
                .await
                .and_then(|r| r.error_for_status())
            {
                Ok(_) => log::debug!("unloaded model instance '{}'", instance.id),
                Err(e) => log::warn!("failed to unload instance '{}': {}", instance.id, e),
            }
        }
    }

    Ok(())
}

/// Core eviction logic operating on an already-fetched models list.
///
/// Unloads every loaded instance of every model whose `key` != `keep_model_key`.
/// Best-effort: per-instance unload failures are logged and skipped.
pub async fn unload_instances_except(
    client: &reqwest::Client,
    models: &NativeModelsResponse,
    keep_model_key: &str,
    unload_url: &str,
) {
    for model in models
        .models
        .iter()
        .filter(|m| m.key != keep_model_key && !m.loaded_instances.is_empty())
    {
        for instance in &model.loaded_instances {
            match client
                .post(unload_url)
                .json(&json!({ "instance_id": instance.id }))
                .send()
                .await
                .and_then(|r| r.error_for_status())
            {
                Ok(_) => log::debug!(
                    "auto-evict: unloaded instance '{}' of '{}'",
                    instance.id,
                    model.key
                ),
                Err(e) => log::warn!(
                    "auto-evict: failed to unload instance '{}' of '{}': {}",
                    instance.id,
                    model.key,
                    e
                ),
            }
        }
    }
}

/// Best-effort: unload every OTHER model's loaded instances (those whose
/// `key` != `keep_model_key`) before a fresh load, mirroring Ollama's
/// single-model default + LM Studio's JIT auto-evict. Never propagates a
/// failure that would abort the caller's load: each per-instance error is
/// logged and skipped, and the function returns `Ok(())` after best-effort
/// attempts.
pub async fn unload_other_models(
    client: &reqwest::Client,
    base_url: &str,
    keep_model_key: &str,
) -> Result<(), ProxyError> {
    let models_url = format!("{}{}", base_url, LM_STUDIO_NATIVE_MODELS);
    let native: NativeModelsResponse = match client.get(&models_url).send().await {
        Ok(response) => match response.json().await {
            Ok(parsed) => parsed,
            Err(e) => {
                log::warn!("auto-evict: parse models response failed, skipping: {}", e);
                return Ok(());
            }
        },
        Err(e) => {
            log::warn!("auto-evict: fetch models failed, skipping: {}", e);
            return Ok(());
        }
    };

    let unload_url = format!("{}{}", base_url, LM_STUDIO_NATIVE_UNLOAD);
    unload_instances_except(client, &native, keep_model_key, &unload_url).await;
    Ok(())
}

/// Proactive eviction: if `keep_model_key` is NOT currently loaded (no
/// `loaded_instances`), unload all other models' instances and return `true`
/// (eviction ran). If the target is already loaded, return `false` (nothing to
/// do). Always best-effort — any fetch/parse failure logs and returns `false`.
///
/// Takes ownership of an already-fetched `NativeModelsResponse` so the caller
/// can share the single GET with its own logic (e.g. checking whether the
/// target needs a context-length reload) without a redundant round trip.
pub async fn proactive_evict_if_unloaded(
    client: &reqwest::Client,
    models: &NativeModelsResponse,
    keep_model_key: &str,
    unload_url: &str,
) -> bool {
    let target_loaded = models
        .models
        .iter()
        .find(|m| m.key == keep_model_key)
        .map(|m| !m.loaded_instances.is_empty())
        .unwrap_or(false);

    if target_loaded {
        return false;
    }

    unload_instances_except(client, models, keep_model_key, unload_url).await;
    true
}

#[cfg(test)]
#[path = "../../tests/unit/lmstudio_keep_alive_spawn.rs"]
mod tests_spawn;

#[cfg(test)]
#[path = "../../tests/unit/lmstudio_keep_alive_parse.rs"]
mod tests_parse;
