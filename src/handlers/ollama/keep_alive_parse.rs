//! Pure helpers for keep_alive translation.
//!
//! Ollama keep_alive accepts:
//!   - integer N: seconds (0 = unload now; negative = stay forever)
//!   - string: Go-style duration ("5m", "1h30m", "500ms", optionally negative)
//!
//! LM Studio's `ttl` field is a non-negative seconds count; we normalize
//! negative values to a `-1` sentinel and omit `ttl` from the request,
//! letting LM Studio's default ("loaded indefinitely") apply.

use humantime::parse_duration;
use serde_json::Value;

use crate::error::ProxyError;

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
    Err(ProxyError::bad_request("keep_alive must be integral"))
}

fn parse_string(text: &str) -> Result<Option<i64>, ProxyError> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }

    // Explicit negative durations: "-1", "-1s", "-30m", ...
    if let Some(rest) = trimmed.strip_prefix('-') {
        if rest.parse::<i64>().is_ok() || parse_duration(rest).is_ok() {
            return Ok(Some(FOREVER_SENTINEL));
        }
        return Err(ProxyError::bad_request(
            "invalid keep_alive value. Use numeric seconds or durations like '5m'",
        ));
    }

    if let Ok(duration) = parse_duration(trimmed) {
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

#[cfg(test)]
#[path = "../../../tests/unit/handlers_ollama_keep_alive_parse.rs"]
mod tests;
