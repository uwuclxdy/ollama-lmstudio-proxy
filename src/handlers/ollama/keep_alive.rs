use humantime::parse_duration;
use serde_json::Value;

use crate::error::ProxyError;

pub fn parse_keep_alive_seconds(raw_value: Option<&Value>) -> Result<Option<i64>, ProxyError> {
    let Some(value) = raw_value else {
        return Ok(None);
    };

    match value {
        Value::Null => Ok(None),
        Value::Number(num) => {
            if let Some(signed) = num.as_i64() {
                Ok(Some(signed))
            } else if let Some(unsigned) = num.as_u64() {
                if unsigned <= i64::MAX as u64 {
                    Ok(Some(unsigned as i64))
                } else {
                    Err(ProxyError::bad_request(
                        "keep_alive value exceeds supported range",
                    ))
                }
            } else {
                Err(ProxyError::bad_request("keep_alive must be integral"))
            }
        }
        Value::String(text) => {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                return Ok(None);
            }

            if let Ok(duration) = parse_duration(trimmed) {
                if duration.as_secs() <= i64::MAX as u64 {
                    Ok(Some(duration.as_secs() as i64))
                } else {
                    Err(ProxyError::bad_request(
                        "keep_alive duration exceeds supported range",
                    ))
                }
            } else {
                trimmed.parse::<i64>().map(Some).map_err(|_| {
                    ProxyError::bad_request(
                        "invalid keep_alive value. Use numeric seconds or durations like '5m'",
                    )
                })
            }
        }
        _ => Err(ProxyError::bad_request(
            "invalid keep_alive value. Use numeric seconds or durations like '5m'",
        )),
    }
}

pub fn apply_keep_alive_ttl(target: &mut Value, keep_alive_seconds: Option<i64>) {
    if let Some(ttl) = keep_alive_seconds
        && let Some(obj) = target.as_object_mut()
    {
        obj.insert("ttl".to_string(), Value::from(ttl));
    }
}

pub fn keep_alive_requests_unload(ttl: Option<i64>) -> bool {
    matches!(ttl, Some(value) if value == 0)
}
