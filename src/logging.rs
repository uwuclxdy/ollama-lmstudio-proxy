use std::sync::OnceLock;
use std::time::{Duration, Instant};

use crate::constants::{LOG_PREFIX_ERROR, LOG_PREFIX_SUCCESS, LOG_PREFIX_WARNING};

pub struct LogConfig {
    pub debug_enabled: bool,
}

static LOG_CONFIG: OnceLock<LogConfig> = OnceLock::new();

impl LogConfig {
    pub fn init(debug: bool) {
        LOG_CONFIG.get_or_init(|| LogConfig {
            debug_enabled: debug,
        });
    }

    pub fn get() -> &'static LogConfig {
        LOG_CONFIG.get().unwrap_or_else(|| {
            static FALLBACK: LogConfig = LogConfig {
                debug_enabled: false,
            };
            &FALLBACK
        })
    }
}

pub fn log_request(method: &str, path: &str, model: Option<&str>) {
    match model {
        Some(m) => log::info!(
            "{} {} (model: {})",
            method,
            sanitize_log_message(path),
            sanitize_log_message(m)
        ),
        None => log::info!("{} {}", method, sanitize_log_message(path)),
    }
}

pub fn log_timed(prefix: &str, operation: &str, start: Instant) {
    let duration = start.elapsed();
    let formatted_duration = format_duration(duration);

    match prefix {
        LOG_PREFIX_SUCCESS => log::info!("{} | {}", operation, formatted_duration),
        LOG_PREFIX_ERROR => log::error!("{} | {}", operation, formatted_duration),
        LOG_PREFIX_WARNING => log::warn!("{} | {}", operation, formatted_duration),
        _ => log::info!("{} | {}", operation, formatted_duration),
    }
}

pub fn format_duration(duration: Duration) -> String {
    let total_nanos = duration.as_nanos();

    if total_nanos < 1_000_000 {
        format!("{:.1}µs", total_nanos as f64 / 1_000.0)
    } else if total_nanos < 1_000_000_000 {
        format!("{:.2}ms", total_nanos as f64 / 1_000_000.0)
    } else {
        format!("{:.2}s", total_nanos as f64 / 1_000_000_000.0)
    }
}

pub fn sanitize_log_message(message: &str) -> String {
    message
        .chars()
        .map(|c| {
            if c.is_control() && !matches!(c, '\t' | '\n' | '\r') {
                '?'
            } else {
                c
            }
        })
        .collect()
}

pub fn log_handler_io(
    endpoint: &str,
    body: Option<&serde_json::Value>,
    response: Option<&serde_json::Value>,
    streaming: bool,
) {
    if LogConfig::get().debug_enabled {
        if let Some(body_value) = body {
            log::debug!(
                "{} request: {}",
                endpoint,
                serde_json::to_string_pretty(body_value).unwrap_or_default()
            );
        }

        if let Some(response_value) = response {
            if streaming {
                log::debug!("{} response: (streaming)", endpoint);
            } else {
                log::debug!(
                    "{} response: {}",
                    endpoint,
                    serde_json::to_string_pretty(response_value).unwrap_or_default()
                );
            }
        }
    }
}
