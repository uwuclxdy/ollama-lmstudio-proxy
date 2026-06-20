use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Thread-safe record of when the proxy last loaded or kept a model alive, plus
/// the resolved keep-alive TTL at that moment.
///
/// LM Studio's loaded-instance list exposes no expiry, so `/api/ps` can't derive
/// a real `expires_at` from the backend alone. This tracker lets the proxy report
/// its own best-effort deadline: a model the proxy loaded at `t` with `ttl` of
/// `d` seconds expires at `t + d`. A `ttl` of `None` means "loaded forever" (no
/// expiry); negative `ttl` is LM Studio's "stay loaded" sentinel and is also
/// treated as forever. Models the proxy never loaded (e.g. loaded out-of-band)
/// stay untracked → the caller falls back to its existing placeholder.
///
/// Exposed behind `Arc` so handlers share one instance across requests.
#[derive(Default)]
pub struct LoadTracker {
    // (loaded_at, ttl): None ttl = no expiry (forever).
    entries: Mutex<HashMap<String, (Instant, Option<Duration>)>>,
}

impl LoadTracker {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Record (or refresh) `key` as loaded now with the given `ttl`.
    ///
    /// `ttl: None` and negative seconds both mean "loaded forever" (no expiry).
    pub fn record(&self, key: &str, ttl: Option<Duration>) {
        // Negative/zero seconds is LM Studio's "stay loaded"/unload sentinel →
        // "no known ttl" (None).
        let normalized = ttl.filter(|d| !d.is_zero() && d.as_secs_f64() >= 0.0);
        let mut guard = self
            .entries
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // Always refresh `loaded_at` (the model was just touched), but only
        // replace `ttl` when this call actually knows it. Warm paths (e.g. the
        // /api/show chat-ping) record `None`; they must NOT overwrite a real
        // keep_alive ttl captured by the preceding inference request, or
        // /api/ps would flip back to "no expiry" on every warm call.
        match guard.get_mut(key) {
            Some((loaded_at, existing_ttl)) => {
                *loaded_at = Instant::now();
                if normalized.is_some() {
                    *existing_ttl = normalized;
                }
            }
            None => {
                guard.insert(key.to_string(), (Instant::now(), normalized));
            }
        }
    }

    /// Unix-seconds expiry for `key`, or `None` when untracked or loaded forever.
    pub fn expires_at_unix(&self, key: &str) -> Option<i64> {
        let guard = self
            .entries
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let (loaded_at, ttl) = guard.get(key)?;
        let ttl = (*ttl)?;
        let elapsed = loaded_at.elapsed();
        // Deadline in wall-clock unix seconds. `SystemTime` is used (not
        // `Instant`) since `expires_at` is reported to clients as an absolute
        // timestamp; `loaded_at` is an `Instant` so we anchor the deadline at
        // `now + remaining`.
        let remaining = ttl.saturating_sub(elapsed);
        let now_unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        Some(now_unix + i64::try_from(remaining.as_secs()).unwrap_or(i64::MAX))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn untracked_returns_none() {
        let t = LoadTracker::new();
        assert!(t.expires_at_unix("nope").is_none());
    }

    #[test]
    fn none_ttl_means_forever_no_expiry() {
        let t = LoadTracker::new();
        t.record("m", None);
        assert!(t.expires_at_unix("m").is_none());
    }

    #[test]
    fn none_ttl_refresh_does_not_clobber_known_ttl() {
        // Regression: a warm-path record(None) (e.g. the /api/show chat-ping)
        // must not overwrite a real keep_alive ttl captured by the prior
        // inference request — otherwise /api/ps flips back to "no expiry".
        let t = LoadTracker::new();
        t.record("m", Some(Duration::from_secs(300)));
        t.record("m", None);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        let exp = t
            .expires_at_unix("m")
            .expect("ttl must be retained after None refresh");
        assert!(exp > now, "expiry must be in the future: {exp} <= {now}");
        assert!(
            exp <= now + 300,
            "expiry must stay within the original ttl: {exp} > {now}+300"
        );
    }

    #[test]
    fn some_ttl_returns_future_unix_deadline() {
        let t = LoadTracker::new();
        t.record("m", Some(Duration::from_secs(300)));
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        let exp = t.expires_at_unix("m").expect("tracked with ttl");
        assert!(exp > now, "expiry must be in the future: {exp} <= {now}");
        assert!(
            exp <= now + 300,
            "expiry must be within ttl: {exp} > {now}+300"
        );
    }

    #[test]
    fn record_refreshes_loaded_at() {
        let t = LoadTracker::new();
        t.record("m", Some(Duration::from_secs(1)));
        std::thread::sleep(Duration::from_millis(20));
        let first = t.expires_at_unix("m").expect("tracked");
        t.record("m", Some(Duration::from_secs(300)));
        let second = t.expires_at_unix("m").expect("still tracked");
        assert!(second > first, "re-record must push the deadline out");
    }
}
