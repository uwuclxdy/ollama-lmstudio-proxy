use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The caller's intent when recording a model load or refresh.
///
/// `Unknown` means "this code path has no keep_alive information" — the warm
/// path and JIT-load path use it. It refreshes `loaded_at` but never overwrites
/// an existing finite/forever deadline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeepAlive {
    /// Positive TTL: the model expires `d` seconds after it was last touched.
    Finite(Duration),
    /// keep_alive < 0 (e.g. -1): stay loaded indefinitely, mirrors Ollama's
    /// far-future `expires_at` for this intent.
    Forever,
    /// No keep_alive information available for this call site; existing deadline
    /// is preserved if the model was already tracked.
    Unknown,
}

/// Stored per-entry state — two states only; `Unknown` is never persisted.
#[derive(Debug, Clone, Copy)]
enum StoredTtl {
    Finite(Duration),
    Forever,
}

/// Thread-safe record of when the proxy last loaded or kept a model alive, plus
/// the resolved keep-alive intent at that moment.
///
/// LM Studio's loaded-instance list exposes no expiry, so `/api/ps` can't derive
/// a real `expires_at` from the backend alone. This tracker lets the proxy report
/// its own best-effort deadline: a model the proxy loaded at `t` with `ttl` of
/// `d` seconds expires at `t + d`. `Forever` yields a far-future timestamp that
/// matches real Ollama's behaviour for `keep_alive: -1`. Models the proxy never
/// loaded (e.g. loaded out-of-band) stay untracked → the caller falls back to
/// its existing placeholder.
///
/// Exposed behind `Arc` so handlers share one instance across requests.
#[derive(Default)]
pub struct LoadTracker {
    entries: Mutex<HashMap<String, (Instant, Option<StoredTtl>)>>,
}

impl LoadTracker {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Record (or refresh) `key` as loaded now with the given intent.
    ///
    /// - `Finite(d)` — set deadline to now + d (d==0 is treated as Unknown).
    /// - `Forever` — mark as loaded indefinitely.
    /// - `Unknown` — refresh `loaded_at`; PRESERVE any existing deadline.
    ///   A fresh Unknown entry has no stored deadline → `expires_at_unix` returns
    ///   `None`, which lets `/api/ps` fall back to its now+default placeholder.
    pub fn record(&self, key: &str, intent: KeepAlive) {
        let stored = match intent {
            KeepAlive::Finite(d) if !d.is_zero() => Some(StoredTtl::Finite(d)),
            // Zero duration is indistinguishable from "no info"; treat as Unknown.
            KeepAlive::Finite(_) => None,
            KeepAlive::Forever => Some(StoredTtl::Forever),
            KeepAlive::Unknown => None,
        };

        let mut guard = self
            .entries
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        match guard.get_mut(key) {
            Some((loaded_at, existing_ttl)) => {
                *loaded_at = Instant::now();
                // Only overwrite when this call carries real information.
                if matches!(intent, KeepAlive::Finite(_) | KeepAlive::Forever) {
                    *existing_ttl = stored;
                }
                // Unknown → leave existing_ttl untouched (no-clobber).
            }
            None => {
                // Fresh entry: store None for Unknown (→ no expiry info yet),
                // or the real deadline for Finite/Forever.
                guard.insert(key.to_string(), (Instant::now(), stored));
            }
        }
    }

    /// Unix-seconds expiry for `key`, or `None` when untracked or deadline is unknown.
    ///
    /// - Finite deadline → `Some(now + remaining)`.
    /// - Forever → `Some(far_future_unix)`. The exact sentinel (~100 years from
    ///   now) mirrors real Ollama's far-future `expires_at` for `keep_alive: -1`;
    ///   the spec (api-docs/ollama/api/ps.md) does not prescribe the value.
    /// - Unknown deadline or untracked → `None` (caller uses placeholder).
    pub fn expires_at_unix(&self, key: &str) -> Option<i64> {
        let guard = self
            .entries
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let (loaded_at, ttl) = guard.get(key)?;
        let now_unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        match ttl {
            Some(StoredTtl::Finite(d)) => {
                let elapsed = loaded_at.elapsed();
                let remaining = d.saturating_sub(elapsed);
                // Anchor deadline at now + remaining (Instant → wall-clock).
                Some(now_unix + i64::try_from(remaining.as_secs()).unwrap_or(i64::MAX))
            }
            Some(StoredTtl::Forever) => {
                // ~100 years in seconds; large enough to be unmistakably "never".
                const HUNDRED_YEARS_SECS: i64 = 100 * 365 * 24 * 3600;
                Some(now_unix + HUNDRED_YEARS_SECS)
            }
            None => None,
        }
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
    fn forever_yields_far_future() {
        let t = LoadTracker::new();
        t.record("m", KeepAlive::Forever);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        let exp = t.expires_at_unix("m").expect("Forever must return Some");
        // Must be well beyond any reasonable finite deadline (at least 10 years out).
        assert!(
            exp >= now + 10 * 365 * 24 * 3600,
            "Forever expiry must be far future: {exp}"
        );
    }

    #[test]
    fn fresh_unknown_yields_none() {
        // A warm-path record with no prior info must yield None (→ placeholder in ps).
        let t = LoadTracker::new();
        t.record("m", KeepAlive::Unknown);
        assert!(
            t.expires_at_unix("m").is_none(),
            "fresh Unknown must yield None"
        );
    }

    #[test]
    fn unknown_does_not_clobber_finite() {
        // Regression: Unknown (warm-path) must not overwrite a real keep_alive ttl
        // captured by the prior inference request — otherwise /api/ps flips back to
        // placeholder on every warm call.
        let t = LoadTracker::new();
        t.record("m", KeepAlive::Finite(Duration::from_secs(300)));
        t.record("m", KeepAlive::Unknown);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        let exp = t
            .expires_at_unix("m")
            .expect("finite ttl must be retained after Unknown refresh");
        assert!(exp > now, "expiry must be in the future: {exp} <= {now}");
        assert!(
            exp <= now + 300,
            "expiry must stay within the original ttl: {exp} > {now}+300"
        );
    }

    #[test]
    fn unknown_does_not_clobber_forever() {
        let t = LoadTracker::new();
        t.record("m", KeepAlive::Forever);
        t.record("m", KeepAlive::Unknown);
        let exp = t
            .expires_at_unix("m")
            .expect("Forever must survive an Unknown refresh");
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        assert!(
            exp >= now + 10 * 365 * 24 * 3600,
            "Forever must remain far future after Unknown: {exp}"
        );
    }

    #[test]
    fn finite_ttl_returns_future_unix_deadline() {
        let t = LoadTracker::new();
        t.record("m", KeepAlive::Finite(Duration::from_secs(300)));
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        let exp = t.expires_at_unix("m").expect("tracked with finite ttl");
        assert!(exp > now, "expiry must be in the future: {exp} <= {now}");
        assert!(
            exp <= now + 300,
            "expiry must be within ttl: {exp} > {now}+300"
        );
    }

    #[test]
    fn record_refreshes_loaded_at() {
        let t = LoadTracker::new();
        t.record("m", KeepAlive::Finite(Duration::from_secs(1)));
        std::thread::sleep(Duration::from_millis(20));
        let first = t.expires_at_unix("m").expect("tracked");
        t.record("m", KeepAlive::Finite(Duration::from_secs(300)));
        let second = t.expires_at_unix("m").expect("still tracked");
        assert!(second > first, "re-record must push the deadline out");
    }
}
