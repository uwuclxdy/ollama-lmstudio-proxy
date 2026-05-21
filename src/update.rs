use std::time::Duration;

use update_informer::{Check, registry};

const REPO: &str = "uwuclxdy/ollama-lmstudio-proxy";
const CHECK_INTERVAL: Duration = Duration::from_secs(60 * 60 * 24);

pub fn check_for_update() {
    let current = crate::VERSION;
    tokio::task::spawn_blocking(move || {
        let informer =
            update_informer::new(registry::GitHub, REPO, current).interval(CHECK_INTERVAL);
        match informer.check_version() {
            Ok(Some(latest)) => {
                let latest = latest.to_string();
                let latest = latest.trim_start_matches('v');
                log::warn!(
                    "new version available: {} -> {} (https://github.com/{}/releases)",
                    current,
                    latest,
                    REPO
                );
            }
            Ok(None) => {}
            Err(e) => log::debug!("update check failed: {}", e),
        }
    });
}
