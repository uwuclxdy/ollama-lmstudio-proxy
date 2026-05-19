use std::error::Error;
use std::path::Path;
use std::time::Duration;

use serde::Deserialize;
use update_informer::{Check, registry};

const REPO: &str = "uwuclxdy/ollama-lmstudio-proxy";
const GITHUB_API_URL: &str =
    "https://api.github.com/repos/uwuclxdy/ollama-lmstudio-proxy/releases/latest";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InstallMethod {
    Cargo,
    Binary,
}

fn classify_install_path(exe: &Path, cargo_home: Option<&str>) -> InstallMethod {
    let path = exe.to_string_lossy();
    if path.contains("/.cargo/bin/") || path.contains("\\.cargo\\bin\\") {
        return InstallMethod::Cargo;
    }
    if let Some(home) = cargo_home
        && exe.starts_with(home)
    {
        return InstallMethod::Cargo;
    }
    InstallMethod::Binary
}

fn detect_install_method() -> InstallMethod {
    let Ok(exe) = std::env::current_exe() else {
        return InstallMethod::Binary;
    };
    let exe = exe.canonicalize().unwrap_or(exe);
    classify_install_path(&exe, std::env::var("CARGO_HOME").ok().as_deref())
}

#[derive(Debug, Deserialize)]
struct GithubRelease {
    assets: Vec<GithubAsset>,
}

#[derive(Debug, Deserialize)]
struct GithubAsset {
    name: String,
    browser_download_url: String,
}

fn platform_asset_name() -> Result<&'static str, Box<dyn Error>> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => Ok("ollama-lmstudio-proxy-linux-x64"),
        ("windows", "x86_64") => Ok("ollama-lmstudio-proxy-windows-x64.exe"),
        (os, arch) => Err(format!("unsupported platform: {}-{}", os, arch).into()),
    }
}

pub async fn check_and_update() -> Result<(), Box<dyn Error>> {
    log::info!("checking for updates...");
    let current = crate::VERSION;

    let informer = update_informer::new(registry::GitHub, REPO, current).interval(Duration::ZERO);
    let latest =
        tokio::task::spawn_blocking(move || informer.check_version().map_err(|e| e.to_string()))
            .await?
            .map_err(|e| format!("failed to check for updates: {}", e))?;

    let Some(latest) = latest else {
        log::info!("already running the latest version ({})", current);
        return Ok(());
    };

    let latest = latest.to_string();
    let latest = latest.trim_start_matches('v');
    log::info!("new version available: {} -> {}", current, latest);

    match detect_install_method() {
        InstallMethod::Cargo => {
            log::info!(
                "installed via cargo; run `cargo install --force {}` to update",
                env!("CARGO_PKG_NAME")
            );
            Ok(())
        }
        InstallMethod::Binary => {
            download_and_replace().await?;
            println!();
            println!("update complete: {} -> {}", current, latest);
            println!("please relaunch the application to use the new version");
            Ok(())
        }
    }
}

async fn download_and_replace() -> Result<(), Box<dyn Error>> {
    let client = reqwest::Client::builder()
        .user_agent(format!("ollama-lmstudio-proxy/{}", crate::VERSION))
        .build()?;

    let release: GithubRelease = client
        .get(GITHUB_API_URL)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    let asset_name = platform_asset_name()?;
    let asset = release
        .assets
        .iter()
        .find(|a| a.name == asset_name)
        .ok_or_else(|| format!("no asset found for platform: {}", asset_name))?;

    log::info!("downloading update from: {}", asset.browser_download_url);
    let bytes = client
        .get(&asset.browser_download_url)
        .send()
        .await?
        .error_for_status()?
        .bytes()
        .await?;
    log::info!("downloaded {} bytes", bytes.len());

    let temp_path = std::env::temp_dir().join(format!(
        "ollama-lmstudio-proxy-update-{}",
        std::process::id()
    ));
    std::fs::write(&temp_path, &bytes)?;

    log::info!("replacing current executable...");
    self_replace::self_replace(&temp_path)?;
    let _ = std::fs::remove_file(&temp_path);

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let current_exe = std::env::current_exe()?;
        let mut perms = std::fs::metadata(&current_exe)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&current_exe, perms)?;
    }

    Ok(())
}

#[cfg(test)]
#[path = "../tests/unit/update.rs"]
mod tests;
