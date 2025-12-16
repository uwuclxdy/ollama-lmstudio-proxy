use serde::Deserialize;
use std::error::Error;

const GITHUB_API_URL: &str =
    "https://api.github.com/repos/uwuclxdy/ollama-lmstudio-proxy/releases/latest";

#[derive(Debug, Deserialize)]
struct GithubRelease {
    tag_name: String,
    assets: Vec<GithubAsset>,
}

#[derive(Debug, Deserialize)]
struct GithubAsset {
    name: String,
    browser_download_url: String,
}

fn get_platform_executable_name() -> Result<&'static str, Box<dyn Error>> {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;

    match (os, arch) {
        ("linux", "x86_64") => Ok("ollama-lmstudio-proxy-linux-x64"),
        ("windows", "x86_64") => Ok("ollama-lmstudio-proxy-windows-x64.exe"),
        _ => Err(format!("unsupported platform: {}-{}", os, arch).into()),
    }
}

pub async fn check_and_update() -> Result<(), Box<dyn Error>> {
    log::info!("checking for updates...");

    let client = reqwest::Client::builder()
        .user_agent(format!("ollama-lmstudio-proxy/{}", crate::VERSION))
        .build()?;

    let response = client.get(GITHUB_API_URL).send().await?;

    if !response.status().is_success() {
        return Err(format!("failed to fetch release info: {}", response.status()).into());
    }

    let release: GithubRelease = response.json().await?;
    let latest_version = release.tag_name.trim_start_matches('v');
    let current_version = crate::VERSION;

    log::info!("current version: {}", current_version);
    log::info!("latest version: {}", latest_version);

    if latest_version == current_version {
        log::info!("already running the latest version");
        return Ok(());
    }

    log::info!(
        "new version available: {} -> {}",
        current_version,
        latest_version
    );

    let platform_name = get_platform_executable_name()?;
    log::info!("looking for asset: {}", platform_name);

    let asset = release
        .assets
        .iter()
        .find(|a| a.name == platform_name)
        .ok_or_else(|| format!("no asset found for platform: {}", platform_name))?;

    log::info!("downloading update from: {}", asset.browser_download_url);

    let download_response = client.get(&asset.browser_download_url).send().await?;

    if !download_response.status().is_success() {
        return Err(format!("failed to download asset: {}", download_response.status()).into());
    }

    let binary_data = download_response.bytes().await?;
    log::info!("downloaded {} bytes", binary_data.len());

    let temp_path = std::env::temp_dir().join(format!(
        "ollama-lmstudio-proxy-update-{}",
        std::process::id()
    ));
    std::fs::write(&temp_path, &binary_data)?;

    log::info!("replacing current executable...");
    self_replace::self_replace(&temp_path)?;

    let _ = std::fs::remove_file(&temp_path);

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let current_exe = std::env::current_exe()?;
        let metadata = std::fs::metadata(&current_exe)?;
        let mut perms = metadata.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&current_exe, perms)?;
        log::info!("set executable permissions");
    }

    log::info!("successfully updated to version {}", latest_version);
    log::info!("please restart the application to use the new version");

    Ok(())
}
