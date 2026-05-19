use super::*;
use std::path::PathBuf;

#[test]
fn classifies_unix_cargo_bin_as_cargo() {
    let exe = PathBuf::from("/home/alice/.cargo/bin/ollama-lmstudio-proxy");
    assert_eq!(classify_install_path(&exe, None), InstallMethod::Cargo);
}

#[test]
fn classifies_windows_cargo_bin_as_cargo() {
    let exe = PathBuf::from(r"C:\Users\alice\.cargo\bin\ollama-lmstudio-proxy.exe");
    assert_eq!(classify_install_path(&exe, None), InstallMethod::Cargo);
}

#[test]
fn classifies_system_path_as_binary() {
    let exe = PathBuf::from("/usr/local/bin/ollama-lmstudio-proxy");
    assert_eq!(classify_install_path(&exe, None), InstallMethod::Binary);
}

#[test]
fn classifies_custom_cargo_home_match_as_cargo() {
    let exe = PathBuf::from("/opt/cargo/bin/ollama-lmstudio-proxy");
    assert_eq!(
        classify_install_path(&exe, Some("/opt/cargo")),
        InstallMethod::Cargo
    );
}

#[test]
fn classifies_unrelated_path_with_cargo_home_set_as_binary() {
    let exe = PathBuf::from("/home/alice/Downloads/ollama-lmstudio-proxy");
    assert_eq!(
        classify_install_path(&exe, Some("/opt/cargo")),
        InstallMethod::Binary
    );
}

#[test]
fn cargo_home_starts_with_uses_path_components_not_byte_prefix() {
    let exe = PathBuf::from("/opt/cargo-other/bin/ollama-lmstudio-proxy");
    assert_eq!(
        classify_install_path(&exe, Some("/opt/cargo")),
        InstallMethod::Binary
    );
}

#[test]
fn returns_platform_asset_name_for_supported_targets() {
    let name = platform_asset_name().expect("current test platform must be supported");
    assert!(name.starts_with("ollama-lmstudio-proxy-"));
}
