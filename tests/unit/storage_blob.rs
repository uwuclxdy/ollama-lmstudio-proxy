use crate::error::ProxyError;

use super::*;

fn fresh_blob_store() -> BlobStore {
    let dir = std::env::temp_dir().join(format!(
        "olp-test-blobs-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    BlobStore::new(&dir).expect("BlobStore::new should succeed in temp dir")
}

fn check_digest(digest: &str) -> Result<bool, ProxyError> {
    let store = fresh_blob_store();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    rt.block_on(store.exists(digest))
}

/// Valid `sha256:` + 64 lowercase hex characters → accepted, returns
/// Ok(false) because no such blob exists in a fresh store.
#[test]
fn blob_digest_valid_sha256_is_accepted() {
    let digest = format!("sha256:{}", "a".repeat(64));
    let result = check_digest(&digest);
    assert!(matches!(result, Ok(false)), "got {result:?}");
}

/// Wrong algorithm prefix (md5) → rejected with 400. Only sha256 is
/// supported per the validator and per the Ollama spec.
#[test]
fn blob_digest_wrong_algorithm_is_rejected() {
    let digest = format!("md5:{}", "a".repeat(32));
    let err = check_digest(&digest).expect_err("non-sha256 must be rejected");
    assert_eq!(err.status_code, 400);
}

/// Missing colon separator → rejected. The validator requires `algo:hex`.
#[test]
fn blob_digest_missing_colon_is_rejected() {
    let digest = "a".repeat(64);
    let err = check_digest(&digest).expect_err("missing colon must be rejected");
    assert_eq!(err.status_code, 400);
}

/// Hex shorter than 64 characters → rejected.
#[test]
fn blob_digest_short_hex_is_rejected() {
    let digest = format!("sha256:{}", "a".repeat(32));
    let err = check_digest(&digest).expect_err("short hex must be rejected");
    assert_eq!(err.status_code, 400);
}

/// Hex longer than 64 characters → rejected.
#[test]
fn blob_digest_long_hex_is_rejected() {
    let digest = format!("sha256:{}", "a".repeat(128));
    let err = check_digest(&digest).expect_err("oversize hex must be rejected");
    assert_eq!(err.status_code, 400);
}

/// Non-hex characters in the hex portion → rejected. Guards against
/// path-traversal payloads slipping through (`/`, `.`, etc.) and against
/// uppercase or unicode confusables.
#[test]
fn blob_digest_non_hex_chars_are_rejected() {
    let bad = [
        // Path traversal attempt — would escape base_dir if accepted as a path.
        format!(
            "sha256:{}",
            "../../../etc/passwd000000000000000000000000000000"
        ),
        // Embedded slash.
        format!("sha256:{}{}", "a".repeat(32), "/".repeat(32)),
        // Non-hex letter.
        format!("sha256:{}", "z".repeat(64)),
    ];
    for digest in bad {
        let err = check_digest(&digest).expect_err(&format!("must reject {digest}"));
        assert_eq!(err.status_code, 400, "digest {digest} should be 400");
    }
}

/// Path-traversal attempt with `..` and slash in the hex portion → rejected.
/// Critical: without this check an attacker could read/write arbitrary files
/// under the blob storage root.
#[test]
fn blob_digest_path_traversal_is_rejected() {
    // exactly 64 chars but containing `..` and `/`.
    let digest = format!("sha256:{}", "../".repeat(21) + "a");
    assert_eq!(digest.len() - "sha256:".len(), 64);
    let err = check_digest(&digest).expect_err("path traversal must be rejected");
    assert_eq!(err.status_code, 400);
}
