use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::AsyncWriteExt;

use futures_util::{Stream, StreamExt};
use sha2::{Digest, Sha256};

use crate::utils::ProxyError;

/// On-disk blob storage used to emulate Ollama's blob endpoints.
#[allow(dead_code)]
pub struct BlobStore {
    base_dir: PathBuf,
}

#[allow(dead_code)]
impl BlobStore {
    pub fn new<P: Into<PathBuf>>(base_dir: P) -> Result<Self, ProxyError> {
        let dir = base_dir.into();
        std::fs::create_dir_all(&dir).map_err(|e| {
            ProxyError::internal_server_error(&format!(
                "Failed to initialize blob storage directory: {}",
                e
            ))
        })?;
        Ok(Self { base_dir: dir })
    }

    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    fn digest_parts(digest: &str) -> Result<(&str, &str), ProxyError> {
        let Some((algo, hex)) = digest.split_once(':') else {
            return Err(ProxyError::bad_request(
                "Invalid digest format. Expected algo:hex (e.g. sha256:abc)",
            ));
        };
        if algo != "sha256" {
            return Err(ProxyError::bad_request("Only sha256 digests are supported"));
        }
        if hex.len() != 64 || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(ProxyError::bad_request("Invalid sha256 digest"));
        }
        Ok((algo, hex))
    }

    fn blob_path(&self, digest: &str) -> Result<PathBuf, ProxyError> {
        let (algo, hex) = Self::digest_parts(digest)?;
        Ok(self.base_dir.join(algo).join(hex))
    }

    pub async fn exists(&self, digest: &str) -> Result<bool, ProxyError> {
        let path = self.blob_path(digest)?;
        tokio::fs::try_exists(path).await.map_err(|e| {
            ProxyError::internal_server_error(&format!(
                "Failed to verify blob existence for {}: {}",
                digest, e
            ))
        })
    }

    pub async fn save_stream<S>(&self, digest: &str, mut stream: S) -> Result<(), ProxyError>
    where
        S: Stream<Item = Result<bytes::Bytes, warp::Error>> + Unpin,
    {
        let (_algo, hex) = Self::digest_parts(digest)?;
        let final_path = self.blob_path(digest)?;

        if let Some(parent) = final_path.parent() {
            fs::create_dir_all(parent).await.map_err(|e| {
                ProxyError::internal_server_error(&format!(
                    "Failed to prepare blob directory: {}",
                    e
                ))
            })?;
        }

        let tmp_path = final_path.with_extension(format!("{}.tmp", std::process::id()));
        let mut file = fs::File::create(&tmp_path).await.map_err(|e| {
            ProxyError::internal_server_error(&format!(
                "Failed to create temporary blob file: {}",
                e
            ))
        })?;
        let mut hasher = Sha256::new();
        let mut total_bytes = 0u64;

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| {
                ProxyError::internal_server_error(&format!("Blob upload error: {}", e))
            })?;
            hasher.update(&chunk);
            file.write_all(&chunk).await.map_err(|e| {
                ProxyError::internal_server_error(&format!("Failed writing blob chunk: {}", e))
            })?;
            total_bytes = total_bytes.saturating_add(chunk.len() as u64);
        }

        file.flush().await.map_err(|e| {
            ProxyError::internal_server_error(&format!("Failed to flush blob data: {}", e))
        })?;

        let actual_hex = format!("{:x}", hasher.finalize());
        if actual_hex != hex {
            let _ = fs::remove_file(&tmp_path).await;
            return Err(ProxyError::bad_request(&format!(
                "Digest mismatch. Expected {}, computed {}",
                hex, actual_hex
            )));
        }

        fs::rename(&tmp_path, &final_path).await.map_err(|e| {
            ProxyError::internal_server_error(&format!("Failed to finalize blob file: {}", e))
        })?;

        crate::utils::log_info(&format!("Stored blob {} ({} bytes)", digest, total_bytes));
        Ok(())
    }
}
