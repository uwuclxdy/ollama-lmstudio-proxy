use std::path::PathBuf;

use futures_util::{Stream, StreamExt};
use sha2::{Digest, Sha256};
use tokio::fs;
use tokio::io::AsyncWriteExt;

use crate::error::ProxyError;

pub struct BlobStore {
    base_dir: PathBuf,
}

impl BlobStore {
    pub fn new<P: Into<PathBuf>>(base_dir: P) -> Result<Self, ProxyError> {
        let dir = base_dir.into();
        std::fs::create_dir_all(&dir).map_err(|e| {
            ProxyError::internal_server_error(&format!(
                "failed to initialize blob storage directory: {}",
                e
            ))
        })?;
        Ok(Self { base_dir: dir })
    }

    fn validated_blob_path(&self, digest: &str) -> Result<PathBuf, ProxyError> {
        let Some((algo, hex)) = digest.split_once(':') else {
            return Err(ProxyError::bad_request(
                "invalid digest format. Expected algo:hex (e.g. sha256:abc)",
            ));
        };
        if algo != "sha256" {
            return Err(ProxyError::bad_request("only sha256 digests are supported"));
        }
        if hex.len() != 64 || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(ProxyError::bad_request("invalid sha256 digest"));
        }
        Ok(self.base_dir.join(algo).join(hex))
    }

    pub async fn exists(&self, digest: &str) -> Result<bool, ProxyError> {
        let path = self.validated_blob_path(digest)?;
        tokio::fs::try_exists(path).await.map_err(|e| {
            ProxyError::internal_server_error(&format!(
                "failed to verify blob existence for {}: {}",
                digest, e
            ))
        })
    }

    pub async fn save_stream<S>(&self, digest: &str, mut stream: S) -> Result<(), ProxyError>
    where
        S: Stream<Item = Result<bytes::Bytes, warp::Error>> + Unpin,
    {
        let final_path = self.validated_blob_path(digest)?;
        let hex = digest.split_once(':').map(|(_, h)| h).unwrap();

        if let Some(parent) = final_path.parent() {
            fs::create_dir_all(parent).await.map_err(|e| {
                ProxyError::internal_server_error(&format!(
                    "failed to prepare blob directory: {}",
                    e
                ))
            })?;
        }

        let tmp_path = final_path.with_extension(format!("{}.tmp", std::process::id()));
        let mut file = fs::File::create(&tmp_path).await.map_err(|e| {
            ProxyError::internal_server_error(&format!(
                "failed to create temporary blob file: {}",
                e
            ))
        })?;
        let mut hasher = Sha256::new();
        let mut total_bytes = 0u64;

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| {
                ProxyError::internal_server_error(&format!("blob upload error: {}", e))
            })?;
            hasher.update(&chunk);
            file.write_all(&chunk).await.map_err(|e| {
                ProxyError::internal_server_error(&format!("failed writing blob chunk: {}", e))
            })?;
            total_bytes = total_bytes.saturating_add(chunk.len() as u64);
        }

        file.flush().await.map_err(|e| {
            ProxyError::internal_server_error(&format!("failed to flush blob data: {}", e))
        })?;

        let actual_hex = format!("{:x}", hasher.finalize());
        if actual_hex != hex {
            let _ = fs::remove_file(&tmp_path).await;
            return Err(ProxyError::bad_request(&format!(
                "digest mismatch. Expected {}, computed {}",
                hex, actual_hex
            )));
        }

        fs::rename(&tmp_path, &final_path).await.map_err(|e| {
            ProxyError::internal_server_error(&format!("failed to finalize blob file: {}", e))
        })?;

        log::info!("stored blob {} ({} bytes)", digest, total_bytes);
        Ok(())
    }
}
