use std::sync::Arc;

use crate::storage::{BlobStore, VirtualModelStore};

#[derive(Clone)]
pub struct RequestContext<'a> {
    pub client: &'a reqwest::Client,
    pub lmstudio_url: &'a str,
    pub virtual_models: Arc<VirtualModelStore>,
    pub blob_store: Arc<BlobStore>,
}

impl<'a> RequestContext<'a> {
    /// Constructs a full URL by joining base URL with an endpoint.
    pub fn endpoint_url(&self, endpoint: &str) -> String {
        format!("{}{}", self.lmstudio_url, endpoint)
    }
}
