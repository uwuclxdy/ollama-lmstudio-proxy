use std::sync::Arc;

use crate::storage::{BlobStore, VirtualModelStore};

#[derive(Clone)]
pub struct RequestContext<'a> {
    pub client: &'a reqwest::Client,
    pub lmstudio_url: &'a str,
    pub virtual_models: Arc<VirtualModelStore>,
    pub blob_store: Arc<BlobStore>,
}
