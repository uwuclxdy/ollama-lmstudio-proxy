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

    /// Appends query parameters to a URL
    pub fn append_query_params(&self, mut base: String, query: Option<&str>) -> String {
        if let Some(qs) = query {
            if base.contains('?') {
                base.push('&');
                base.push_str(qs);
            } else {
                base.push_str(&format!("?{}", qs));
            }
        }
        base
    }
}
