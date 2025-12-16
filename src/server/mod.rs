pub mod proxy;
pub mod rejection;
pub mod routes;

pub use proxy::ProxyServer;
pub use rejection::handle_rejection;

use std::sync::Arc;

use crate::model::ModelResolver;

#[derive(Clone)]
pub enum ModelResolverType {
    Native(Arc<ModelResolver>),
}
