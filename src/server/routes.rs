use std::sync::Arc;

use axum::extract::rejection::JsonRejection;
use axum::extract::{DefaultBodyLimit, FromRequest, Path, Query, Request, State};
use axum::response::Response;
use axum::routing::{delete, get, head, post};
use axum::{Json, Router};
use bytes::Bytes;
use http::HeaderMap;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::constants::MAX_JSON_BODY_SIZE_BYTES;
use crate::error::ProxyError;
use crate::handlers::ollama::{EmbeddingResponseMode, handle_ollama_embeddings};
use crate::handlers::{RequestContext, lmstudio, ollama};
use crate::http::json_response;
use crate::server::ProxyServer;

pub type AppState = Arc<ProxyServer>;

/// JSON body extractor that surfaces parse errors as ProxyError::bad_request
/// (a JSON `{"error": ..., "status": 400}` response), matching the Ollama-shaped
/// error envelope the proxy uses everywhere else.
pub struct JsonBody<T>(pub T);

impl<S, T> FromRequest<S> for JsonBody<T>
where
    S: Send + Sync,
    T: serde::de::DeserializeOwned,
    Json<T>: FromRequest<S, Rejection = JsonRejection>,
{
    type Rejection = ProxyError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        use axum::response::IntoResponse;
        match Json::<T>::from_request(req, state).await {
            Ok(Json(value)) => Ok(JsonBody(value)),
            Err(rejection) => {
                let status = rejection.status();
                let message = match status.as_u16() {
                    413 => "request body too large".to_string(),
                    _ => format!(
                        "invalid request body: {}",
                        rejection.into_response().status()
                    ),
                };
                Err(ProxyError::new(message, status.as_u16()))
            }
        }
    }
}

pub fn create_router(server: AppState) -> Router {
    let lmstudio_router = Router::new()
        .route(
            "/v1/{*path}",
            get(passthrough_v1)
                .post(passthrough_v1)
                .put(passthrough_v1)
                .delete(passthrough_v1)
                .head(passthrough_v1)
                .options(passthrough_v1),
        )
        .route(
            "/api/{version}/{*path}",
            get(passthrough_native_versioned)
                .post(passthrough_native_versioned)
                .put(passthrough_native_versioned)
                .delete(passthrough_native_versioned)
                .head(passthrough_native_versioned)
                .options(passthrough_native_versioned),
        )
        .route(
            "/api/{version}",
            get(passthrough_native_version_root)
                .post(passthrough_native_version_root)
                .put(passthrough_native_version_root)
                .delete(passthrough_native_version_root)
                .head(passthrough_native_version_root)
                .options(passthrough_native_version_root),
        );

    Router::new()
        .route("/", get(root_handler))
        .route("/health", get(health_handler))
        .route("/api/tags", get(tags_handler))
        .route("/api/chat", post(chat_handler))
        .route("/api/generate", post(generate_handler))
        .route("/api/embed", post(embed_handler))
        .route("/api/embeddings", post(embeddings_handler))
        .route("/api/pull", post(pull_handler))
        .route("/api/create", post(create_handler))
        .route("/api/copy", post(copy_handler))
        .route("/api/delete", delete(delete_handler))
        .route("/api/push", post(push_handler))
        .route("/api/show", post(show_handler))
        .route("/api/ps", get(ps_handler))
        .route("/api/version", get(version_handler))
        .route(
            "/api/blobs/{digest}",
            head(blob_head_handler).post(blob_upload_handler),
        )
        .merge(lmstudio_router)
        .method_not_allowed_fallback(method_not_allowed_handler)
        .fallback(not_found_handler)
        .layer(DefaultBodyLimit::max(MAX_JSON_BODY_SIZE_BYTES as usize))
        .with_state(server)
}

async fn not_found_handler() -> ProxyError {
    ProxyError::not_found("endpoint not found")
}

async fn method_not_allowed_handler() -> ProxyError {
    ProxyError::new("method not allowed".to_string(), 405)
}

fn create_context(s: &Arc<ProxyServer>) -> RequestContext<'_> {
    RequestContext {
        client: &s.client,
        lmstudio_url: &s.config.lmstudio_url,
        virtual_models: s.virtual_models.clone(),
        blob_store: s.blob_store.clone(),
    }
}

async fn root_handler(State(_): State<AppState>) -> Result<Response, ProxyError> {
    ollama::handle_ollama_root().await
}

async fn version_handler(State(_): State<AppState>) -> Result<Response, ProxyError> {
    ollama::handle_ollama_version().await
}

async fn health_handler(State(s): State<AppState>) -> Result<Response, ProxyError> {
    let context = create_context(&s);
    let value = ollama::handle_health_check(context, CancellationToken::new()).await?;
    Ok(json_response(&value))
}

async fn tags_handler(State(s): State<AppState>) -> Result<Response, ProxyError> {
    let context = create_context(&s);
    ollama::handle_ollama_tags(context, s.model_resolver.clone(), CancellationToken::new()).await
}

async fn chat_handler(
    State(s): State<AppState>,
    JsonBody(body): JsonBody<Value>,
) -> Result<Response, ProxyError> {
    let context = create_context(&s);
    ollama::handle_ollama_chat(
        context,
        s.model_resolver.clone(),
        body,
        CancellationToken::new(),
        s.config.load_timeout_seconds,
    )
    .await
}

async fn generate_handler(
    State(s): State<AppState>,
    JsonBody(body): JsonBody<Value>,
) -> Result<Response, ProxyError> {
    let context = create_context(&s);
    ollama::handle_ollama_generate(
        context,
        s.model_resolver.clone(),
        body,
        CancellationToken::new(),
        s.config.load_timeout_seconds,
    )
    .await
}

async fn embed_handler(
    State(s): State<AppState>,
    JsonBody(body): JsonBody<Value>,
) -> Result<Response, ProxyError> {
    embedding_handler_inner(s, body, EmbeddingResponseMode::Embed).await
}

async fn embeddings_handler(
    State(s): State<AppState>,
    JsonBody(body): JsonBody<Value>,
) -> Result<Response, ProxyError> {
    embedding_handler_inner(s, body, EmbeddingResponseMode::LegacyEmbeddings).await
}

async fn embedding_handler_inner(
    s: AppState,
    body: Value,
    mode: EmbeddingResponseMode,
) -> Result<Response, ProxyError> {
    let context = create_context(&s);
    handle_ollama_embeddings(
        context,
        s.model_resolver.clone(),
        body,
        mode,
        CancellationToken::new(),
        s.config.load_timeout_seconds,
    )
    .await
}

async fn pull_handler(
    State(s): State<AppState>,
    JsonBody(body): JsonBody<Value>,
) -> Result<Response, ProxyError> {
    let context = create_context(&s);
    ollama::handle_ollama_pull(
        context,
        s.model_resolver.clone(),
        body,
        CancellationToken::new(),
    )
    .await
}

async fn create_handler(
    State(s): State<AppState>,
    JsonBody(body): JsonBody<Value>,
) -> Result<Response, ProxyError> {
    let context = create_context(&s);
    ollama::handle_ollama_create(
        context,
        s.model_resolver.clone(),
        body,
        CancellationToken::new(),
    )
    .await
}

async fn copy_handler(
    State(s): State<AppState>,
    JsonBody(body): JsonBody<Value>,
) -> Result<Response, ProxyError> {
    let context = create_context(&s);
    ollama::handle_ollama_copy(
        context,
        s.model_resolver.clone(),
        body,
        CancellationToken::new(),
    )
    .await
}

async fn delete_handler(
    State(s): State<AppState>,
    JsonBody(body): JsonBody<Value>,
) -> Result<Response, ProxyError> {
    let context = create_context(&s);
    ollama::handle_ollama_delete(context, body).await
}

async fn push_handler(
    State(s): State<AppState>,
    JsonBody(body): JsonBody<Value>,
) -> Result<Response, ProxyError> {
    let context = create_context(&s);
    ollama::handle_ollama_push(
        context,
        s.model_resolver.clone(),
        body,
        CancellationToken::new(),
    )
    .await
}

async fn show_handler(
    State(s): State<AppState>,
    JsonBody(body): JsonBody<Value>,
) -> Result<Response, ProxyError> {
    let context = create_context(&s);
    ollama::handle_ollama_show(
        context,
        s.model_resolver.clone(),
        body,
        CancellationToken::new(),
    )
    .await
}

async fn ps_handler(State(s): State<AppState>) -> Result<Response, ProxyError> {
    let context = create_context(&s);
    ollama::handle_ollama_ps(context, s.model_resolver.clone(), CancellationToken::new()).await
}

async fn blob_head_handler(
    State(s): State<AppState>,
    Path(digest): Path<String>,
) -> Result<Response, ProxyError> {
    let context = create_context(&s);
    ollama::handle_blob_head(context, digest).await
}

async fn blob_upload_handler(
    State(s): State<AppState>,
    Path(digest): Path<String>,
    request: Request,
) -> Result<Response, ProxyError> {
    let context = create_context(&s);
    let body_stream = request.into_body().into_data_stream();
    ollama::handle_blob_upload(context, digest, body_stream).await
}

async fn passthrough_v1(
    State(s): State<AppState>,
    Path(path): Path<String>,
    Query(query): Query<Vec<(String, String)>>,
    method: http::Method,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ProxyError> {
    let full_path = format!("/v1/{}", path);
    let query_string = encode_query(&query);
    forward_passthrough(s, method, full_path, body, headers, query_string).await
}

async fn passthrough_native_versioned(
    State(s): State<AppState>,
    Path((version, path)): Path<(String, String)>,
    Query(query): Query<Vec<(String, String)>>,
    method: http::Method,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ProxyError> {
    if !version.starts_with('v') {
        return Err(ProxyError::not_found("endpoint not found"));
    }
    let full_path = format!("/api/{}/{}", version, path);
    let query_string = encode_query(&query);
    forward_passthrough(s, method, full_path, body, headers, query_string).await
}

async fn passthrough_native_version_root(
    State(s): State<AppState>,
    Path(version): Path<String>,
    Query(query): Query<Vec<(String, String)>>,
    method: http::Method,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ProxyError> {
    if !version.starts_with('v') {
        return Err(ProxyError::not_found("endpoint not found"));
    }
    let full_path = format!("/api/{}", version);
    let query_string = encode_query(&query);
    forward_passthrough(s, method, full_path, body, headers, query_string).await
}

async fn forward_passthrough(
    s: AppState,
    method: http::Method,
    full_path: String,
    body: Bytes,
    headers: HeaderMap,
    query: Option<String>,
) -> Result<Response, ProxyError> {
    let context = create_context(&s);
    lmstudio::handle_lmstudio_passthrough(
        context,
        s.model_resolver.clone(),
        lmstudio::LmStudioPassthroughRequest {
            method,
            endpoint: full_path,
            body,
            headers,
            query,
        },
        CancellationToken::new(),
        s.config.load_timeout_seconds,
    )
    .await
}

fn encode_query(pairs: &[(String, String)]) -> Option<String> {
    if pairs.is_empty() {
        return None;
    }
    let mut out = String::new();
    for (i, (k, v)) in pairs.iter().enumerate() {
        if i > 0 {
            out.push('&');
        }
        out.push_str(&urlencode(k));
        out.push('=');
        out.push_str(&urlencode(v));
    }
    Some(out)
}

fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}

// Re-export for tests/common to keep call sites stable.
pub use create_router as create_routes;
