use std::sync::Arc;

use bytes::Bytes;
use serde_json::Value;
use tokio_util::sync::CancellationToken;
use warp::Filter;
use warp::http::HeaderMap;

use crate::constants::MAX_JSON_BODY_SIZE_BYTES;
use crate::handlers::ollama::{EmbeddingResponseMode, handle_ollama_embeddings};
use crate::handlers::{RequestContext, lmstudio, ollama};
use crate::http::json_response;
use crate::server::ProxyServer;

pub fn create_routes(
    server: Arc<ProxyServer>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    let with_server_state = warp::any().map(move || server.clone());

    let health_route = warp::path!("health")
        .and(warp::get())
        .and(with_server_state.clone())
        .and_then(|s: Arc<ProxyServer>| async move {
            let context = create_context(&s);
            let token = CancellationToken::new();
            ollama::handle_health_check(context, token)
                .await
                .map(|value| json_response(&value))
                .map_err(warp::reject::custom)
        });

    let ollama_tags_route = warp::path!("api" / "tags")
        .and(warp::get())
        .and(with_server_state.clone())
        .and_then(|s: Arc<ProxyServer>| async move {
            let context = create_context(&s);
            let token = CancellationToken::new();
            ollama::handle_ollama_tags(context, s.model_resolver.clone(), token)
                .await
                .map_err(warp::reject::custom)
        });

    let ollama_chat_route = warp::path!("api" / "chat")
        .and(warp::post())
        .and(tolerant_json_body())
        .and(with_server_state.clone())
        .and_then(|body: Value, s: Arc<ProxyServer>| async move {
            let context = create_context(&s);
            let token = CancellationToken::new();
            ollama::handle_ollama_chat(
                context,
                s.model_resolver.clone(),
                body,
                token,
                s.config.load_timeout_seconds,
            )
            .await
            .map_err(warp::reject::custom)
        });

    let ollama_generate_route = warp::path!("api" / "generate")
        .and(warp::post())
        .and(tolerant_json_body())
        .and(with_server_state.clone())
        .and_then(|body: Value, s: Arc<ProxyServer>| async move {
            let context = create_context(&s);
            let token = CancellationToken::new();
            ollama::handle_ollama_generate(
                context,
                s.model_resolver.clone(),
                body,
                token,
                s.config.load_timeout_seconds,
            )
            .await
            .map_err(warp::reject::custom)
        });

    let embed_endpoint = warp::path!("api" / "embed").map(|| EmbeddingResponseMode::Embed);
    let embeddings_endpoint =
        warp::path!("api" / "embeddings").map(|| EmbeddingResponseMode::LegacyEmbeddings);

    let ollama_embeddings_route = embed_endpoint
        .or(embeddings_endpoint)
        .unify()
        .and(warp::post())
        .and(tolerant_json_body())
        .and(with_server_state.clone())
        .and_then(|mode, body: Value, s: Arc<ProxyServer>| async move {
            let context = create_context(&s);
            let token = CancellationToken::new();
            handle_ollama_embeddings(
                context,
                s.model_resolver.clone(),
                body,
                mode,
                token,
                s.config.load_timeout_seconds,
            )
            .await
            .map_err(warp::reject::custom)
        });

    let ollama_pull_route = warp::path!("api" / "pull")
        .and(warp::post())
        .and(tolerant_json_body())
        .and(with_server_state.clone())
        .and_then(|body: Value, s: Arc<ProxyServer>| async move {
            let context = create_context(&s);
            let token = CancellationToken::new();
            ollama::handle_ollama_pull(context, s.model_resolver.clone(), body, token)
                .await
                .map_err(warp::reject::custom)
        });

    let ollama_create_route = warp::path!("api" / "create")
        .and(warp::post())
        .and(tolerant_json_body())
        .and(with_server_state.clone())
        .and_then(|body: Value, s: Arc<ProxyServer>| async move {
            let context = create_context(&s);
            let token = CancellationToken::new();
            ollama::handle_ollama_create(context, s.model_resolver.clone(), body, token)
                .await
                .map_err(warp::reject::custom)
        });

    let ollama_copy_route = warp::path!("api" / "copy")
        .and(warp::post())
        .and(tolerant_json_body())
        .and(with_server_state.clone())
        .and_then(|body: Value, s: Arc<ProxyServer>| async move {
            let context = create_context(&s);
            let token = CancellationToken::new();
            ollama::handle_ollama_copy(context, s.model_resolver.clone(), body, token)
                .await
                .map_err(warp::reject::custom)
        });

    let ollama_delete_route = warp::path!("api" / "delete")
        .and(warp::delete())
        .and(tolerant_json_body())
        .and(with_server_state.clone())
        .and_then(|body: Value, s: Arc<ProxyServer>| async move {
            let context = create_context(&s);
            ollama::handle_ollama_delete(context, body)
                .await
                .map_err(warp::reject::custom)
        });

    let ollama_push_route = warp::path!("api" / "push")
        .and(warp::post())
        .and(tolerant_json_body())
        .and(with_server_state.clone())
        .and_then(|body: Value, s: Arc<ProxyServer>| async move {
            let context = create_context(&s);
            let token = CancellationToken::new();
            ollama::handle_ollama_push(context, s.model_resolver.clone(), body, token)
                .await
                .map_err(warp::reject::custom)
        });

    let ollama_show_route = warp::path!("api" / "show")
        .and(warp::post())
        .and(tolerant_json_body())
        .and(with_server_state.clone())
        .and_then(|body: Value, s: Arc<ProxyServer>| async move {
            let context = create_context(&s);
            let token = CancellationToken::new();
            ollama::handle_ollama_show(context, s.model_resolver.clone(), body, token)
                .await
                .map_err(warp::reject::custom)
        });

    let ollama_ps_route = warp::path!("api" / "ps")
        .and(warp::get())
        .and(with_server_state.clone())
        .and_then(|s: Arc<ProxyServer>| async move {
            let context = create_context(&s);
            let token = CancellationToken::new();
            ollama::handle_ollama_ps(context, s.model_resolver.clone(), token)
                .await
                .map_err(warp::reject::custom)
        });

    let ollama_version_route = warp::path!("api" / "version")
        .and(warp::get())
        .and(with_server_state.clone())
        .and_then(|_s: Arc<ProxyServer>| async move {
            ollama::handle_ollama_version()
                .await
                .map_err(warp::reject::custom)
        });

    let blob_head_route = warp::path!("api" / "blobs" / String)
        .and(warp::head())
        .and(with_server_state.clone())
        .and_then(|digest: String, s: Arc<ProxyServer>| async move {
            let context = create_context(&s);
            ollama::handle_blob_head(context, digest)
                .await
                .map_err(warp::reject::custom)
        });

    let blob_upload_route = warp::path!("api" / "blobs" / String)
        .and(warp::post())
        .and(warp::body::stream())
        .and(with_server_state.clone())
        .and_then(|digest: String, stream, s: Arc<ProxyServer>| async move {
            let context = create_context(&s);
            ollama::handle_blob_upload(context, digest, stream)
                .await
                .map_err(warp::reject::custom)
        });

    let lmstudio_passthrough_route = warp::path("v1")
        .and(warp::path::tail())
        .and(warp::method())
        .and(warp::body::bytes())
        .and(warp::header::headers_cloned())
        .and(
            warp::query::raw()
                .map(Some)
                .or(warp::any().map(|| None))
                .unify(),
        )
        .and(with_server_state.clone())
        .and_then(
            |tail: warp::path::Tail,
             method: warp::http::Method,
             body_bytes: Bytes,
             headers: HeaderMap,
             query: Option<String>,
             s: Arc<ProxyServer>| async move {
                let context = create_context(&s);
                let token = CancellationToken::new();
                let full_path = format!("/v1/{}", tail.as_str());
                lmstudio::handle_lmstudio_passthrough(
                    context,
                    s.model_resolver.clone(),
                    lmstudio::LmStudioPassthroughRequest {
                        method,
                        endpoint: full_path,
                        body: body_bytes,
                        headers,
                        query,
                    },
                    token,
                    s.config.load_timeout_seconds,
                )
                .await
                .map_err(warp::reject::custom)
            },
        );

    let lmstudio_native_passthrough_route = warp::path("api")
        .and(warp::path::param::<String>())
        .and(warp::path::tail())
        .and(warp::method())
        .and(warp::body::bytes())
        .and(warp::header::headers_cloned())
        .and(
            warp::query::raw()
                .map(Some)
                .or(warp::any().map(|| None))
                .unify(),
        )
        .and(with_server_state.clone())
        .and_then(
            |version: String,
             tail: warp::path::Tail,
             method: warp::http::Method,
             body_bytes: Bytes,
             headers: HeaderMap,
             query: Option<String>,
             s: Arc<ProxyServer>| async move {
                if !version.starts_with('v') {
                    return Err(warp::reject::not_found());
                }
                let context = create_context(&s);
                let token = CancellationToken::new();
                let tail_str = tail.as_str();
                let full_path = if tail_str.is_empty() {
                    format!("/api/{}", version)
                } else {
                    format!("/api/{}/{}", version, tail_str)
                };
                lmstudio::handle_lmstudio_passthrough(
                    context,
                    s.model_resolver.clone(),
                    lmstudio::LmStudioPassthroughRequest {
                        method,
                        endpoint: full_path,
                        body: body_bytes,
                        headers,
                        query,
                    },
                    token,
                    s.config.load_timeout_seconds,
                )
                .await
                .map_err(warp::reject::custom)
            },
        );

    health_route
        .or(ollama_tags_route)
        .or(ollama_chat_route)
        .or(ollama_generate_route)
        .or(ollama_embeddings_route)
        .or(ollama_pull_route)
        .or(ollama_create_route)
        .or(ollama_copy_route)
        .or(ollama_delete_route)
        .or(ollama_push_route)
        .or(ollama_show_route)
        .or(ollama_ps_route)
        .or(ollama_version_route)
        .or(blob_head_route)
        .or(blob_upload_route)
        .or(lmstudio_passthrough_route)
        .or(lmstudio_native_passthrough_route)
}

fn create_context(s: &Arc<ProxyServer>) -> RequestContext<'_> {
    RequestContext {
        client: &s.client,
        lmstudio_url: &s.config.lmstudio_url,
        virtual_models: s.virtual_models.clone(),
        blob_store: s.blob_store.clone(),
    }
}

fn tolerant_json_body() -> impl Filter<Extract = (Value,), Error = warp::Rejection> + Clone {
    warp::body::content_length_limit(MAX_JSON_BODY_SIZE_BYTES).and(warp::body::json())
}
