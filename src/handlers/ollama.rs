use bytes::{Buf, Bytes};
use futures_util::{Stream, TryStreamExt};
use humantime::parse_duration;
use reqwest::Method;
use serde::Deserialize;
use serde_json::{Value, json};
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;

use crate::common::{CancellableRequest, RequestContext, extract_model_name, handle_json_response};
use crate::constants::*;
use crate::handlers::helpers::{
    LMStudioRequestType, ResponseTransformer, build_lm_studio_request, execute_request_with_retry,
    extract_system_prompt, json_response,
};
use crate::handlers::retry::trigger_model_loading_for_ollama;
use crate::handlers::streaming::{
    create_ndjson_stream_response, handle_streaming_response, is_streaming_request,
};
use crate::model::{ModelInfo, clean_model_name};
use crate::server::{Config, ModelResolverType};
use crate::storage::virtual_models::{VirtualModelEntry, VirtualModelMetadata};
use crate::utils::{ProxyError, log_error, log_request, log_timed, log_warning};

const DOWNLOAD_STATUS_POLL_INTERVAL_MS: u64 = 900;

/// Response formatting mode for embeddings endpoints
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum EmbeddingResponseMode {
    /// /api/embed – modern endpoint that returns `embeddings`
    Embed,
    /// /api/embeddings – legacy endpoint that returns a single `embedding`
    LegacyEmbeddings,
}

/// Handle GET /api/tags - list available models
pub async fn handle_ollama_tags(
    context: RequestContext<'_>,
    model_resolver: ModelResolverType,
    cancellation_token: CancellationToken,
    config: &Config,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();
    if config.debug {
        println!("[DEBUG] Tags Request");
    }

    let operation = || {
        let context = context.clone();
        let model_resolver = model_resolver.clone();
        let cancellation_token = cancellation_token.clone();
        async move {
            match model_resolver {
                ModelResolverType::Native(resolver) => {
                    let models = resolver
                        .get_all_models(context.client, cancellation_token)
                        .await?;
                    let mut ollama_models: Vec<Value> = models
                        .iter()
                        .map(|model| model.to_ollama_tags_model())
                        .collect();

                    let model_lookup: HashMap<String, ModelInfo> = models
                        .into_iter()
                        .map(|model| (model.id.clone(), model))
                        .collect();
                    let alias_entries = context.virtual_models.list().await;
                    for alias in alias_entries {
                        if let Some(base_model) = model_lookup.get(&alias.target_model_id) {
                            let alias_info = base_model.with_alias_name(&alias.name);
                            ollama_models.push(alias_info.to_ollama_tags_model());
                        }
                    }
                    Ok(json!({ "models": ollama_models }))
                }
            }
        }
    };

    let result = execute_request_with_retry(
        &context,
        "_system_tags_",
        operation,
        false,
        0,
        cancellation_token.clone(),
    )
    .await
    .unwrap_or_else(|e| {
        log_error("Tags fetch", &e.message);
        json!({ "models": [] })
    });

    log_timed(LOG_PREFIX_SUCCESS, "Ollama tags", start_time);
    if config.debug {
        println!(
            "[DEBUG] Tags Response: {}",
            serde_json::to_string_pretty(&result).unwrap_or_default()
        );
    }
    Ok(json_response(&result))
}

/// Handle GET /api/ps - list running models
pub async fn handle_ollama_ps(
    context: RequestContext<'_>,
    model_resolver: ModelResolverType,
    cancellation_token: CancellationToken,
    config: &Config,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();
    log_request("GET", "/api/ps", None);
    if config.debug {
        println!("[DEBUG] PS Request");
    }

    let operation = || {
        let context = context.clone();
        let model_resolver = model_resolver.clone();
        let cancellation_token = cancellation_token.clone();
        async move {
            match model_resolver {
                ModelResolverType::Native(resolver) => {
                    let models = resolver
                        .get_loaded_models(context.client, cancellation_token)
                        .await?;
                    let mut ollama_models: Vec<Value> = models
                        .iter()
                        .map(|model| model.to_ollama_ps_model())
                        .collect();

                    let model_lookup: HashMap<String, ModelInfo> = models
                        .into_iter()
                        .map(|model| (model.id.clone(), model))
                        .collect();
                    let alias_entries = context.virtual_models.list().await;
                    for alias in alias_entries {
                        if let Some(base_model) = model_lookup.get(&alias.target_model_id) {
                            let alias_info = base_model.with_alias_name(&alias.name);
                            ollama_models.push(alias_info.to_ollama_ps_model());
                        }
                    }
                    Ok(json!({ "models": ollama_models }))
                }
            }
        }
    };

    let result = execute_request_with_retry(
        &context,
        "_system_ps_",
        operation,
        false,
        0,
        cancellation_token.clone(),
    )
    .await
    .unwrap_or_else(|e| {
        log_error("PS fetch", &e.message);
        json!({ "models": [] })
    });

    log_timed(LOG_PREFIX_SUCCESS, "Ollama ps", start_time);
    if config.debug {
        println!(
            "[DEBUG] PS Response: {}",
            serde_json::to_string_pretty(&result).unwrap_or_default()
        );
    }
    Ok(json_response(&result))
}

/// Handle POST /api/show - show model info
pub async fn handle_ollama_show(
    context: RequestContext<'_>,
    model_resolver: ModelResolverType,
    body: Value,
    cancellation_token: CancellationToken,
    config: &Config,
) -> Result<warp::reply::Response, ProxyError> {
    if config.debug {
        println!(
            "[DEBUG] Show Request: {}",
            serde_json::to_string_pretty(&body).unwrap_or_default()
        );
    }
    let ollama_model_name = extract_model_name(&body, "model")?;

    let virtual_entry = context.virtual_models.get(ollama_model_name).await;

    let resolved_id = if let Some(entry) = &virtual_entry {
        entry.target_model_id.clone()
    } else {
        match model_resolver {
            ModelResolverType::Native(ref resolver) => {
                resolver
                    .resolve_model_name(
                        ollama_model_name,
                        context.client,
                        cancellation_token.clone(),
                    )
                    .await?
            }
        }
    };

    let response = match model_resolver {
        ModelResolverType::Native(resolver) => {
            let models = resolver
                .get_all_models(context.client, cancellation_token.clone())
                .await?;

            if let Some(model_info) = models.into_iter().find(|model| model.id == resolved_id) {
                let display_info = if virtual_entry.is_some() {
                    model_info.with_alias_name(ollama_model_name)
                } else {
                    model_info
                };
                let mut payload = display_info.to_show_response();
                if let Some(entry) = virtual_entry
                    && let Some(obj) = payload.as_object_mut()
                {
                    obj.insert(
                        "proxy_virtual_model".to_string(),
                        json!({
                            "source_model": entry.source_model,
                            "target_model_id": entry.target_model_id,
                            "metadata": entry.metadata,
                            "created_at": entry.created_at.to_rfc3339(),
                            "updated_at": entry.updated_at.to_rfc3339(),
                        }),
                    );
                }
                payload
            } else {
                return Err(ProxyError::not_found(&format!(
                    "Model '{}' metadata unavailable in LM Studio",
                    ollama_model_name
                )));
            }
        }
    };

    if config.debug {
        println!(
            "[DEBUG] Show Response: {}",
            serde_json::to_string_pretty(&response).unwrap_or_default()
        );
    }
    Ok(json_response(&response))
}

/// Handle POST /api/chat - chat completion with streaming support
pub async fn handle_ollama_chat(
    context: RequestContext<'_>,
    model_resolver: ModelResolverType,
    body: Value,
    cancellation_token: CancellationToken,
    config: &Config,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();
    let ollama_model_name = extract_model_name(&body, "model")?;
    let keep_alive_seconds = parse_keep_alive_seconds(body.get("keep_alive"))?;

    let messages = body
        .get("messages")
        .and_then(|m| m.as_array())
        .ok_or_else(|| ProxyError::bad_request(ERROR_MISSING_MESSAGES))?;

    // Empty messages trigger
    if messages.is_empty() {
        let (raw_model_id, _) = resolve_model_target(
            &context,
            &model_resolver,
            ollama_model_name,
            cancellation_token.clone(),
        )
        .await?;

        if keep_alive_requests_unload(keep_alive_seconds) {
            log_timed(
                LOG_PREFIX_INFO,
                &format!("Unload hint for {}", ollama_model_name),
                start_time,
            );
            let fabricated_response = json!({
                "model": ollama_model_name,
                "created_at": chrono::Utc::now().to_rfc3339(),
                "message": {"role": "assistant", "content": ""},
                "done_reason": "unload",
                "done": true
            });
            log_timed(LOG_PREFIX_SUCCESS, "Ollama chat (unload hint)", start_time);
            return Ok(json_response(&fabricated_response));
        }

        log_timed(
            LOG_PREFIX_INFO,
            &format!("Load hint for {}", ollama_model_name),
            start_time,
        );
        trigger_model_loading_for_ollama(&context, &raw_model_id, cancellation_token.clone())
            .await?;
        let fabricated_response = json!({
            "model": ollama_model_name,
            "created_at": chrono::Utc::now().to_rfc3339(),
            "message": {"role": "assistant", "content": ""},
            "done_reason": "load",
            "done": true
        });
        log_timed(LOG_PREFIX_SUCCESS, "Ollama chat (load hint)", start_time);
        return Ok(json_response(&fabricated_response));
    }

    let (resolved_model_id, resolved_virtual_entry) = resolve_model_target(
        &context,
        &model_resolver,
        ollama_model_name,
        cancellation_token.clone(),
    )
    .await?;
    let resolved_model_id_for_retry = resolved_model_id.clone();
    let resolved_model_id_shared = Arc::new(resolved_model_id);
    let resolved_virtual_entry_shared = Arc::new(resolved_virtual_entry);
    let debug = config.debug;

    let operation = || {
        let context = context.clone();
        let model_resolver = model_resolver.clone();
        let body_clone = body.clone();
        let cancellation_token_clone = cancellation_token.clone();
        let ollama_model_name_clone = ollama_model_name.to_string();
        let resolved_model_id = resolved_model_id_shared.clone();
        let resolved_virtual_entry = resolved_virtual_entry_shared.clone();
        let keep_alive_seconds_for_request = keep_alive_seconds;

        async move {
            if debug {
                println!(
                    "[DEBUG] Chat Request: {}",
                    serde_json::to_string_pretty(&body_clone).unwrap_or_default()
                );
            }

            let current_messages = body_clone
                .get("messages")
                .and_then(|m| m.as_array())
                .ok_or_else(|| ProxyError::bad_request(ERROR_MISSING_MESSAGES))?;
            let mut provided_system_prompt = extract_system_prompt(&body_clone);
            let stream = is_streaming_request(&body_clone);
            let lm_studio_model_id = resolved_model_id.as_ref().clone();
            let virtual_model_entry = resolved_virtual_entry.as_ref().clone();

            if provided_system_prompt.is_none() {
                provided_system_prompt = virtual_model_entry
                    .as_ref()
                    .and_then(|entry| entry.metadata.system_prompt.clone());
            }

            let normalized_messages =
                normalize_chat_messages(current_messages, provided_system_prompt.as_deref());

            let merged_options_owned = merge_option_maps(
                virtual_model_entry
                    .as_ref()
                    .and_then(|entry| entry.metadata.parameters.as_ref()),
                body_clone.get("options"),
            );
            let ollama_options = match merged_options_owned.as_ref() {
                Some(value) => Some(value),
                None => body_clone.get("options"),
            };
            let ollama_tools = body_clone.get("tools");
            let structured_format = body_clone.get("format");
            let endpoint_url = format!("{}{}", context.lmstudio_url, LM_STUDIO_NATIVE_CHAT);

            let mut lm_request = build_lm_studio_request(
                &lm_studio_model_id,
                LMStudioRequestType::Chat {
                    messages: &normalized_messages,
                    stream,
                },
                ollama_options,
                ollama_tools,
                structured_format,
            );
            apply_keep_alive_ttl(&mut lm_request, keep_alive_seconds_for_request);

            let request_obj =
                CancellableRequest::new(context.client, cancellation_token_clone.clone());
            log_request("POST", &endpoint_url, Some(&lm_studio_model_id));

            let response = request_obj
                .make_request(Method::POST, &endpoint_url, Some(lm_request))
                .await?;

            if stream {
                handle_streaming_response(
                    response,
                    true,
                    &ollama_model_name_clone,
                    start_time,
                    cancellation_token_clone.clone(),
                    60,
                )
                .await
            } else {
                let lm_response_value =
                    handle_json_response(response, cancellation_token_clone).await?;
                let ollama_response = ResponseTransformer::convert_to_ollama_chat(
                    &lm_response_value,
                    &ollama_model_name_clone,
                    current_messages.len(),
                    start_time,
                    matches!(model_resolver, ModelResolverType::Native(_)),
                );
                if debug {
                    println!(
                        "[DEBUG] Chat Response: {}",
                        serde_json::to_string_pretty(&ollama_response).unwrap_or_default()
                    );
                }
                Ok(json_response(&ollama_response))
            }
        }
    };

    let result = execute_request_with_retry(
        &context,
        &resolved_model_id_for_retry,
        operation,
        true,
        config.load_timeout_seconds,
        cancellation_token.clone(),
    )
    .await?;

    log_timed(LOG_PREFIX_SUCCESS, "Ollama chat", start_time);
    Ok(result)
}

/// Handle POST /api/generate - text completion with streaming support
pub async fn handle_ollama_generate(
    context: RequestContext<'_>,
    model_resolver: ModelResolverType,
    body: Value,
    cancellation_token: CancellationToken,
    config: &Config,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();
    let ollama_model_name = extract_model_name(&body, "model")?;

    let prompt = body
        .get("prompt")
        .and_then(|p| p.as_str())
        .ok_or_else(|| ProxyError::bad_request(ERROR_MISSING_PROMPT))?;
    let images = body.get("images");
    let keep_alive_seconds = parse_keep_alive_seconds(body.get("keep_alive"))?;

    let (resolved_model_id, resolved_virtual_entry) = resolve_model_target(
        &context,
        &model_resolver,
        ollama_model_name,
        cancellation_token.clone(),
    )
    .await?;

    let resolved_model_id_for_retry = resolved_model_id.clone();
    let images_empty = images.is_none_or(|i| i.as_array().is_none_or(|a| a.is_empty()));
    let debug = config.debug;

    // Empty prompt trigger
    if prompt.is_empty() && images_empty {
        if keep_alive_requests_unload(keep_alive_seconds) {
            log_timed(
                LOG_PREFIX_INFO,
                &format!("Unload hint for {}", ollama_model_name),
                start_time,
            );
            let fabricated_response = json!({
                "model": ollama_model_name,
                "created_at": chrono::Utc::now().to_rfc3339(),
                "response": "",
                "done_reason": "unload",
                "done": true
            });
            log_timed(
                LOG_PREFIX_SUCCESS,
                "Ollama generate (unload hint)",
                start_time,
            );
            return Ok(json_response(&fabricated_response));
        }

        log_timed(
            LOG_PREFIX_INFO,
            &format!("Load hint for {}", ollama_model_name),
            start_time,
        );
        trigger_model_loading_for_ollama(&context, &resolved_model_id, cancellation_token.clone())
            .await?;
        let fabricated_response = json!({
            "model": ollama_model_name,
            "created_at": chrono::Utc::now().to_rfc3339(),
            "response": "",
            "done_reason": "load",
            "done": true
        });
        log_timed(
            LOG_PREFIX_SUCCESS,
            "Ollama generate (load hint)",
            start_time,
        );
        return Ok(json_response(&fabricated_response));
    }

    let operation = || {
        let context = context.clone();
        let model_resolver = model_resolver.clone();
        let body_clone = body.clone();
        let cancellation_token_clone = cancellation_token.clone();
        let ollama_model_name_clone = ollama_model_name.to_string();
        let resolved_model_id = resolved_model_id.clone();
        let resolved_virtual_entry = resolved_virtual_entry.clone();
        let keep_alive_seconds_for_request = keep_alive_seconds;

        async move {
            if debug {
                println!(
                    "[DEBUG] Generate Request: {}",
                    serde_json::to_string_pretty(&body_clone).unwrap_or_default()
                );
            }

            let current_prompt = body_clone
                .get("prompt")
                .and_then(|p| p.as_str())
                .ok_or_else(|| ProxyError::bad_request(ERROR_MISSING_PROMPT))?;
            let current_images = body_clone.get("images");
            let stream = is_streaming_request(&body_clone);
            let ollama_options = body_clone.get("options");
            let structured_format = body_clone.get("format");
            let provided_system_prompt = extract_system_prompt(&body_clone);

            let endpoint_url_base = context.lmstudio_url.to_string();
            let lm_studio_model_id = resolved_model_id.clone();
            let virtual_model_entry = resolved_virtual_entry.clone();

            let mut applied_system_prompt = provided_system_prompt;
            if applied_system_prompt.is_none() {
                applied_system_prompt = virtual_model_entry
                    .as_ref()
                    .and_then(|entry| entry.metadata.system_prompt.clone());
            }

            let merged_options_owned = merge_option_maps(
                virtual_model_entry
                    .as_ref()
                    .and_then(|entry| entry.metadata.parameters.as_ref()),
                ollama_options,
            );

            let effective_options_value = merged_options_owned
                .clone()
                .or_else(|| ollama_options.cloned());

            let struct_format_override = virtual_model_entry
                .as_ref()
                .and_then(|entry| entry.metadata.parameters.as_ref())
                .and_then(|value| value.get("format"));

            let effective_format_value = struct_format_override
                .cloned()
                .or_else(|| structured_format.cloned());

            let mut chat_messages_payload: Option<Value> = None;
            let mut prompt_override_storage: Option<String> = None;
            let mut prompt_for_estimation = current_prompt;

            let images_present = current_images
                .and_then(|img| img.as_array())
                .map(|arr| !arr.is_empty())
                .unwrap_or(false);

            // Determine endpoint based on API type and whether images are present
            let (lm_studio_target_url, lm_request_type) = if images_present {
                let chat_endpoint = match &model_resolver {
                    ModelResolverType::Native(_) => LM_STUDIO_NATIVE_CHAT,
                };

                let mut message_list = Vec::new();
                if let Some(system_text) = applied_system_prompt.as_deref()
                    && !system_text.trim().is_empty()
                {
                    message_list.push(json!({
                        "role": "system",
                        "content": system_text,
                    }));
                }

                let mut user_message = json!({
                    "role": "user",
                    "content": current_prompt,
                });
                if let Some(images_val) = current_images
                    && let Some(obj) = user_message.as_object_mut()
                {
                    obj.insert("images".to_string(), images_val.clone());
                }
                message_list.push(user_message);
                chat_messages_payload = Some(Value::Array(message_list));
                let messages_ref = chat_messages_payload.as_ref().unwrap();

                (
                    format!("{}{}", endpoint_url_base, chat_endpoint),
                    LMStudioRequestType::Chat {
                        messages: messages_ref,
                        stream,
                    },
                )
            } else {
                if let Some(system_text) = applied_system_prompt.as_deref() {
                    let trimmed = system_text.trim();
                    if !trimmed.is_empty() {
                        let combined = if current_prompt.is_empty() {
                            trimmed.to_string()
                        } else {
                            format!("{trimmed}\n\n{current_prompt}")
                        };
                        prompt_override_storage = Some(combined);
                    }
                }

                if let Some(override_str) = prompt_override_storage.as_deref() {
                    prompt_for_estimation = override_str;
                }

                let effective_prompt = prompt_override_storage.as_deref().unwrap_or(current_prompt);

                let completions_endpoint = match &model_resolver {
                    ModelResolverType::Native(_) => LM_STUDIO_NATIVE_COMPLETIONS,
                };
                (
                    format!("{}{}", endpoint_url_base, completions_endpoint),
                    LMStudioRequestType::Completion {
                        prompt: Cow::Borrowed(effective_prompt),
                        stream,
                    },
                )
            };

            let _ = &chat_messages_payload;

            let effective_options_ref: Option<&Value> =
                effective_options_value.as_ref().or(ollama_options);
            let effective_format_ref: Option<&Value> =
                effective_format_value.as_ref().or(structured_format);

            let mut lm_request = build_lm_studio_request(
                &lm_studio_model_id,
                lm_request_type,
                effective_options_ref,
                None,
                effective_format_ref,
            );
            apply_keep_alive_ttl(&mut lm_request, keep_alive_seconds_for_request);

            let request_obj =
                CancellableRequest::new(context.client, cancellation_token_clone.clone());
            log_request("POST", &lm_studio_target_url, Some(&lm_studio_model_id));

            let response = request_obj
                .make_request(Method::POST, &lm_studio_target_url, Some(lm_request))
                .await?;

            if stream {
                handle_streaming_response(
                    response,
                    false,
                    &ollama_model_name_clone,
                    start_time,
                    cancellation_token_clone.clone(),
                    60,
                )
                .await
            } else {
                let lm_response_value =
                    handle_json_response(response, cancellation_token_clone).await?;
                let ollama_response = ResponseTransformer::convert_to_ollama_generate(
                    &lm_response_value,
                    &ollama_model_name_clone,
                    prompt_for_estimation,
                    start_time,
                    matches!(model_resolver, ModelResolverType::Native(_)),
                );
                if debug {
                    println!(
                        "[DEBUG] Generate Response: {}",
                        serde_json::to_string_pretty(&ollama_response).unwrap_or_default()
                    );
                }
                Ok(json_response(&ollama_response))
            }
        }
    };

    let result = execute_request_with_retry(
        &context,
        &resolved_model_id_for_retry,
        operation,
        true,
        config.load_timeout_seconds,
        cancellation_token.clone(),
    )
    .await?;

    log_timed(LOG_PREFIX_SUCCESS, "Ollama generate", start_time);
    Ok(result)
}

/// Handle POST /api/embed or /api/embeddings - generate embeddings
pub async fn handle_ollama_embeddings(
    context: RequestContext<'_>,
    model_resolver: ModelResolverType,
    body: Value,
    response_mode: EmbeddingResponseMode,
    cancellation_token: CancellationToken,
    config: &Config,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();
    let ollama_model_name = extract_model_name(&body, "model")?;
    let keep_alive_seconds = parse_keep_alive_seconds(body.get("keep_alive"))?;
    let debug = config.debug;

    let operation = || {
        let context = context.clone();
        let model_resolver = model_resolver.clone();
        let body_clone = body.clone();
        let cancellation_token_clone = cancellation_token.clone();
        let ollama_model_name_clone = ollama_model_name.to_string();
        let response_mode_for_request = response_mode;
        let keep_alive_seconds_for_request = keep_alive_seconds;

        async move {
            if debug {
                println!(
                    "[DEBUG] Embeddings Request: {}",
                    serde_json::to_string_pretty(&body_clone).unwrap_or_default()
                );
            }

            let current_ollama_model_name = extract_model_name(&body_clone, "model")?;
            let input_value = body_clone
                .get("input")
                .or_else(|| body_clone.get("prompt"))
                .cloned()
                .ok_or_else(|| ProxyError::bad_request(ERROR_MISSING_INPUT))?;

            let (lm_studio_model_id, virtual_model_entry) = resolve_model_target(
                &context,
                &model_resolver,
                current_ollama_model_name,
                cancellation_token_clone.clone(),
            )
            .await?;
            let endpoint_url = format!("{}{}", context.lmstudio_url, LM_STUDIO_NATIVE_EMBEDDINGS);

            let merged_options_owned = merge_option_maps(
                virtual_model_entry
                    .as_ref()
                    .and_then(|entry| entry.metadata.parameters.as_ref()),
                body_clone.get("options"),
            );
            let effective_options_ref = merged_options_owned
                .as_ref()
                .or_else(|| body_clone.get("options"));

            let mut lm_request = build_lm_studio_request(
                &lm_studio_model_id,
                LMStudioRequestType::Embeddings {
                    input: &input_value,
                },
                effective_options_ref,
                None,
                None,
            );
            apply_keep_alive_ttl(&mut lm_request, keep_alive_seconds_for_request);

            let request_obj =
                CancellableRequest::new(context.client, cancellation_token_clone.clone());
            log_request("POST", &endpoint_url, Some(&lm_studio_model_id));

            let response = request_obj
                .make_request(Method::POST, &endpoint_url, Some(lm_request))
                .await?;
            let lm_response_value =
                handle_json_response(response, cancellation_token_clone).await?;

            let ollama_response = ResponseTransformer::convert_to_ollama_embeddings(
                &lm_response_value,
                &ollama_model_name_clone,
                start_time,
                matches!(model_resolver, ModelResolverType::Native(_)),
            );
            let final_payload =
                finalize_embedding_response(ollama_response, response_mode_for_request);
            if debug {
                println!(
                    "[DEBUG] Embeddings Response: {}",
                    serde_json::to_string_pretty(&final_payload).unwrap_or_default()
                );
            }
            Ok(json_response(&final_payload))
        }
    };

    let result = execute_request_with_retry(
        &context,
        ollama_model_name,
        operation,
        true,
        config.load_timeout_seconds,
        cancellation_token.clone(),
    )
    .await?;

    log_timed(LOG_PREFIX_SUCCESS, "Ollama embeddings", start_time);
    Ok(result)
}

/// Handle POST /api/pull - download models via LM Studio catalog
pub async fn handle_ollama_pull(
    context: RequestContext<'_>,
    model_resolver: ModelResolverType,
    body: Value,
    cancellation_token: CancellationToken,
    config: &Config,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();
    if config.debug {
        println!(
            "[DEBUG] Pull Request: {}",
            serde_json::to_string_pretty(&body).unwrap_or_default()
        );
    }
    let requested_model = extract_model_name(&body, "model")?;
    let stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(true);
    let quantization = body
        .get("quantization")
        .and_then(|q| q.as_str())
        .map(|s| s.to_string());
    let source_override = body.get("source").and_then(|s| s.as_str());

    log_request("POST", "/api/pull", Some(requested_model));

    let client = context.client.clone();
    let base_url = context.lmstudio_url.to_string();

    let resolved_model_context =
        if source_override.is_none() && !looks_like_remote_identifier(requested_model) {
            Some(
                resolve_model_target(
                    &context,
                    &model_resolver,
                    requested_model,
                    cancellation_token.clone(),
                )
                .await?,
            )
        } else {
            None
        };

    let download_identifier = determine_download_identifier(
        &context,
        &model_resolver,
        requested_model,
        source_override,
        resolved_model_context,
        cancellation_token.clone(),
    )
    .await?;

    let initial_status = initiate_lmstudio_download(
        &client,
        &base_url,
        &download_identifier,
        quantization.as_deref(),
        cancellation_token.clone(),
    )
    .await?;

    if !stream {
        let final_status = if initial_status.is_terminal() {
            initial_status
        } else {
            wait_for_download_completion(
                &client,
                &base_url,
                initial_status,
                cancellation_token.clone(),
            )
            .await?
        };

        let response_body = final_status.into_final_response(requested_model)?;
        log_timed(LOG_PREFIX_SUCCESS, "Ollama pull", start_time);
        if config.debug {
            println!(
                "[DEBUG] Pull Response: {}",
                serde_json::to_string_pretty(&response_body).unwrap_or_default()
            );
        }
        return Ok(json_response(&response_body));
    }

    let (tx, rx) = mpsc::unbounded_channel();
    let stream_client = client.clone();
    let stream_base_url = base_url.clone();
    let model_for_stream = requested_model.to_string();
    let token_for_stream = cancellation_token.clone();

    tokio::spawn(async move {
        let progress_sender = tx.clone();
        if let Err(e) = stream_download_status_updates(
            stream_client,
            stream_base_url,
            initial_status,
            model_for_stream.clone(),
            token_for_stream,
            progress_sender,
        )
        .await
        {
            log_error("Ollama pull stream", &e.message);
            send_status_error_chunk(&tx, &model_for_stream, &e.message);
        }
    });

    let response = create_ndjson_stream_response(rx, "Failed to create pull streaming response")?;
    log_timed(LOG_PREFIX_CONN, "Ollama pull stream open", start_time);
    if config.debug {
        println!("[DEBUG] Pull Response: (Streaming)");
    }
    Ok(response)
}

/// Handle POST /api/create - virtual model aliases backed by LM Studio models
pub async fn handle_ollama_create(
    context: RequestContext<'_>,
    model_resolver: ModelResolverType,
    body: Value,
    cancellation_token: CancellationToken,
    config: &Config,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();
    if config.debug {
        println!(
            "[DEBUG] Create Request: {}",
            serde_json::to_string_pretty(&body).unwrap_or_default()
        );
    }
    let new_model_name = extract_model_name(&body, "model")?;
    let source_model_name = body
        .get("from")
        .and_then(|value| value.as_str())
        .unwrap_or(new_model_name);
    let stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(true);

    log_request("POST", "/api/create", Some(new_model_name));

    let entry = create_virtual_model_alias(
        &context,
        &model_resolver,
        new_model_name,
        source_model_name,
        &body,
        cancellation_token,
    )
    .await?;

    log_timed(LOG_PREFIX_SUCCESS, "Ollama create", start_time);

    if stream {
        let statuses = vec![
            json!({"status": "reading model metadata", "model": new_model_name}),
            json!({
                "status": "creating alias",
                "model": new_model_name,
                "source": source_model_name,
                "target_model_id": entry.target_model_id
            }),
            json!({"status": "writing manifest", "model": new_model_name}),
            json!({"status": "success", "model": new_model_name, "virtual": true}),
        ];
        if config.debug {
            println!("[DEBUG] Create Response: (Streaming)");
        }
        return stream_status_messages(statuses, "Failed to create model alias stream");
    }

    let response = build_virtual_model_response(&entry);
    if config.debug {
        println!(
            "[DEBUG] Create Response: {}",
            serde_json::to_string_pretty(&response).unwrap_or_default()
        );
    }
    Ok(json_response(&response))
}

/// Handle POST /api/copy - duplicate existing aliases or create new ones referencing LM Studio models
pub async fn handle_ollama_copy(
    context: RequestContext<'_>,
    model_resolver: ModelResolverType,
    body: Value,
    cancellation_token: CancellationToken,
    config: &Config,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();
    if config.debug {
        println!(
            "[DEBUG] Copy Request: {}",
            serde_json::to_string_pretty(&body).unwrap_or_default()
        );
    }
    let source = body
        .get("source")
        .and_then(|value| value.as_str())
        .ok_or_else(|| ProxyError::bad_request("Missing 'source' field"))?;
    let destination = body
        .get("destination")
        .and_then(|value| value.as_str())
        .ok_or_else(|| ProxyError::bad_request("Missing 'destination' field"))?;

    log_request("POST", "/api/copy", Some(destination));

    let entry = if let Some(existing) = context.virtual_models.get(source).await {
        context
            .virtual_models
            .create_alias(
                destination,
                existing.source_model.clone(),
                existing.target_model_id.clone(),
                existing.metadata.clone(),
            )
            .await?
    } else {
        let (resolved_id, _) =
            resolve_model_target(&context, &model_resolver, source, cancellation_token).await?;

        context
            .virtual_models
            .create_alias(
                destination,
                source.to_string(),
                resolved_id,
                VirtualModelMetadata::default(),
            )
            .await?
    };

    log_timed(LOG_PREFIX_SUCCESS, "Ollama copy", start_time);
    let response = build_virtual_model_response(&entry);
    if config.debug {
        println!(
            "[DEBUG] Copy Response: {}",
            serde_json::to_string_pretty(&response).unwrap_or_default()
        );
    }
    Ok(json_response(&response))
}

/// Handle DELETE /api/delete - remove virtual aliases if managed by the proxy
pub async fn handle_ollama_delete(
    context: RequestContext<'_>,
    body: Value,
    config: &Config,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();
    if config.debug {
        println!(
            "[DEBUG] Delete Request: {}",
            serde_json::to_string_pretty(&body).unwrap_or_default()
        );
    }
    let model_name = extract_model_name(&body, "model")?;
    log_request("DELETE", "/api/delete", Some(model_name));

    if context.virtual_models.get(model_name).await.is_none() {
        return Err(ProxyError::not_found(&format!(
            "Model '{}' not managed by this proxy",
            model_name
        )));
    }

    let removed = context.virtual_models.delete(model_name).await?;
    log_timed(LOG_PREFIX_SUCCESS, "Ollama delete", start_time);
    let response = json!({
        "status": "success",
        "model": removed.name,
        "virtual": true
    });
    if config.debug {
        println!(
            "[DEBUG] Delete Response: {}",
            serde_json::to_string_pretty(&response).unwrap_or_default()
        );
    }
    Ok(json_response(&response))
}

/// Handle POST /api/push - validate model availability and acknowledge request
pub async fn handle_ollama_push(
    context: RequestContext<'_>,
    model_resolver: ModelResolverType,
    body: Value,
    cancellation_token: CancellationToken,
    config: &Config,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();
    if config.debug {
        println!(
            "[DEBUG] Push Request: {}",
            serde_json::to_string_pretty(&body).unwrap_or_default()
        );
    }
    let model_name = extract_model_name(&body, "model")?;
    let stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(true);

    log_request("POST", "/api/push", Some(model_name));

    // Validate model exists either as alias or in LM Studio
    let (resolved_model_id, _) =
        resolve_model_target(&context, &model_resolver, model_name, cancellation_token).await?;

    log_timed(LOG_PREFIX_SUCCESS, "Ollama push (noop)", start_time);

    if stream {
        let statuses = vec![
            json!({"status": "retrieving manifest", "model": model_name}),
            json!({
                "status": "starting upload",
                "model": model_name,
                "target_model_id": resolved_model_id
            }),
            json!({"status": "pushing manifest", "model": model_name}),
            json!({
                "status": "success",
                "model": model_name,
                "detail": "Push is a no-op when targeting LM Studio"
            }),
        ];
        if config.debug {
            println!("[DEBUG] Push Response: (Streaming)");
        }
        return stream_status_messages(statuses, "Failed to stream push status");
    }

    let response = json!({
        "status": "success",
        "model": model_name,
        "detail": "Push is a no-op when targeting LM Studio",
        "target_model_id": resolved_model_id
    });
    if config.debug {
        println!(
            "[DEBUG] Push Response: {}",
            serde_json::to_string_pretty(&response).unwrap_or_default()
        );
    }
    Ok(json_response(&response))
}

/// Handle HEAD /api/blobs/:digest - check if proxy-managed blob exists
pub async fn handle_blob_head(
    context: RequestContext<'_>,
    digest: String,
    config: &Config,
) -> Result<warp::reply::Response, ProxyError> {
    if config.debug {
        println!("[DEBUG] Blob Head Request: {}", digest);
    }
    let exists = context.blob_store.exists(&digest).await?;
    let status = if exists {
        warp::http::StatusCode::OK
    } else {
        warp::http::StatusCode::NOT_FOUND
    };

    if config.debug {
        println!("[DEBUG] Blob Head Response: {}", status);
    }

    warp::http::Response::builder()
        .status(status)
        .body(warp::hyper::Body::empty())
        .map_err(|_| ProxyError::internal_server_error("Failed to build blob response"))
}

/// Handle POST /api/blobs/:digest - store uploaded blobs for future model creation
pub async fn handle_blob_upload<S, B>(
    context: RequestContext<'_>,
    digest: String,
    stream: S,
    config: &Config,
) -> Result<warp::reply::Response, ProxyError>
where
    S: Stream<Item = Result<B, warp::Error>> + Unpin,
    B: Buf,
{
    let start_time = Instant::now();
    log_request("POST", "/api/blobs", Some(&digest));
    if config.debug {
        println!("[DEBUG] Blob Upload Request: {}", digest);
    }

    let byte_stream = stream.map_ok(|mut buf| buf.copy_to_bytes(buf.remaining()));

    context.blob_store.save_stream(&digest, byte_stream).await?;

    log_timed(
        LOG_PREFIX_SUCCESS,
        &format!("Stored blob {}", digest),
        start_time,
    );

    if config.debug {
        println!("[DEBUG] Blob Upload Response: Created");
    }

    warp::http::Response::builder()
        .status(warp::http::StatusCode::CREATED)
        .body(warp::hyper::Body::empty())
        .map_err(|_| ProxyError::internal_server_error("Failed to build blob upload response"))
}

/// Handle GET /api/version - return version info
pub async fn handle_ollama_version(config: &Config) -> Result<warp::reply::Response, ProxyError> {
    if config.debug {
        println!("[DEBUG] Version Request");
    }
    let response = json!({
        "version": OLLAMA_SERVER_VERSION
    });
    if config.debug {
        println!(
            "[DEBUG] Version Response: {}",
            serde_json::to_string_pretty(&response).unwrap_or_default()
        );
    }
    Ok(json_response(&response))
}

/// Handle unsupported endpoints with helpful messages
#[allow(dead_code)]
pub async fn handle_unsupported(endpoint: &str) -> Result<warp::reply::Response, ProxyError> {
    let (message, suggestion) = match endpoint {
        "/api/create" => (
            "Model creation not supported via proxy",
            "Load models directly in LM Studio",
        ),
        "/api/pull" => (
            "Model pulling not supported via proxy",
            "Download models through LM Studio interface",
        ),
        "/api/push" => (
            "Model pushing not supported via proxy",
            "Use LM Studio for model management",
        ),
        "/api/delete" => (
            "Model deletion not supported via proxy",
            "Remove models through LM Studio",
        ),
        "/api/copy" => (
            "Model copying not supported via proxy",
            "Use LM Studio for model operations",
        ),
        _ => (
            "Endpoint requires direct Ollama functionality not available via this LM Studio proxy",
            "This proxy focuses on inference and basic model listing operations",
        ),
    };

    Err(ProxyError::not_implemented(&format!(
        "{}. Suggestion: {}.",
        message, suggestion
    )))
}

/// Handle health check that tests actual model availability
pub async fn handle_health_check(
    context: RequestContext<'_>,
    cancellation_token: CancellationToken,
    config: &Config,
) -> Result<Value, ProxyError> {
    let start_time = Instant::now();
    if config.debug {
        println!("[DEBUG] Health Check Request");
    }
    let url = format!("{}{}", context.lmstudio_url, LM_STUDIO_NATIVE_MODELS);
    let request = CancellableRequest::new(context.client, cancellation_token.clone());

    match request.make_request(Method::GET, &url, None::<Value>).await {
        Ok(response) => {
            let status = response.status();
            let is_healthy = status.is_success();
            let mut model_count = 0;

            if is_healthy && let Ok(models_response) = response.json::<Value>().await {
                model_count = models_response
                    .get("models")
                    .or_else(|| models_response.get("data"))
                    .and_then(|d| d.as_array())
                    .map(|arr| arr.len())
                    .unwrap_or(0);
            }

            log_timed(
                if is_healthy {
                    LOG_PREFIX_SUCCESS
                } else {
                    LOG_PREFIX_ERROR
                },
                &format!("Health check - {} models", model_count),
                start_time,
            );

            let response = json!({
                "status": if is_healthy { "healthy" } else { "unhealthy" },
                "lmstudio_url": context.lmstudio_url,
                "http_status": status.as_u16(),
                "models_known_to_lmstudio": model_count,
                "response_time_ms": start_time.elapsed().as_millis(),
                "timestamp": chrono::Utc::now().to_rfc3339(),
                "proxy_version": crate::VERSION
            });
            if config.debug {
                println!(
                    "[DEBUG] Health Check Response: {}",
                    serde_json::to_string_pretty(&response).unwrap_or_default()
                );
            }
            Ok(response)
        }
        Err(e) if e.is_cancelled() => Err(ProxyError::request_cancelled()),
        Err(e) => {
            log_timed(
                LOG_PREFIX_ERROR,
                &format!("Health check failed: {}", e.message),
                start_time,
            );
            let response = json!({
                "status": "unreachable",
                "lmstudio_url": context.lmstudio_url,
                "error_message": e.message,
                "error_details": ERROR_LM_STUDIO_UNAVAILABLE,
                "response_time_ms": start_time.elapsed().as_millis(),
                "timestamp": chrono::Utc::now().to_rfc3339(),
                "proxy_version": crate::VERSION
            });
            if config.debug {
                println!(
                    "[DEBUG] Health Check Response (Error): {}",
                    serde_json::to_string_pretty(&response).unwrap_or_default()
                );
            }
            Ok(response)
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct LmStudioDownloadStatus {
    #[serde(default)]
    job_id: Option<String>,
    status: String,
    #[serde(default)]
    total_size_bytes: Option<u64>,
    #[serde(default)]
    downloaded_bytes: Option<u64>,
    #[serde(default)]
    bytes_per_second: Option<f64>,
    #[serde(default)]
    estimated_completion: Option<String>,
    #[serde(default)]
    started_at: Option<String>,
    #[serde(default)]
    completed_at: Option<String>,
    #[serde(default)]
    error: Option<String>,
}

impl LmStudioDownloadStatus {
    fn translated_status(&self) -> String {
        match self.status.as_str() {
            "completed" | "already_downloaded" => "success".to_string(),
            other => other.to_string(),
        }
    }

    fn is_terminal(&self) -> bool {
        matches!(
            self.status.as_str(),
            "completed" | "already_downloaded" | "failed"
        )
    }

    fn is_failure(&self) -> bool {
        matches!(self.status.as_str(), "failed")
    }

    fn job_id(&self) -> Result<&str, ProxyError> {
        self.job_id.as_deref().ok_or_else(|| {
            ProxyError::internal_server_error("LM Studio download response missing job identifier")
        })
    }

    fn to_chunk(&self, model: &str) -> Value {
        let mut chunk = serde_json::Map::new();
        chunk.insert(
            "status".to_string(),
            Value::String(self.translated_status()),
        );
        chunk.insert("model".to_string(), Value::String(model.to_string()));
        chunk.insert("detail".to_string(), Value::String(self.status.clone()));
        if let Some(job_id) = &self.job_id {
            chunk.insert("job_id".to_string(), Value::String(job_id.clone()));
        }
        if let Some(total) = self.total_size_bytes {
            chunk.insert("total".to_string(), Value::from(total));
        }
        if let Some(done) = self.downloaded_bytes {
            chunk.insert("completed".to_string(), Value::from(done));
        }
        if let Some(rate) = self.bytes_per_second {
            chunk.insert("bytes_per_second".to_string(), Value::from(rate));
        }
        if let Some(eta) = &self.estimated_completion {
            chunk.insert(
                "estimated_completion".to_string(),
                Value::String(eta.clone()),
            );
        }
        if let Some(started) = &self.started_at {
            chunk.insert("started_at".to_string(), Value::String(started.clone()));
        }
        if let Some(done_at) = &self.completed_at {
            chunk.insert("completed_at".to_string(), Value::String(done_at.clone()));
        }
        if let Some(err) = &self.error {
            chunk.insert("error".to_string(), Value::String(err.clone()));
        }
        Value::Object(chunk)
    }

    fn into_final_response(self, model: &str) -> Result<Value, ProxyError> {
        match self.status.as_str() {
            "completed" | "already_downloaded" => {
                let mut map = serde_json::Map::new();
                map.insert("status".to_string(), Value::String("success".to_string()));
                map.insert("model".to_string(), Value::String(model.to_string()));
                map.insert("detail".to_string(), Value::String(self.status));
                if let Some(job_id) = self.job_id {
                    map.insert("job_id".to_string(), Value::String(job_id));
                }
                if let Some(total) = self.total_size_bytes {
                    map.insert("total".to_string(), Value::from(total));
                }
                if let Some(done) = self.downloaded_bytes {
                    map.insert("completed".to_string(), Value::from(done));
                }
                if let Some(done_at) = self.completed_at {
                    map.insert("completed_at".to_string(), Value::String(done_at));
                }
                Ok(Value::Object(map))
            }
            "failed" => Err(ProxyError::internal_server_error(
                &self
                    .error
                    .clone()
                    .unwrap_or_else(|| "LM Studio reported download failure".to_string()),
            )),
            other => Err(ProxyError::internal_server_error(&format!(
                "Unexpected download status: {}",
                other
            ))),
        }
    }
}

async fn initiate_lmstudio_download(
    client: &reqwest::Client,
    base_url: &str,
    model_identifier: &str,
    quantization: Option<&str>,
    cancellation_token: CancellationToken,
) -> Result<LmStudioDownloadStatus, ProxyError> {
    let mut payload = serde_json::Map::new();
    payload.insert(
        "model".to_string(),
        Value::String(model_identifier.to_string()),
    );
    if let Some(q) = quantization {
        payload.insert("quantization".to_string(), Value::String(q.to_string()));
    }

    let url = format!("{}{}", base_url, LM_STUDIO_NATIVE_DOWNLOAD);
    log_request("POST", &url, Some(model_identifier));
    let response_value = send_json_request(
        client,
        Method::POST,
        &url,
        Some(&Value::Object(payload)),
        cancellation_token,
    )
    .await?;

    serde_json::from_value(response_value).map_err(|e| {
        ProxyError::internal_server_error(&format!("Invalid download response: {}", e))
    })
}

async fn wait_for_download_completion(
    client: &reqwest::Client,
    base_url: &str,
    mut status: LmStudioDownloadStatus,
    cancellation_token: CancellationToken,
) -> Result<LmStudioDownloadStatus, ProxyError> {
    while !status.is_terminal() {
        let job_id = status.job_id()?.to_string();
        tokio::select! {
            _ = cancellation_token.cancelled() => {
                return Err(ProxyError::request_cancelled());
            }
            _ = sleep(Duration::from_millis(DOWNLOAD_STATUS_POLL_INTERVAL_MS)) => {}
        }
        status = fetch_lmstudio_download_status_with_client(
            client,
            base_url,
            &job_id,
            cancellation_token.clone(),
        )
        .await?;
    }
    Ok(status)
}

async fn stream_download_status_updates(
    client: reqwest::Client,
    base_url: String,
    mut status: LmStudioDownloadStatus,
    model_name: String,
    cancellation_token: CancellationToken,
    tx: mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
) -> Result<(), ProxyError> {
    loop {
        if !send_status_chunk(&tx, &status.to_chunk(&model_name)) {
            return Ok(());
        }

        if status.is_failure() {
            return Err(ProxyError::internal_server_error(
                &status
                    .error
                    .clone()
                    .unwrap_or_else(|| "LM Studio download failed".to_string()),
            ));
        }

        if status.is_terminal() {
            return Ok(());
        }

        let job_id = status.job_id()?.to_string();
        tokio::select! {
            _ = cancellation_token.cancelled() => {
                return Err(ProxyError::request_cancelled());
            }
            _ = sleep(Duration::from_millis(DOWNLOAD_STATUS_POLL_INTERVAL_MS)) => {}
        }

        status = fetch_lmstudio_download_status_with_client(
            &client,
            &base_url,
            &job_id,
            cancellation_token.clone(),
        )
        .await?;
    }
}

async fn fetch_lmstudio_download_status_with_client(
    client: &reqwest::Client,
    base_url: &str,
    job_id: &str,
    cancellation_token: CancellationToken,
) -> Result<LmStudioDownloadStatus, ProxyError> {
    let url = format!(
        "{}{}/{}",
        base_url, LM_STUDIO_NATIVE_DOWNLOAD_STATUS, job_id
    );
    let response_value =
        send_json_request(client, Method::GET, &url, None, cancellation_token).await?;

    serde_json::from_value(response_value).map_err(|e| {
        ProxyError::internal_server_error(&format!("Invalid download status payload: {}", e))
    })
}

async fn send_json_request(
    client: &reqwest::Client,
    method: Method,
    url: &str,
    body: Option<&Value>,
    cancellation_token: CancellationToken,
) -> Result<Value, ProxyError> {
    let mut builder = client.request(method, url);
    if let Some(payload) = body {
        builder = builder.json(payload);
    }

    tokio::select! {
        response = builder.send() => {
            match response {
                Ok(resp) => handle_json_response(resp, cancellation_token).await,
                Err(err) => {
                    if err.is_connect() {
                        Err(ProxyError::lm_studio_unavailable(ERROR_LM_STUDIO_UNAVAILABLE))
                    } else if err.is_timeout() {
                        Err(ProxyError::lm_studio_unavailable(ERROR_TIMEOUT))
                    } else {
                        Err(ProxyError::internal_server_error(&format!(
                            "LM Studio request failed: {}",
                            err
                        )))
                    }
                }
            }
        }
        _ = cancellation_token.cancelled() => Err(ProxyError::request_cancelled()),
    }
}

fn send_status_chunk(
    tx: &mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
    chunk: &Value,
) -> bool {
    match serde_json::to_string(chunk) {
        Ok(serialized) => tx
            .send(Ok(Bytes::from(format!("{}\n", serialized))))
            .is_ok(),
        Err(e) => {
            log_warning("Pull chunk", &format!("Serialization failed: {}", e));
            false
        }
    }
}

fn send_status_error_chunk(
    tx: &mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
    model: &str,
    message: &str,
) {
    let chunk = json!({
        "status": "failed",
        "model": model,
        "error": message
    });
    let _ = send_status_chunk(tx, &chunk);
}

fn finalize_embedding_response(mut response: Value, mode: EmbeddingResponseMode) -> Value {
    if matches!(mode, EmbeddingResponseMode::LegacyEmbeddings) {
        let fallback = Value::Array(Vec::new());
        let first_vector = response
            .get("embeddings")
            .and_then(|value| value.as_array())
            .and_then(|arr| arr.first().cloned())
            .unwrap_or(fallback);

        if let Some(obj) = response.as_object_mut() {
            obj.insert("embedding".to_string(), first_vector);
            obj.remove("embeddings");
        }
    }

    response
}

fn normalize_chat_messages(messages: &[Value], system_prompt: Option<&str>) -> Value {
    if let Some(system_text) = system_prompt {
        let already_has_system = messages.iter().any(|message| {
            message
                .get("role")
                .and_then(|role| role.as_str())
                .map(|role| role.eq_ignore_ascii_case("system"))
                .unwrap_or(false)
        });

        if already_has_system {
            json!(messages)
        } else {
            let mut combined = Vec::with_capacity(messages.len() + 1);
            combined.push(json!({
                "role": "system",
                "content": system_text,
            }));
            combined.extend(messages.iter().cloned());
            Value::Array(combined)
        }
    } else {
        json!(messages)
    }
}

async fn resolve_model_target<'a>(
    context: &RequestContext<'a>,
    model_resolver: &ModelResolverType,
    requested_model: &str,
    cancellation_token: CancellationToken,
) -> Result<(String, Option<VirtualModelEntry>), ProxyError> {
    if let Some(entry) = context.virtual_models.get(requested_model).await {
        return Ok((entry.target_model_id.clone(), Some(entry)));
    }

    match model_resolver {
        ModelResolverType::Native(resolver) => resolver
            .resolve_model_name(requested_model, context.client, cancellation_token)
            .await
            .map(|id| (id, None)),
    }
}

async fn determine_download_identifier(
    context: &RequestContext<'_>,
    model_resolver: &ModelResolverType,
    requested_model: &str,
    source_override: Option<&str>,
    resolved_model: Option<(String, Option<VirtualModelEntry>)>,
    cancellation_token: CancellationToken,
) -> Result<String, ProxyError> {
    if let Some(source) = source_override {
        return Ok(source.to_string());
    }

    if looks_like_remote_identifier(requested_model) {
        return Ok(requested_model.to_string());
    }

    if let Some((resolved_model_id, virtual_entry)) = resolved_model {
        if let Some(source) = virtual_entry
            .as_ref()
            .and_then(extract_virtual_download_source)
        {
            return Ok(source);
        }

        if looks_like_remote_identifier(&resolved_model_id) {
            return Ok(resolved_model_id);
        }

        if resolved_model_id.contains('/') && !resolved_model_id.contains(' ') {
            return Ok(resolved_model_id);
        }

        if let Some(model_info) = fetch_model_info_for_id(
            context,
            model_resolver,
            &resolved_model_id,
            cancellation_token,
        )
        .await?
        {
            let cleaned_id = clean_model_name(&model_info.id).to_string();
            if publisher_prefers_hf_link(&model_info.publisher) {
                return Ok(build_hf_download_url(&model_info.publisher, &cleaned_id));
            }

            if let Some(identifier) = build_catalog_identifier(&model_info.publisher, &cleaned_id) {
                return Ok(identifier);
            }
        }

        return Ok(resolved_model_id);
    }

    Ok(requested_model.to_string())
}

fn looks_like_remote_identifier(identifier: &str) -> bool {
    let lowered = identifier.to_ascii_lowercase();
    lowered.starts_with("http://")
        || lowered.starts_with("https://")
        || lowered.starts_with("hf://")
        || lowered.starts_with("s3://")
        || lowered.starts_with("gs://")
}

fn extract_virtual_download_source(entry: &VirtualModelEntry) -> Option<String> {
    entry
        .metadata
        .parameters
        .as_ref()
        .and_then(|params| params.get("download_source"))
        .and_then(|value| value.as_str())
        .map(|value| value.to_string())
}

async fn fetch_model_info_for_id(
    context: &RequestContext<'_>,
    model_resolver: &ModelResolverType,
    target_model_id: &str,
    cancellation_token: CancellationToken,
) -> Result<Option<ModelInfo>, ProxyError> {
    match model_resolver {
        ModelResolverType::Native(resolver) => {
            let models = resolver
                .get_all_models(context.client, cancellation_token)
                .await?;
            Ok(models.into_iter().find(|model| model.id == target_model_id))
        }
    }
}

fn publisher_prefers_hf_link(publisher: &str) -> bool {
    matches!(
        publisher.to_ascii_lowercase().as_str(),
        "lmstudio-community" | "huggingface"
    )
}

fn build_hf_download_url(publisher: &str, model_id: &str) -> String {
    format!(
        "https://huggingface.co/{}/{}",
        publisher.trim().trim_end_matches('/'),
        model_id.trim_start_matches('/')
    )
}

fn build_catalog_identifier(publisher: &str, model_id: &str) -> Option<String> {
    let trimmed = publisher.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(format!(
        "{}/{}",
        trimmed.trim_end_matches('/'),
        model_id.trim_start_matches('/')
    ))
}

fn merge_option_maps(base: Option<&Value>, overrides: Option<&Value>) -> Option<Value> {
    match (base, overrides) {
        (None, None) => None,
        (Some(b), None) => Some(b.clone()),
        (None, Some(o)) => Some(o.clone()),
        (Some(b), Some(o)) => match (b.as_object(), o.as_object()) {
            (Some(base_obj), Some(override_obj)) => {
                let mut combined = serde_json::Map::new();
                for (k, v) in base_obj {
                    combined.insert(k.clone(), v.clone());
                }
                for (k, v) in override_obj {
                    combined.insert(k.clone(), v.clone());
                }
                Some(Value::Object(combined))
            }
            _ => Some(o.clone()),
        },
    }
}

fn parse_keep_alive_seconds(raw_value: Option<&Value>) -> Result<Option<i64>, ProxyError> {
    let Some(value) = raw_value else {
        return Ok(None);
    };

    match value {
        Value::Null => Ok(None),
        Value::Number(num) => {
            if let Some(signed) = num.as_i64() {
                Ok(Some(signed))
            } else if let Some(unsigned) = num.as_u64() {
                if unsigned <= i64::MAX as u64 {
                    Ok(Some(unsigned as i64))
                } else {
                    Err(ProxyError::bad_request(
                        "keep_alive value exceeds supported range",
                    ))
                }
            } else {
                Err(ProxyError::bad_request("keep_alive must be integral"))
            }
        }
        Value::String(text) => {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                return Ok(None);
            }

            if let Ok(duration) = parse_duration(trimmed) {
                if duration.as_secs() <= i64::MAX as u64 {
                    Ok(Some(duration.as_secs() as i64))
                } else {
                    Err(ProxyError::bad_request(
                        "keep_alive duration exceeds supported range",
                    ))
                }
            } else {
                trimmed.parse::<i64>().map(Some).map_err(|_| {
                    ProxyError::bad_request(
                        "Invalid keep_alive value. Use numeric seconds or durations like '5m'",
                    )
                })
            }
        }
        _ => Err(ProxyError::bad_request(
            "Invalid keep_alive value. Use numeric seconds or durations like '5m'",
        )),
    }
}

fn apply_keep_alive_ttl(target: &mut Value, keep_alive_seconds: Option<i64>) {
    if let Some(ttl) = keep_alive_seconds
        && let Some(obj) = target.as_object_mut()
    {
        obj.insert("ttl".to_string(), Value::from(ttl));
    }
}

fn keep_alive_requests_unload(ttl: Option<i64>) -> bool {
    matches!(ttl, Some(value) if value == 0)
}

fn build_virtual_metadata(
    body: &Value,
    base: Option<VirtualModelMetadata>,
) -> VirtualModelMetadata {
    let mut metadata = base.unwrap_or_default();

    if let Some(system_prompt) = body.get("system").and_then(|v| v.as_str()) {
        metadata.system_prompt = Some(system_prompt.to_string());
    }

    if let Some(template) = body.get("template").and_then(|v| v.as_str()) {
        metadata.template = Some(template.to_string());
    }

    if let Some(parameters) = body.get("parameters") {
        metadata.parameters = Some(parameters.clone());
    }

    if let Some(license) = body.get("license") {
        metadata.license = Some(license.clone());
    }

    if let Some(adapters) = body.get("adapters") {
        metadata.adapters = Some(adapters.clone());
    }

    if let Some(messages) = body.get("messages").and_then(|m| m.as_array()).cloned() {
        metadata.messages = Some(messages);
    }

    metadata
}

fn build_virtual_model_response(entry: &VirtualModelEntry) -> Value {
    json!({
        "status": "success",
        "model": entry.name,
        "virtual": true,
        "source_model": entry.source_model,
        "target_model_id": entry.target_model_id,
        "created_at": entry.created_at.to_rfc3339(),
        "updated_at": entry.updated_at.to_rfc3339(),
    })
}

fn stream_status_messages(
    chunks: Vec<Value>,
    error_label: &str,
) -> Result<warp::reply::Response, ProxyError> {
    let (tx, rx) = mpsc::unbounded_channel();
    for chunk in chunks {
        if !send_status_chunk(&tx, &chunk) {
            break;
        }
    }
    drop(tx);
    create_ndjson_stream_response(rx, error_label)
}

async fn create_virtual_model_alias(
    context: &RequestContext<'_>,
    model_resolver: &ModelResolverType,
    alias_name: &str,
    source_name: &str,
    body: &Value,
    cancellation_token: CancellationToken,
) -> Result<VirtualModelEntry, ProxyError> {
    if let Some(files) = body.get("files") {
        let has_content = match files {
            Value::Object(map) => !map.is_empty(),
            Value::Array(arr) => !arr.is_empty(),
            Value::Null => false,
            _ => true,
        };
        if has_content {
            return Err(ProxyError::not_implemented(
                "Creating models from custom blobs is not supported via LM Studio proxy",
            ));
        }
    }

    if body.get("quantize").is_some() {
        return Err(ProxyError::not_implemented(
            "Quantization is not available via LM Studio proxy",
        ));
    }

    let (resolved_id, source_virtual_entry) =
        resolve_model_target(context, model_resolver, source_name, cancellation_token).await?;

    let base_metadata = source_virtual_entry.map(|entry| entry.metadata);
    let metadata = build_virtual_metadata(body, base_metadata);

    context
        .virtual_models
        .create_alias(alias_name, source_name.to_string(), resolved_id, metadata)
        .await
}

#[cfg(test)]
mod pull_tests {
    use super::LmStudioDownloadStatus;

    #[test]
    fn final_response_populates_success_fields() {
        let status = LmStudioDownloadStatus {
            job_id: Some("job_123".to_string()),
            status: "completed".to_string(),
            total_size_bytes: Some(1024),
            downloaded_bytes: Some(1024),
            bytes_per_second: Some(2048.0),
            estimated_completion: None,
            started_at: Some("2025-01-01T00:00:00Z".to_string()),
            completed_at: Some("2025-01-01T00:00:10Z".to_string()),
            error: None,
        };

        let response = status
            .into_final_response("mistral:latest")
            .expect("should succeed");

        assert_eq!(response["status"], "success");
        assert_eq!(response["model"], "mistral:latest");
        assert_eq!(response["detail"], "completed");
        assert_eq!(response["job_id"], "job_123");
        assert_eq!(response["total"], 1024);
        assert_eq!(response["completed"], 1024);
    }

    #[test]
    fn final_response_errors_on_failure() {
        let status = LmStudioDownloadStatus {
            job_id: None,
            status: "failed".to_string(),
            total_size_bytes: None,
            downloaded_bytes: None,
            bytes_per_second: None,
            estimated_completion: None,
            started_at: None,
            completed_at: None,
            error: Some("network".to_string()),
        };

        let err = status.into_final_response("model").unwrap_err();
        assert!(err.message.contains("network"));
    }
}

#[cfg(test)]
mod embeddings_response_tests {
    use serde_json::json;

    use super::{EmbeddingResponseMode, finalize_embedding_response};

    #[test]
    fn legacy_mode_promotes_single_embedding() {
        let payload = json!({
            "model": "all-minilm",
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        });

        let transformed =
            finalize_embedding_response(payload, EmbeddingResponseMode::LegacyEmbeddings);

        assert!(transformed.get("embeddings").is_none());
        assert_eq!(
            transformed
                .get("embedding")
                .and_then(|value| value.as_array())
                .map(|values| values.len())
                .unwrap_or_default(),
            3
        );
    }

    #[test]
    fn modern_mode_keeps_embeddings_array() {
        let payload = json!({
            "model": "all-minilm",
            "embeddings": [[0.1, 0.2, 0.3]],
        });

        let transformed =
            finalize_embedding_response(payload.clone(), EmbeddingResponseMode::Embed);

        assert!(transformed.get("embedding").is_none());
        assert_eq!(transformed.get("embeddings"), payload.get("embeddings"));
    }
}

#[cfg(test)]
mod keep_alive_tests {
    use serde_json::json;

    use super::{apply_keep_alive_ttl, keep_alive_requests_unload, parse_keep_alive_seconds};

    #[test]
    fn parses_duration_strings() {
        let parsed = parse_keep_alive_seconds(Some(&json!("5m"))).unwrap();
        assert_eq!(parsed, Some(300));

        let indefinite = parse_keep_alive_seconds(Some(&json!(-1))).unwrap();
        assert_eq!(indefinite, Some(-1));
    }

    #[test]
    fn detects_unload_requests() {
        assert!(keep_alive_requests_unload(Some(0)));
        assert!(!keep_alive_requests_unload(Some(10)));
    }

    #[test]
    fn applies_ttl_field() {
        let mut payload = json!({ "model": "demo" });
        apply_keep_alive_ttl(&mut payload, Some(30));
        assert_eq!(payload["ttl"], 30);
    }

    #[test]
    fn rejects_invalid_keep_alive_string() {
        let err = parse_keep_alive_seconds(Some(&json!("abc"))).unwrap_err();
        assert!(err.message.contains("Invalid keep_alive value"));
    }
}
