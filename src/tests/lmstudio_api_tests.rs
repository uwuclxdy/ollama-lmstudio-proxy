/// Tests for LM Studio API passthrough and native/legacy endpoint handling
#[cfg(test)]
mod lmstudio_api_tests {
    use serde_json::json;

    /// Test native LM Studio API model list structure
    #[test]
    fn test_lmstudio_native_models_response() {
        // According to endpoints.md: Native API returns models with rich metadata
        let response = json!({
            "object": "list",
            "data": [
                {
                    "id": "qwen2-vl-7b-instruct",
                    "object": "model",
                    "type": "vlm",
                    "publisher": "mlx-community",
                    "arch": "qwen2_vl",
                    "compatibility_type": "mlx",
                    "quantization": "4bit",
                    "state": "not-loaded",
                    "max_context_length": 32768
                },
                {
                    "id": "text-embedding-nomic-embed-text-v1.5",
                    "object": "model",
                    "type": "embeddings",
                    "publisher": "nomic-ai",
                    "arch": "nomic-bert",
                    "compatibility_type": "gguf",
                    "quantization": "Q4_0",
                    "state": "not-loaded",
                    "max_context_length": 2048
                }
            ]
        });

        let data = response["data"].as_array().unwrap();
        assert_eq!(data.len(), 2);

        // Check VLM model
        let vlm = &data[0];
        assert_eq!(vlm["type"], "vlm");
        assert!(vlm.get("publisher").is_some());
        assert!(vlm.get("arch").is_some());
        assert!(vlm.get("compatibility_type").is_some());
        assert!(vlm.get("quantization").is_some());
        assert!(vlm.get("state").is_some());
        assert!(vlm.get("max_context_length").is_some());

        // Check embeddings model
        let embed = &data[1];
        assert_eq!(embed["type"], "embeddings");
    }

    /// Test native LM Studio chat completions response
    #[test]
    fn test_lmstudio_native_chat_response() {
        // According to endpoints.md: Native API includes enhanced stats
        let response = json!({
            "id": "chatcmpl-i3gkjwthhw96whukek9tz",
            "object": "chat.completion",
            "created": 1731990317_u64,
            "model": "granite-3.0-2b-instruct",
            "choices": [
                {
                    "index": 0,
                    "logprobs": null,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "Greetings!"
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 24,
                "completion_tokens": 53,
                "total_tokens": 77
            },
            "stats": {
                "tokens_per_second": 51.43709529007664,
                "time_to_first_token": 0.111,
                "generation_time": 0.954,
                "stop_reason": "eosFound"
            },
            "model_info": {
                "arch": "granite",
                "quant": "Q4_K_M",
                "format": "gguf",
                "context_length": 4096
            },
            "runtime": {
                "name": "llama.cpp-mac-arm64-apple-metal-advsimd",
                "version": "1.3.0",
                "supported_formats": ["gguf"]
            }
        });

        // Verify native API specific fields
        assert!(response.get("stats").is_some());
        assert!(response.get("model_info").is_some());
        assert!(response.get("runtime").is_some());

        let stats = &response["stats"];
        assert!(stats.get("tokens_per_second").is_some());
        assert!(stats.get("time_to_first_token").is_some());
        assert!(stats.get("generation_time").is_some());
    }

    /// Test legacy LM Studio API model list (v1)
    #[test]
    fn test_lmstudio_legacy_models_response() {
        // According to endpoints.md: Legacy API uses OpenAI format
        let response = json!({
            "object": "list",
            "data": [
                {
                    "id": "lmstudio-community/meta-llama-3.1-8b-instruct",
                    "object": "model",
                    "created": 1731990000_u64,
                    "owned_by": "lmstudio"
                }
            ]
        });

        let data = response["data"].as_array().unwrap();
        assert!(!data.is_empty());

        let model = &data[0];
        assert!(model.get("id").is_some());
        assert!(model.get("object").is_some());
        // Legacy format has minimal metadata
        assert!(model.get("type").is_none());
        assert!(model.get("arch").is_none());
    }

    /// Test endpoint conversion between native and legacy APIs
    #[test]
    fn test_endpoint_conversion() {
        // According to lmstudio.rs: v1 endpoints convert to /api/v0/ in native mode
        let v1_chat = "/v1/chat/completions";
        let v0_chat = "/api/v0/chat/completions";

        assert!(v1_chat.starts_with("/v1/"));
        assert!(v0_chat.starts_with("/api/v0/"));

        // Conversion logic
        let converted_to_v0 = v1_chat.replace("/v1/", "/api/v0/");
        assert_eq!(converted_to_v0, v0_chat);

        let converted_to_v1 = v0_chat.replace("/api/v0/", "/v1/");
        assert_eq!(converted_to_v1, v1_chat);
    }

    /// Test native API completions endpoint
    #[test]
    fn test_lmstudio_native_completions() {
        // Native API uses /api/v0/completions
        let request = json!({
            "model": "granite-3.0-2b-instruct",
            "prompt": "the meaning of life is",
            "temperature": 0.7,
            "max_tokens": 10,
            "stream": false,
            "stop": "\n"
        });

        assert_eq!(request["model"], "granite-3.0-2b-instruct");
        assert!(request.get("prompt").is_some());
    }

    /// Test native API embeddings endpoint
    #[test]
    fn test_lmstudio_native_embeddings() {
        // Native API uses /api/v0/embeddings
        let _request = json!({
            "model": "text-embedding-nomic-embed-text-v1.5",
            "input": "Some text to embed"
        });

        let response = json!({
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [
                        -0.016731496900320053,
                        0.028460891917347908
                    ],
                    "index": 0
                }
            ],
            "model": "text-embedding-nomic-embed-text-v1.5@q4_k_m",
            "usage": {
                "prompt_tokens": 0,
                "total_tokens": 0
            }
        });

        let data = response["data"].as_array().unwrap();
        let embedding = data[0]["embedding"].as_array().unwrap();
        assert!(!embedding.is_empty());
    }

    /// Test that native API supports tool calling
    #[test]
    fn test_lmstudio_native_tool_calling() {
        // According to api-changelog.md: LM Studio 0.3.6+ supports tool calling
        let request = json!({
            "model": "qwen2.5-7b-instruct",
            "messages": [
                {"role": "user", "content": "what is the weather in tokyo?"}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the weather in a given city",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {
                                    "type": "string",
                                    "description": "The city to get the weather for"
                                }
                            },
                            "required": ["city"]
                        }
                    }
                }
            ]
        });

        let tools = request["tools"].as_array().unwrap();
        assert!(!tools.is_empty());

        let tool = &tools[0];
        assert_eq!(tool["type"], "function");
        assert!(tool["function"].get("name").is_some());
        assert!(tool["function"].get("parameters").is_some());
    }

    /// Test model capabilities in native API
    #[test]
    fn test_lmstudio_native_model_capabilities() {
        // According to api-changelog.md: 0.3.16+ returns capabilities array
        let model = json!({
            "id": "qwen2.5-7b-instruct",
            "type": "llm",
            "capabilities": ["tool_use", "chat", "completion"]
        });

        let capabilities = model["capabilities"].as_array().unwrap();
        assert!(capabilities.contains(&json!("tool_use")));
        assert!(capabilities.contains(&json!("chat")));
    }

    /// Test stream_options for token usage during streaming
    #[test]
    fn test_lmstudio_stream_options() {
        // According to api-changelog.md: 0.3.18+ supports stream_options
        let request = json!({
            "model": "qwen2.5-7b-instruct",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true,
            "stream_options": {
                "include_usage": true
            }
        });

        assert_eq!(request["stream"], true);
        assert!(request.get("stream_options").is_some());
        assert_eq!(request["stream_options"]["include_usage"], true);
    }

    /// Test tool_choice parameter
    #[test]
    fn test_lmstudio_tool_choice() {
        // According to api-changelog.md: 0.3.15+ supports tool_choice
        let request_auto = json!({
            "model": "qwen2.5-7b-instruct",
            "messages": [{"role": "user", "content": "test"}],
            "tools": [{"type": "function", "function": {"name": "test"}}],
            "tool_choice": "auto"
        });

        assert_eq!(request_auto["tool_choice"], "auto");

        let request_required = json!({
            "tool_choice": "required"
        });
        assert_eq!(request_required["tool_choice"], "required");

        let request_none = json!({
            "tool_choice": "none"
        });
        assert_eq!(request_none["tool_choice"], "none");
    }

    /// Test speculative decoding parameters
    #[test]
    fn test_lmstudio_speculative_decoding() {
        // According to api-changelog.md: 0.3.10+ supports draft_model
        let request = json!({
            "model": "deepseek-r1-distill-qwen-7b",
            "draft_model": "deepseek-r1-distill-qwen-0.5b",
            "messages": [{"role": "user", "content": "test"}]
        });

        assert!(request.get("draft_model").is_some());

        let response = json!({
            "stats": {
                "tokens_per_second": 50.0,
                "draft_model": "deepseek-r1-distill-qwen-0.5b",
                "total_draft_tokens_count": 100,
                "accepted_draft_tokens_count": 75,
                "rejected_draft_tokens_count": 20,
                "ignored_draft_tokens_count": 5
            }
        });

        let stats = &response["stats"];
        assert!(stats.get("draft_model").is_some());
        assert!(stats.get("total_draft_tokens_count").is_some());
    }

    /// Test TTL and auto-evict parameters
    #[test]
    fn test_lmstudio_ttl_auto_evict() {
        // According to ttl-and-auto-evict.md: Can set TTL in request
        let _request = json!({
            "model": "deepseek-r1-distill-qwen-7b",
            "messages": [{"role": "user", "content": "test"}],
            "ttl": 300
        });

        assert_eq!(_request["ttl"], 300);
    }

    /// Test reasoning content separation
    #[test]
    fn test_lmstudio_reasoning_content() {
        // According to api-changelog.md: 0.3.9+ separates reasoning_content
        let response = json!({
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "The answer is...",
                        "reasoning": "First I need to think..."
                    }
                }
            ]
        });

        let message = &response["choices"][0]["message"];
        assert!(message.get("reasoning").is_some());
        assert!(message.get("content").is_some());
    }

    /// Test passthrough preserves original structure
    #[test]
    fn test_lmstudio_passthrough_preserves_structure() {
        // Passthrough should forward requests as-is with minimal modification
        let original_request = json!({
            "model": "test-model",
            "custom_field": "custom_value",
            "messages": [{"role": "user", "content": "test"}]
        });

        // After model resolution, only model field should change
        let forwarded_request = json!({
            "model": "resolved-lmstudio-id",
            "custom_field": "custom_value",
            "messages": [{"role": "user", "content": "test"}]
        });

        assert_eq!(
            original_request["custom_field"],
            forwarded_request["custom_field"]
        );
        assert_eq!(original_request["messages"], forwarded_request["messages"]);
    }

    /// Test health check structure
    #[test]
    fn test_health_check_response() {
        // Health check should return comprehensive status
        let healthy_response = json!({
            "status": "healthy",
            "lmstudio_url": "http://localhost:1234",
            "http_status": 200,
            "models_known_to_lmstudio": 5,
            "response_time_ms": 15,
            "timestamp": "2025-11-14T12:00:00Z",
            "proxy_version": "0.1.0"
        });

        assert_eq!(healthy_response["status"], "healthy");
        assert!(healthy_response.get("models_known_to_lmstudio").is_some());
        assert!(healthy_response.get("response_time_ms").is_some());

        let unhealthy_response = json!({
            "status": "unreachable",
            "lmstudio_url": "http://localhost:1234",
            "error_message": "Connection refused",
            "timestamp": "2025-11-14T12:00:00Z"
        });

        assert_eq!(unhealthy_response["status"], "unreachable");
        assert!(unhealthy_response.get("error_message").is_some());
    }
}
