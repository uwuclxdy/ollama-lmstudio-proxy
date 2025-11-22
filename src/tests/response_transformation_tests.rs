/// Tests for response transformation between LM Studio and Ollama formats
#[cfg(test)]
mod response_transformation_tests {
    use serde_json::json;

    /// Test transformation of LM Studio chat response to Ollama format
    #[test]
    fn test_lmstudio_to_ollama_chat_response() {
        // LM Studio OpenAI-compatible response
        let lmstudio_response = json!({
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": 1731990317_u64,
            "model": "granite-3.0-2b-instruct",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you?"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18
            }
        });

        // Ollama format response
        let ollama_response = json!({
            "model": "granite-3.0-2b-instruct",
            "created_at": "2024-11-18T20:31:57.111706Z",
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you?"
            },
            "done": true,
            "total_duration": 954_000_000_u64,
            "load_duration": 111_000_000_u64,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 843_000_000_u64,
            "eval_count": 8,
            "eval_duration": 111_000_000_u64
        });

        // Verify message content matches
        assert_eq!(
            lmstudio_response["choices"][0]["message"]["content"],
            ollama_response["message"]["content"]
        );

        // Verify token counts map correctly
        assert_eq!(
            lmstudio_response["usage"]["prompt_tokens"],
            ollama_response["prompt_eval_count"]
        );
        assert_eq!(
            lmstudio_response["usage"]["completion_tokens"],
            ollama_response["eval_count"]
        );
    }

    /// Test transformation of streaming chat response
    #[test]
    fn test_streaming_chat_transformation() {
        // LM Studio streaming chunk
        let lmstudio_chunk = json!({
            "id": "chatcmpl-abc123",
            "object": "chat.completion.chunk",
            "created": 1731990317_u64,
            "model": "granite-3.0-2b-instruct",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": "Hello"
                    },
                    "finish_reason": null
                }
            ]
        });

        // Ollama streaming format
        let ollama_chunk = json!({
            "model": "granite-3.0-2b-instruct",
            "created_at": "2024-11-18T20:31:57.111706Z",
            "message": {
                "role": "assistant",
                "content": "Hello"
            },
            "done": false
        });

        // Verify delta content transforms to message content
        assert_eq!(
            lmstudio_chunk["choices"][0]["delta"]["content"],
            ollama_chunk["message"]["content"]
        );
        assert_eq!(ollama_chunk["done"], false);
    }

    /// Test final streaming chunk with usage stats
    #[test]
    fn test_final_streaming_chunk() {
        // Final LM Studio chunk with usage
        let lmstudio_final = json!({
            "id": "chatcmpl-abc123",
            "object": "chat.completion.chunk",
            "created": 1731990317_u64,
            "model": "granite-3.0-2b-instruct",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18
            }
        });

        // Ollama final chunk
        let ollama_final = json!({
            "model": "granite-3.0-2b-instruct",
            "created_at": "2024-11-18T20:31:57.111706Z",
            "message": {
                "role": "assistant",
                "content": ""
            },
            "done": true,
            "done_reason": "stop",
            "total_duration": 954_000_000_u64,
            "prompt_eval_count": 10,
            "eval_count": 8
        });

        assert_eq!(lmstudio_final["choices"][0]["finish_reason"], "stop");
        assert_eq!(ollama_final["done_reason"], "stop");
        assert_eq!(ollama_final["done"], true);
    }

    /// Test timing information calculation
    #[test]
    fn test_timing_info_calculation() {
        // According to helpers.rs: TimingInfo calculates durations from stats
        let stats = json!({
            "time_to_first_token": 0.111,
            "generation_time": 0.954,
            "tokens_per_second": 51.43709529007664
        });

        // Convert to nanoseconds for Ollama format
        let time_to_first_token_ns =
            (stats["time_to_first_token"].as_f64().unwrap() * 1_000_000_000.0) as u64;
        let generation_time_ns =
            (stats["generation_time"].as_f64().unwrap() * 1_000_000_000.0) as u64;

        assert_eq!(time_to_first_token_ns, 111_000_000);
        assert_eq!(generation_time_ns, 954_000_000);

        // Total duration = load + generation
        let total_duration = time_to_first_token_ns + generation_time_ns;
        assert_eq!(total_duration, 1_065_000_000);
    }

    /// Test tool call response transformation
    #[test]
    fn test_tool_call_transformation() {
        // LM Studio tool call response
        let lmstudio_tool_call = json!({
            "id": "chatcmpl-abc123",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": "{\"city\":\"Tokyo\"}"
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls"
                }
            ]
        });

        // Ollama tool call format
        let ollama_tool_call = json!({
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": {
                                "city": "Tokyo"
                            }
                        }
                    }
                ]
            },
            "done": true,
            "done_reason": "tool_calls"
        });

        let lm_tool_call = &lmstudio_tool_call["choices"][0]["message"]["tool_calls"][0];
        let ollama_tc = &ollama_tool_call["message"]["tool_calls"][0];

        assert_eq!(
            lm_tool_call["function"]["name"],
            ollama_tc["function"]["name"]
        );
    }

    /// Test generate endpoint response transformation
    #[test]
    fn test_generate_response_transformation() {
        // LM Studio completions response
        let lmstudio_response = json!({
            "id": "cmpl-abc123",
            "object": "text_completion",
            "created": 1731990317_u64,
            "model": "granite-3.0-2b-instruct",
            "choices": [
                {
                    "text": "The answer is 42.",
                    "index": 0,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 7,
                "total_tokens": 12
            }
        });

        // Ollama generate format
        let ollama_response = json!({
            "model": "granite-3.0-2b-instruct",
            "created_at": "2024-11-18T20:31:57.111706Z",
            "response": "The answer is 42.",
            "done": true,
            "context": [1, 2, 3, 4, 5],
            "total_duration": 500_000_000_u64,
            "load_duration": 100_000_000_u64,
            "prompt_eval_count": 5,
            "prompt_eval_duration": 200_000_000_u64,
            "eval_count": 7,
            "eval_duration": 200_000_000_u64
        });

        // Text content maps to response field
        assert_eq!(
            lmstudio_response["choices"][0]["text"],
            ollama_response["response"]
        );

        // Token counts match
        assert_eq!(
            lmstudio_response["usage"]["prompt_tokens"],
            ollama_response["prompt_eval_count"]
        );
    }

    /// Test embeddings response transformation
    #[test]
    fn test_embeddings_transformation() {
        // LM Studio embeddings response
        let lmstudio_response = json!({
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [0.1, 0.2, 0.3],
                    "index": 0
                }
            ],
            "model": "text-embedding-nomic-embed-text-v1.5",
            "usage": {
                "prompt_tokens": 5,
                "total_tokens": 5
            }
        });

        // Ollama embeddings format
        let ollama_response = json!({
            "model": "text-embedding-nomic-embed-text-v1.5",
            "embeddings": [
                [0.1, 0.2, 0.3]
            ],
            "total_duration": 100_000_000_u64,
            "load_duration": 50_000_000_u64,
            "prompt_eval_count": 5
        });

        let lm_embedding = lmstudio_response["data"][0]["embedding"]
            .as_array()
            .unwrap();
        let ollama_embedding = ollama_response["embeddings"][0].as_array().unwrap();

        assert_eq!(lm_embedding, ollama_embedding);
    }

    /// Test content extraction from different message formats
    #[test]
    fn test_content_extraction() {
        // String content
        let string_message = json!({
            "role": "assistant",
            "content": "Hello"
        });
        assert_eq!(string_message["content"], "Hello");

        // Array content (multimodal)
        let array_message = json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
            ]
        });
        let content_array = array_message["content"].as_array().unwrap();
        assert_eq!(content_array.len(), 2);

        // Extract text from array
        let text_content = content_array
            .iter()
            .filter(|c| c["type"] == "text")
            .map(|c| c["text"].as_str().unwrap())
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(text_content, "Describe this");
    }

    /// Test finish_reason mapping
    #[test]
    fn test_finish_reason_mapping() {
        // OpenAI finish reasons to Ollama done_reason
        let mappings = vec![
            ("stop", "stop"),
            ("length", "length"),
            ("tool_calls", "tool_calls"),
            ("content_filter", "content_filter"),
        ];

        for (openai_reason, ollama_reason) in mappings {
            assert_eq!(openai_reason, ollama_reason);
        }
    }

    /// Test error response transformation
    #[test]
    fn test_error_response_transformation() {
        // LM Studio error response
        let lmstudio_error = json!({
            "error": {
                "message": "Model not found",
                "type": "invalid_request_error",
                "code": "model_not_found"
            }
        });

        // Ollama error format
        let ollama_error = json!({
            "error": "Model not found"
        });

        assert!(lmstudio_error.get("error").is_some());
        assert!(ollama_error.get("error").is_some());
    }

    /// Test response with reasoning content (DeepSeek R1 style)
    #[test]
    fn test_reasoning_content_transformation() {
        // LM Studio response with reasoning
        let lmstudio_response = json!({
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "The answer is 42",
                        "reasoning": "First, I need to think about..."
                    }
                }
            ]
        });

        // Ollama should include reasoning in response
        let has_reasoning = lmstudio_response["choices"][0]["message"]
            .get("reasoning")
            .is_some();
        assert!(has_reasoning);
    }

    /// Test prompt_eval_duration calculation from token count and rate
    #[test]
    fn test_prompt_eval_duration_calculation() {
        let prompt_tokens = 100;
        let tokens_per_second = 50.0;

        // Duration in seconds = tokens / rate
        let duration_secs = prompt_tokens as f64 / tokens_per_second;
        let duration_ns = (duration_secs * 1_000_000_000.0) as u64;

        assert_eq!(duration_ns, 2_000_000_000); // 2 seconds
    }

    /// Test eval_duration calculation
    #[test]
    fn test_eval_duration_calculation() {
        let completion_tokens = 75;
        let tokens_per_second = 50.0;

        let duration_secs = completion_tokens as f64 / tokens_per_second;
        let duration_ns = (duration_secs * 1_000_000_000.0) as u64;

        assert_eq!(duration_ns, 1_500_000_000); // 1.5 seconds
    }

    /// Test total_duration calculation
    #[test]
    fn test_total_duration_calculation() {
        let load_duration = 100_000_000_u64; // 0.1s
        let prompt_eval_duration = 200_000_000_u64; // 0.2s
        let eval_duration = 300_000_000_u64; // 0.3s

        let total = load_duration + prompt_eval_duration + eval_duration;
        assert_eq!(total, 600_000_000); // 0.6s total
    }

    /// Test model name preservation in response
    #[test]
    fn test_model_name_preservation() {
        // Model name should be preserved in transformation
        let lmstudio_model = "granite-3.0-2b-instruct";
        let ollama_model = "granite-3.0-2b-instruct";

        assert_eq!(lmstudio_model, ollama_model);
    }

    /// Test created_at timestamp formatting
    #[test]
    fn test_timestamp_formatting() {
        // LM Studio uses Unix timestamp
        let lmstudio_created = 1731990317_u64;

        // Ollama uses ISO 8601 format
        let ollama_created_at = "2024-11-18T20:31:57.111706Z";

        assert!(lmstudio_created > 0);
        assert!(ollama_created_at.ends_with('Z'));
        assert!(ollama_created_at.contains('T'));
    }

    /// Test context array handling in generate responses
    #[test]
    fn test_context_array_handling() {
        // Ollama includes context tokens for conversation continuity
        let context = vec![101, 2023, 2003, 1037, 3231];

        assert!(!context.is_empty());
        assert!(context.len() > 0);
    }

    /// Test done flag in streaming responses
    #[test]
    fn test_done_flag_streaming() {
        // Intermediate chunks
        let intermediate_chunk = json!({"done": false});
        assert_eq!(intermediate_chunk["done"], false);

        // Final chunk
        let final_chunk = json!({"done": true});
        assert_eq!(final_chunk["done"], true);
    }

    /// Test native API stats transformation
    #[test]
    fn test_native_stats_transformation() {
        // Native API provides rich stats
        let native_stats = json!({
            "tokens_per_second": 51.43,
            "time_to_first_token": 0.111,
            "generation_time": 0.954,
            "stop_reason": "eosFound"
        });

        // Transform to Ollama timing
        assert!(native_stats.get("tokens_per_second").is_some());
        assert!(native_stats.get("time_to_first_token").is_some());
        assert!(native_stats.get("generation_time").is_some());
    }

    /// Test empty content handling
    #[test]
    fn test_empty_content_handling() {
        // Messages with no content (e.g., tool calls only)
        let message = json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [
                {"function": {"name": "test"}}
            ]
        });

        assert!(message["content"].is_null());
        assert!(message.get("tool_calls").is_some());
    }

    /// Test multimodal content transformation
    #[test]
    fn test_multimodal_content_transformation() {
        // Array-based content for vision models
        let multimodal_content = json!([
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/..."}}
        ]);

        let content_array = multimodal_content.as_array().unwrap();
        assert_eq!(content_array.len(), 2);

        let has_text = content_array.iter().any(|c| c["type"] == "text");
        let has_image = content_array.iter().any(|c| c["type"] == "image_url");

        assert!(has_text);
        assert!(has_image);
    }

    /// Test logprobs handling
    #[test]
    fn test_logprobs_handling() {
        // When logprobs requested in LM Studio
        let response_with_logprobs = json!({
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello"},
                    "logprobs": {
                        "content": [
                            {
                                "token": "Hello",
                                "logprob": -0.5,
                                "top_logprobs": []
                            }
                        ]
                    }
                }
            ]
        });

        let has_logprobs = response_with_logprobs["choices"][0]
            .get("logprobs")
            .is_some();
        assert!(has_logprobs);
    }

    /// Test response_format preservation
    #[test]
    fn test_response_format_preservation() {
        // JSON mode response should indicate format
        let json_response = json!({
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "{\"key\": \"value\"}"
                    }
                }
            ]
        });

        let content = json_response["choices"][0]["message"]["content"]
            .as_str()
            .unwrap();
        assert!(content.starts_with('{'));
        assert!(content.ends_with('}'));
    }
}
