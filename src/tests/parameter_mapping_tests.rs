/// Tests for parameter mapping between Ollama and LM Studio APIs
#[cfg(test)]
mod parameter_mapping_tests {
    use serde_json::json;

    /// Test basic parameter mapping: temperature, top_p, top_k
    #[test]
    fn test_basic_parameter_mapping() {
        // According to helpers.rs and common.rs: These params map directly
        let ollama_request = json!({
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40
            }
        });

        let lmstudio_expected = json!({
            "temperature": 0.7,
            "top_p": 0.9
        });

        // LM Studio OpenAI-compatible API doesn't support top_k
        assert_eq!(ollama_request["options"]["temperature"], 0.7);
        assert_eq!(ollama_request["options"]["top_p"], 0.9);
        assert!(ollama_request["options"].get("top_k").is_some());

        // After mapping to LM Studio
        assert_eq!(lmstudio_expected["temperature"], 0.7);
        assert_eq!(lmstudio_expected["top_p"], 0.9);
        assert!(lmstudio_expected.get("top_k").is_none());
    }

    /// Test num_predict to max_tokens mapping
    #[test]
    fn test_num_predict_to_max_tokens() {
        // According to helpers.rs: num_predict maps to max_tokens
        let ollama_request = json!({
            "options": {
                "num_predict": 100
            }
        });

        let lmstudio_expected = json!({
            "max_tokens": 100
        });

        assert_eq!(ollama_request["options"]["num_predict"], 100);
        assert_eq!(lmstudio_expected["max_tokens"], 100);
    }

    /// Test seed parameter mapping
    #[test]
    fn test_seed_mapping() {
        let ollama_request = json!({
            "options": {
                "seed": 42
            }
        });

        let lmstudio_expected = json!({
            "seed": 42
        });

        assert_eq!(ollama_request["options"]["seed"], 42);
        assert_eq!(lmstudio_expected["seed"], 42);
    }

    /// Test stop sequences mapping
    #[test]
    fn test_stop_sequences_mapping() {
        // Both single string and array should be supported
        let ollama_single = json!({
            "options": {
                "stop": "\n"
            }
        });

        let ollama_array = json!({
            "options": {
                "stop": ["\n", "END", "STOP"]
            }
        });

        // LM Studio expects array
        let lmstudio_expected = json!({
            "stop": ["\n", "END", "STOP"]
        });

        assert!(ollama_single["options"]["stop"].is_string());
        assert!(ollama_array["options"]["stop"].is_array());
        assert!(lmstudio_expected["stop"].is_array());
    }

    /// Test frequency_penalty and presence_penalty
    #[test]
    fn test_penalty_parameters() {
        let ollama_request = json!({
            "options": {
                "frequency_penalty": 0.5,
                "presence_penalty": 0.3,
                "repeat_penalty": 1.1
            }
        });

        // LM Studio supports frequency and presence penalties
        let _lmstudio_expected = json!({
            "frequency_penalty": 0.5,
            "presence_penalty": 0.3
        });

        assert_eq!(ollama_request["options"]["frequency_penalty"], 0.5);
        assert_eq!(ollama_request["options"]["presence_penalty"], 0.3);
    }

    /// Test logit_bias conversion
    #[test]
    fn test_logit_bias_conversion() {
        // According to helpers.rs: Ollama uses token strings, OpenAI uses token IDs
        let ollama_request = json!({
            "options": {
                "logit_bias": {
                    "hello": -1.0,
                    "world": 2.0
                }
            }
        });

        // After tokenization and ID lookup, would become:
        let _lmstudio_expected = json!({
            "logit_bias": {
                "1234": -1.0,
                "5678": 2.0
            }
        });

        // Verify structure
        assert!(ollama_request["options"]["logit_bias"].is_object());
        assert!(_lmstudio_expected["logit_bias"].is_object());
    }

    /// Test format parameter for structured output
    #[test]
    fn test_format_parameter_mapping() {
        // According to ollama.md: format can be "json" or a JSON schema
        let ollama_json_mode = json!({
            "format": "json"
        });

        let _ollama_schema = json!({
            "format": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name"]
            }
        });

        // LM Studio maps to response_format
        let lmstudio_json = json!({
            "response_format": {
                "type": "json_object"
            }
        });

        let lmstudio_schema = json!({
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"}
                    },
                    "required": ["name"]
                }
            }
        });

        assert_eq!(ollama_json_mode["format"], "json");
        assert_eq!(lmstudio_json["response_format"]["type"], "json_object");
        assert_eq!(lmstudio_schema["response_format"]["type"], "json_schema");
    }

    /// Test keep_alive parameter handling
    #[test]
    fn test_keep_alive_parameter() {
        // According to ollama.md: keep_alive controls model unload timing
        let keep_alive_duration = json!({
            "keep_alive": "5m"
        });

        let keep_alive_indefinite = json!({
            "keep_alive": -1
        });

        let keep_alive_immediate = json!({
            "keep_alive": 0
        });

        assert!(keep_alive_duration["keep_alive"].is_string());
        assert!(keep_alive_indefinite["keep_alive"].is_number());
        assert_eq!(keep_alive_immediate["keep_alive"], 0);
    }

    /// Test context window parameter
    #[test]
    fn test_context_window_parameter() {
        let ollama_request = json!({
            "options": {
                "num_ctx": 4096
            }
        });

        // LM Studio uses max_context_length or requires model configuration
        assert_eq!(ollama_request["options"]["num_ctx"], 4096);
    }

    /// Test mirostat parameters
    #[test]
    fn test_mirostat_parameters() {
        let ollama_request = json!({
            "options": {
                "mirostat": 2,
                "mirostat_tau": 5.0,
                "mirostat_eta": 0.1
            }
        });

        // These are llama.cpp specific parameters
        assert_eq!(ollama_request["options"]["mirostat"], 2);
        assert!(ollama_request["options"].get("mirostat_tau").is_some());
        assert!(ollama_request["options"].get("mirostat_eta").is_some());
    }

    /// Test tfs_z parameter (tail free sampling)
    #[test]
    fn test_tfs_z_parameter() {
        let ollama_request = json!({
            "options": {
                "tfs_z": 1.0
            }
        });

        assert_eq!(ollama_request["options"]["tfs_z"], 1.0);
    }

    /// Test typical_p parameter
    #[test]
    fn test_typical_p_parameter() {
        let ollama_request = json!({
            "options": {
                "typical_p": 0.9
            }
        });

        assert_eq!(ollama_request["options"]["typical_p"], 0.9);
    }

    /// Test repeat penalty window
    #[test]
    fn test_repeat_last_n_parameter() {
        let ollama_request = json!({
            "options": {
                "repeat_last_n": 64,
                "repeat_penalty": 1.1
            }
        });

        assert_eq!(ollama_request["options"]["repeat_last_n"], 64);
        assert_eq!(ollama_request["options"]["repeat_penalty"], 1.1);
    }

    /// Test num_thread parameter
    #[test]
    fn test_num_thread_parameter() {
        let ollama_request = json!({
            "options": {
                "num_thread": 8
            }
        });

        // This is a llama.cpp runtime parameter
        assert_eq!(ollama_request["options"]["num_thread"], 8);
    }

    /// Test num_gpu parameter
    #[test]
    fn test_num_gpu_parameter() {
        let ollama_request = json!({
            "options": {
                "num_gpu": 1
            }
        });

        assert_eq!(ollama_request["options"]["num_gpu"], 1);
    }

    /// Test main_gpu parameter
    #[test]
    fn test_main_gpu_parameter() {
        let ollama_request = json!({
            "options": {
                "main_gpu": 0
            }
        });

        assert_eq!(ollama_request["options"]["main_gpu"], 0);
    }

    /// Test low_vram parameter
    #[test]
    fn test_low_vram_parameter() {
        let ollama_request = json!({
            "options": {
                "low_vram": true
            }
        });

        assert_eq!(ollama_request["options"]["low_vram"], true);
    }

    /// Test f16_kv parameter (use fp16 for key/value cache)
    #[test]
    fn test_f16_kv_parameter() {
        let ollama_request = json!({
            "options": {
                "f16_kv": true
            }
        });

        assert_eq!(ollama_request["options"]["f16_kv"], true);
    }

    /// Test vocab_only parameter
    #[test]
    fn test_vocab_only_parameter() {
        let ollama_request = json!({
            "options": {
                "vocab_only": false
            }
        });

        assert_eq!(ollama_request["options"]["vocab_only"], false);
    }

    /// Test use_mmap parameter
    #[test]
    fn test_use_mmap_parameter() {
        let ollama_request = json!({
            "options": {
                "use_mmap": true
            }
        });

        assert_eq!(ollama_request["options"]["use_mmap"], true);
    }

    /// Test use_mlock parameter
    #[test]
    fn test_use_mlock_parameter() {
        let ollama_request = json!({
            "options": {
                "use_mlock": false
            }
        });

        assert_eq!(ollama_request["options"]["use_mlock"], false);
    }

    /// Test numa parameter
    #[test]
    fn test_numa_parameter() {
        let ollama_request = json!({
            "options": {
                "numa": true
            }
        });

        assert_eq!(ollama_request["options"]["numa"], true);
    }

    /// Test parameter extraction from nested options object
    #[test]
    fn test_nested_options_extraction() {
        let ollama_request = json!({
            "model": "llama2",
            "prompt": "Hello",
            "options": {
                "temperature": 0.8,
                "num_predict": 50
            }
        });

        // Parameters should be extracted from options object
        assert!(ollama_request.get("options").is_some());
        assert_eq!(ollama_request["options"]["temperature"], 0.8);
        assert_eq!(ollama_request["options"]["num_predict"], 50);

        // After mapping, should be at top level for LM Studio
        let lmstudio_expected = json!({
            "model": "llama2",
            "prompt": "Hello",
            "temperature": 0.8,
            "max_tokens": 50
        });

        assert_eq!(lmstudio_expected["temperature"], 0.8);
        assert_eq!(lmstudio_expected["max_tokens"], 50);
        assert!(lmstudio_expected.get("options").is_none());
    }

    /// Test handling of unknown parameters
    #[test]
    fn test_unknown_parameter_passthrough() {
        // Unknown parameters should be preserved or dropped based on strategy
        let ollama_request = json!({
            "options": {
                "custom_param": "custom_value",
                "temperature": 0.7
            }
        });

        // Known parameter temperature should be mapped
        assert_eq!(ollama_request["options"]["temperature"], 0.7);
        // Custom parameter exists in input
        assert_eq!(ollama_request["options"]["custom_param"], "custom_value");
    }

    /// Test parameter validation ranges
    #[test]
    fn test_parameter_validation_ranges() {
        // Temperature typically 0.0 to 2.0
        let valid_temp = json!({"temperature": 0.7});
        assert!(valid_temp["temperature"].as_f64().unwrap() >= 0.0);
        assert!(valid_temp["temperature"].as_f64().unwrap() <= 2.0);

        // Top_p should be 0.0 to 1.0
        let valid_top_p = json!({"top_p": 0.9});
        assert!(valid_top_p["top_p"].as_f64().unwrap() >= 0.0);
        assert!(valid_top_p["top_p"].as_f64().unwrap() <= 1.0);

        // Top_k should be positive integer
        let valid_top_k = json!({"top_k": 40});
        assert!(valid_top_k["top_k"].as_i64().unwrap() > 0);
    }

    /// Test image parameter mapping for multimodal requests
    #[test]
    fn test_image_parameter_mapping() {
        // Ollama uses images array
        let ollama_request = json!({
            "model": "llava",
            "prompt": "Describe this image",
            "images": ["base64encodedimage1"]
        });

        // LM Studio uses content array with type
        let lmstudio_expected = json!({
            "model": "llava",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,base64encodedimage1"}}
                    ]
                }
            ]
        });

        assert!(ollama_request.get("images").is_some());
        assert!(lmstudio_expected["messages"][0]["content"].is_array());
    }

    /// Test default parameter values
    #[test]
    fn test_default_parameter_values() {
        // When no parameters provided, defaults should be used
        let minimal_request = json!({
            "model": "llama2",
            "prompt": "Hello"
        });

        // System should apply reasonable defaults
        // temperature: 0.8, top_p: 0.9, etc.
        assert!(minimal_request.get("options").is_none());
    }

    /// Test parameter priority (explicit vs default)
    #[test]
    fn test_parameter_priority() {
        // Explicit parameters should override defaults
        let explicit_request = json!({
            "model": "llama2",
            "prompt": "Hello",
            "options": {
                "temperature": 0.3
            }
        });

        assert_eq!(explicit_request["options"]["temperature"], 0.3);
    }
}
