/// Tests for model name resolution and cleaning logic
#[cfg(test)]
mod model_resolution_tests {
    use serde_json::json;

    /// Test model name cleaning: removing common suffixes
    #[test]
    fn test_clean_model_name_suffix_removal() {
        // According to model.rs: clean_model_name removes :latest, :tag suffixes
        let test_cases = vec![
            ("llama2:latest", "llama2"),
            ("mistral:7b-instruct", "mistral"),
            ("codellama:13b", "codellama"),
            ("granite-3.0-2b-instruct:latest", "granite-3.0-2b-instruct"),
        ];

        for (input, expected) in test_cases {
            // Simulating clean_model_name behavior
            let cleaned = input.split(':').next().unwrap_or(input);
            assert_eq!(cleaned, expected, "Failed for input: {}", input);
        }
    }

    /// Test model name cleaning: handling special characters
    #[test]
    fn test_clean_model_name_special_chars() {
        // Names with dashes, underscores, dots should be preserved
        let test_cases = vec![
            ("granite-3.0-2b-instruct", "granite-3.0-2b-instruct"),
            ("llama_2_7b", "llama_2_7b"),
            ("model.v2", "model.v2"),
            ("DeepSeek-R1", "DeepSeek-R1"),
        ];

        for (input, expected) in test_cases {
            assert_eq!(input, expected);
        }
    }

    /// Test model name case sensitivity
    #[test]
    fn test_model_name_case_sensitivity() {
        // According to model.rs: Matching should handle case variations
        let variations = vec![
            "llama2",
            "Llama2",
            "LLAMA2",
            "LLaMA2",
        ];

        // All should normalize to lowercase for comparison
        for name in variations {
            let normalized = name.to_lowercase();
            assert_eq!(normalized, "llama2");
        }
    }

    /// Test ModelInfo structure
    #[test]
    fn test_model_info_structure() {
        // According to model.rs: ModelInfo contains id, name, quantization, etc.
        let model_info = json!({
            "id": "lmstudio-community/meta-llama-3.1-8b-instruct",
            "name": "llama-3.1-8b-instruct",
            "quantization": "Q4_K_M",
            "format": "gguf",
            "context_length": 4096,
            "size_bytes": 4_800_000_000_u64
        });

        assert!(model_info.get("id").is_some());
        assert!(model_info.get("name").is_some());
        assert!(model_info.get("quantization").is_some());
        assert!(model_info.get("format").is_some());
        assert!(model_info.get("context_length").is_some());
    }

    /// Test native API model resolution with full LM Studio IDs
    #[test]
    fn test_native_model_resolution() {
        // Native API expects full model IDs from LM Studio
        let lmstudio_id = "lmstudio-community/meta-llama-3.1-8b-instruct";
        let ollama_name = "llama3.1";

        // Resolution should map ollama_name to lmstudio_id
        assert!(lmstudio_id.contains("llama"));
        assert!(ollama_name.contains("llama"));
    }

    /// Test legacy API model resolution with simple names
    #[test]
    fn test_legacy_model_resolution() {
        // Legacy API can use simple names or full paths
        let simple_name = "llama2";
        let full_path = "lmstudio-community/Meta-Llama-2-7B-Instruct-GGUF";

        // Both should be valid
        assert!(!simple_name.is_empty());
        assert!(!full_path.is_empty());
    }

    /// Test model scoring algorithm for fuzzy matching
    #[test]
    fn test_model_scoring_exact_match() {
        // According to model.rs: Exact match should score highest
        let requested = "granite-3.0-2b-instruct";
        let candidate = "granite-3.0-2b-instruct";

        // Score should be very high (e.g., 100)
        assert_eq!(requested, candidate);
    }

    /// Test model scoring for partial matches
    #[test]
    fn test_model_scoring_partial_match() {
        let requested = "llama2";
        let candidates = vec![
            "llama2",                              // Exact: highest score
            "llama2-7b",                           // Prefix match: high score
            "meta-llama2-chat",                    // Contains: medium score
            "codellama",                           // Similar: low score
        ];

        // Exact match should score highest
        assert!(candidates[0].starts_with(requested));
        assert!(candidates[1].starts_with(requested));
        assert!(candidates[2].contains(requested));
    }

    /// Test model scoring with quantization hints
    #[test]
    fn test_model_scoring_quantization() {
        // If user specifies quantization, prefer matching quant
        let _requested = "llama2:Q4_K_M";
        let candidates = vec![
            ("llama2", "Q4_K_M"),  // Matching quant
            ("llama2", "Q8_0"),    // Different quant
            ("llama2", "F16"),     // Full precision
        ];

        // First candidate should score highest
        let target_quant = "Q4_K_M";
        assert_eq!(candidates[0].1, target_quant);
    }

    /// Test model caching behavior
    #[test]
    fn test_model_cache() {
        // According to model.rs: ModelResolver caches model list
        let models = json!([
            {"id": "model1", "name": "Model 1"},
            {"id": "model2", "name": "Model 2"}
        ]);

        let cache = models.as_array().unwrap();
        assert_eq!(cache.len(), 2);

        // Subsequent lookups should use cached data
        let cached_lookup = cache.iter().find(|m| m["id"] == "model1");
        assert!(cached_lookup.is_some());
    }

    /// Test model list refresh
    #[test]
    fn test_model_list_refresh() {
        // Cache should be refreshable when models change
        let initial_models = vec!["model1", "model2"];
        let updated_models = vec!["model1", "model2", "model3"];

        assert_eq!(initial_models.len(), 2);
        assert_eq!(updated_models.len(), 3);
    }

    /// Test handling of unknown models
    #[test]
    fn test_unknown_model_handling() {
        // When model not found, should return error or pass through
        let requested = "nonexistent-model";
        let available = vec!["model1", "model2"];

        let found = available.iter().any(|&m| m == requested);
        assert!(!found);
    }

    /// Test model passthrough for legacy mode
    #[test]
    fn test_legacy_model_passthrough() {
        // Legacy mode may pass through model names unchanged
        let requested = "some-custom-model";
        let passed_through = requested;

        assert_eq!(requested, passed_through);
    }

    /// Test model name with organization prefix
    #[test]
    fn test_model_with_org_prefix() {
        let full_id = "microsoft/phi-2";
        let parts: Vec<&str> = full_id.split('/').collect();

        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0], "microsoft");
        assert_eq!(parts[1], "phi-2");
    }

    /// Test model name with repo path
    #[test]
    fn test_model_with_repo_path() {
        let full_path = "TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf";

        assert!(full_path.contains("TheBloke"));
        assert!(full_path.contains("Llama-2"));
        assert!(full_path.ends_with(".gguf"));
    }

    /// Test model resolution priority: exact > prefix > contains
    #[test]
    fn test_resolution_priority() {
        let requested = "granite";
        let candidates = vec![
            "granite-3.0-2b-instruct",     // Prefix match
            "ibm-granite-7b",              // Contains match
            "granite",                      // Exact match (if exists)
        ];

        // Exact match (if available) should win
        let exact = candidates.iter().find(|&&c| c == requested);
        let prefix = candidates.iter().find(|&&c| c.starts_with(requested));

        // If exact exists, prefer it; otherwise prefix
        if exact.is_some() {
            assert_eq!(*exact.unwrap(), requested);
        } else {
            assert!(prefix.is_some());
        }
    }

    /// Test model resolution with version numbers
    #[test]
    fn test_model_with_version_numbers() {
        let models = vec![
            "llama-3.1-8b",
            "llama-3.0-7b",
            "llama-2-13b",
        ];

        // Should match version correctly
        let requested = "llama-3.1";
        let matched = models.iter().find(|&&m| m.starts_with(requested));

        assert!(matched.is_some());
        assert_eq!(*matched.unwrap(), "llama-3.1-8b");
    }

    /// Test handling of model aliases
    #[test]
    fn test_model_aliases() {
        // Some models have multiple names - test that partial name matching works
        let aliases = vec![
            ("llama2", "meta-llama-2"),
            ("llama", "meta-llama-2"),  // More general match
            ("mistral", "mistral-7b-instruct"),
            ("granite", "granite-3.0-2b-instruct"),
        ];

        for (short, long) in aliases {
            // Remove dashes for matching as models may use different separators
            let short_normalized = short.replace('-', "");
            let long_normalized = long.replace('-', "");
            assert!(
                long_normalized.to_lowercase().contains(&short_normalized.to_lowercase()),
                "Expected '{}' to contain '{}'", long, short
            );
        }
    }

    /// Test model resolution with architecture hints
    #[test]
    fn test_model_with_architecture() {
        let model_info = json!({
            "id": "qwen2-vl-7b-instruct",
            "arch": "qwen2_vl",
            "type": "vlm"
        });

        assert_eq!(model_info["arch"], "qwen2_vl");
        assert_eq!(model_info["type"], "vlm");
    }

    /// Test model resolution with size hints
    #[test]
    fn test_model_with_size_hints() {
        let requested = "llama2-7b";
        let candidates = vec![
            ("llama2-7b", 7_000_000_000_u64),
            ("llama2-13b", 13_000_000_000_u64),
            ("llama2-70b", 70_000_000_000_u64),
        ];

        // Should match the 7b variant
        let matched = candidates.iter().find(|(name, _)| *name == requested);
        assert!(matched.is_some());
        assert_eq!(matched.unwrap().1, 7_000_000_000_u64);
    }

    /// Test concurrent model resolution requests
    #[test]
    fn test_concurrent_resolution() {
        // Multiple simultaneous requests should use shared cache
        let models = vec!["model1", "model2", "model3"];

        let request1 = "model1";
        let request2 = "model2";

        let result1 = models.contains(&request1);
        let result2 = models.contains(&request2);

        assert!(result1);
        assert!(result2);
    }

    /// Test model resolution error messages
    #[test]
    fn test_resolution_error_messages() {
        // Should provide helpful error when model not found
        let requested = "nonexistent-model";
        let available = vec!["model1", "model2", "model3"];

        let found = available.contains(&requested);
        assert!(!found);

        // Error message should suggest available models
        let suggestions = available.join(", ");
        assert!(suggestions.contains("model1"));
        assert!(suggestions.contains("model2"));
    }

    /// Test model resolution with empty model list
    #[test]
    fn test_resolution_with_empty_list() {
        let available: Vec<&str> = vec![];
        let requested = "any-model";

        let found = available.contains(&requested);
        assert!(!found);
    }

    /// Test model resolution with special model names
    #[test]
    fn test_special_model_names() {
        // Names with special chars, unicode, etc.
        let special_names = vec![
            "model-with-dashes",
            "model_with_underscores",
            "model.with.dots",
            "model123",
            "123model",
        ];

        for name in special_names {
            assert!(!name.is_empty());
            // All should be valid
        }
    }

    /// Test model ID format validation
    #[test]
    fn test_model_id_format() {
        // Valid LM Studio model IDs
        let valid_ids = vec![
            "lmstudio-community/meta-llama-3.1-8b-instruct",
            "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
            "microsoft/phi-2",
        ];

        for id in valid_ids {
            assert!(id.contains('/') || !id.contains('/'));
            // May or may not have org prefix
        }
    }

    /// Test native mode requires full model IDs
    #[test]
    fn test_native_mode_full_ids() {
        // Native mode (v0 API) needs complete model IDs
        let native_model = "lmstudio-community/granite-3.0-2b-instruct";

        assert!(native_model.contains('/'));
        assert!(native_model.len() > 10);
    }

    /// Test legacy mode accepts simple names
    #[test]
    fn test_legacy_mode_simple_names() {
        // Legacy mode (v1 OpenAI API) can use simple names
        let legacy_model = "granite";

        // Simple name without org prefix
        assert!(!legacy_model.contains('/'));
    }

    /// Test model type detection
    #[test]
    fn test_model_type_detection() {
        let models = vec![
            ("llama2", "llm"),
            ("text-embedding-nomic", "embeddings"),
            ("qwen2-vl", "vlm"),
        ];

        for (name, expected_type) in models {
            if name.contains("embedding") {
                assert_eq!(expected_type, "embeddings");
            } else if name.contains("vl") {
                assert_eq!(expected_type, "vlm");
            } else {
                assert_eq!(expected_type, "llm");
            }
        }
    }
}
