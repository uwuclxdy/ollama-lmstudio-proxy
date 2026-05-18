//! Ollama /api/generate `context` field handling.
//!
//! Per api_docs/ollama.md:
//!   - line 60: `context` (deprecated): "the context parameter returned from a
//!     previous request to /generate, this can be used to keep a short
//!     conversational memory"
//!   - line 116: `context` is "an encoding of the conversation used in this
//!     response" (an array of token IDs).
//!   - line 340: "raw mode will not return a context" — proving the field is
//!     legitimately omittable.
//!
//! LM Studio's OpenAI-compat endpoints do NOT expose token IDs, so the proxy
//! cannot synthesize a meaningful `context`. Returning `context: []` is
//! misleading: clients that chain context across calls see an empty history
//! and silently lose conversational memory. Omit the field entirely instead.

#[path = "../src/constants.rs"]
#[allow(dead_code)]
mod constants;

#[path = "../src/handlers/transform.rs"]
#[allow(dead_code)]
mod transform;

use serde_json::json;
use std::time::Instant;
use transform::ResponseTransformer;

#[test]
fn generate_response_does_not_emit_empty_context_array() {
    let lm = json!({
        "choices": [{
            "text": "hello",
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1}
    });
    let result = ResponseTransformer::convert_to_ollama_generate(
        &lm,
        "model",
        "prompt text",
        Instant::now(),
        false,
    );
    assert!(
        result.get("context").is_none(),
        "context must be omitted (proxy has no token IDs to fabricate); got {}",
        result
    );
}

#[test]
fn chat_response_does_not_emit_context() {
    // /api/chat never had a `context` field — guard against accidental leakage.
    let lm = json!({
        "choices": [{
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1}
    });
    let result =
        ResponseTransformer::convert_to_ollama_chat(&lm, "model", 1, Instant::now(), false);
    assert!(result.get("context").is_none(), "got {}", result);
}
