//! Tests for TimingInfo translation from LM Studio stats to Ollama timing fields.
//!
//! Reference (LM Studio /api/v0/chat/completions stats):
//!   api_docs/lmstudio/1_developer/2_rest/endpoints.mdx lines 169-174
//!   "tokens_per_second": 51.43, "time_to_first_token": 0.111, "generation_time": 0.954
//!
//! Reference (LM Studio /v1/responses stats):
//!   api_docs/lmstudio/1_developer/2_rest/chat.md lines 319-341
//!   "input_tokens", "total_output_tokens", "tokens_per_second",
//!   "time_to_first_token_seconds", "model_load_time_seconds"
//!
//! Reference (Ollama timing fields):
//!   api_docs/ollama.md /api/chat and /api/generate response shape
//!   total_duration, load_duration, prompt_eval_count, prompt_eval_duration,
//!   eval_count, eval_duration (all in nanoseconds)

#[path = "../src/constants.rs"]
#[allow(dead_code)]
mod constants;

#[path = "../src/handlers/transform.rs"]
#[allow(dead_code)]
mod transform;

use serde_json::json;
use transform::TimingInfo;

/// LM Studio `/api/v0/*` response: `time_to_first_token` is the prompt-processing phase,
/// `generation_time` is the post-TTFT output-generation phase. Both are SEPARATE phases
/// and must NOT be subtracted from each other.
///
/// Real example from api_docs: ttft=0.111s, generation=0.954s, tokens_per_second=51.4
/// completion_tokens=53. Wall time = ttft + generation = 1.065s. tokens/sec on
/// generation_time = 53/0.954 ≈ 55.5 (matches Ollama's eval_duration interpretation:
/// generation_time is the eval phase duration, NOT total).
#[test]
fn timing_from_native_v0_stats_does_not_subtract_ttft() {
    let lm = json!({
        "choices": [{
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 24, "completion_tokens": 53, "total_tokens": 77},
        "stats": {
            "tokens_per_second": 51.43709529007664,
            "time_to_first_token": 0.111,
            "generation_time": 0.954,
            "stop_reason": "eosFound"
        }
    });

    let timing = TimingInfo::from_native_stats(&lm, 24, 53);

    let ttft_ns = 111_000_000u64;
    let gen_ns = 954_000_000u64;

    assert_eq!(
        timing.prompt_eval_duration, ttft_ns,
        "prompt_eval_duration must equal time_to_first_token (got {}, want {})",
        timing.prompt_eval_duration, ttft_ns
    );
    assert_eq!(
        timing.eval_duration, gen_ns,
        "eval_duration must equal generation_time, NOT generation_time - ttft (got {}, want {})",
        timing.eval_duration, gen_ns
    );
    assert_eq!(
        timing.total_duration,
        ttft_ns + gen_ns,
        "total_duration must equal ttft + generation_time (got {}, want {})",
        timing.total_duration,
        ttft_ns + gen_ns
    );
    assert_eq!(timing.prompt_eval_count, 24);
    assert_eq!(timing.eval_count, 53);
}

/// LM Studio `/v1/responses` and newer endpoints expose a different stats shape:
///   `input_tokens`, `total_output_tokens`, `time_to_first_token_seconds`,
///   `model_load_time_seconds`, `tokens_per_second`
///
/// Reference: api_docs/lmstudio/1_developer/2_rest/chat.md lines 319-341 and 387-396
///
/// The TimingInfo translator must recognize these field names. When the stats block
/// contains `time_to_first_token_seconds`, use it as the prompt phase; when there is no
/// explicit `generation_time` field, derive eval phase from
/// `total_output_tokens / tokens_per_second`.
#[test]
fn timing_from_native_v1_responses_stats() {
    let lm = json!({
        "model_instance_id": "ibm/granite-4-micro",
        "output": [{"type": "message", "content": "hi"}],
        "stats": {
            "input_tokens": 646,
            "total_output_tokens": 586,
            "reasoning_output_tokens": 0,
            "tokens_per_second": 29.753900615398926,
            "time_to_first_token_seconds": 1.088,
            "model_load_time_seconds": 2.656
        }
    });

    let timing = TimingInfo::from_native_stats(&lm, 0, 0);

    let ttft_ns = 1_088_000_000u64;
    // generation_time = output_tokens / tokens_per_second = 586 / 29.753... = 19.6948... seconds
    let expected_gen_ns = ((586.0_f64 / 29.753900615398926_f64) * 1_000_000_000.0) as u64;
    let load_ns = 2_656_000_000u64;

    assert_eq!(
        timing.prompt_eval_duration, ttft_ns,
        "prompt_eval_duration must equal time_to_first_token_seconds"
    );
    // allow small rounding drift (within 5ms)
    let drift = timing.eval_duration.abs_diff(expected_gen_ns);
    assert!(
        drift < 5_000_000,
        "eval_duration ≈ output_tokens / tokens_per_second; got {}, want ~{} (drift {})",
        timing.eval_duration,
        expected_gen_ns,
        drift
    );
    assert_eq!(
        timing.load_duration, load_ns,
        "load_duration must come from model_load_time_seconds when present"
    );
    assert_eq!(timing.prompt_eval_count, 646);
    assert_eq!(timing.eval_count, 586);
}
