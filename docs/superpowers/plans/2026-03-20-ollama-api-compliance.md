# Ollama API Compliance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close all gaps between the proxy's current Ollama API behaviour and the full Ollama wire format as documented in `api docs/lmstudio_vs_ollama.md`.

**Architecture:** Eight self-contained tasks, each leaving the codebase in a compilable state. Tasks 1–3 are independent and can be done in any order. Task 4 refactors the request builder signature and must precede Tasks 5–6. Tasks 7–8 touch the response side and are independent of 1–6 but must be done in order relative to each other only if the same file is involved (they aren't).

**Tech Stack:** Rust 2024 edition, Warp, Serde/serde_json, tokio, reqwest. No new dependencies needed.

---

## File Map

| File | What changes |
|---|---|
| `src/model/types.rs` | Add `size_bytes`/`params_string` to `NativeModelData` and `ModelInfo`; use real values over heuristics |
| `src/handlers/ollama/health.rs` | Add `handle_ollama_root` handler |
| `src/server/routes.rs` | Register `GET /` route |
| `src/http/request.rs` | Add 3 direct params; add `TopLevelParams<'a>` struct; add `top_level` param to `build_lm_studio_request`; add `log_unsupported_options` |
| `src/handlers/ollama/chat.rs` | Extract `think`/`logprobs`/`top_logprobs`; pass `TopLevelParams` |
| `src/handlers/ollama/generate.rs` | Extract `think`/`logprobs`/`top_logprobs`/`suffix`; pass `TopLevelParams`; forward `suffix` on completions path |
| `src/handlers/ollama/embeddings.rs` | Update `build_lm_studio_request` call site to pass `None` for new param |
| `src/handlers/transform.rs` | Split reasoning extraction; fix `thinking` placement in chat and generate |
| `src/streaming/chunks.rs` | Add `thinking` to `ChoiceDeltaPayload`; update `None`-guard; update `create_ollama_streaming_chunk` sig |
| `src/streaming/sse.rs` | Update 3 call sites to pass `thinking` from payload; update 3 emit guards |

---

## Task 1: Accurate Model Metadata from LM Studio

**Spec:** Section 3b, 3c

**Files:**
- Modify: `src/model/types.rs`

### What & why
LM Studio's `/api/v1/models` response includes `size_bytes` and `params_string` fields. The proxy currently ignores both and falls back to name-based heuristics. This task parses the real values and uses them when available, falling back to heuristics only when absent.

### Steps

- [ ] **Step 1: Write the failing tests**

Add a `#[cfg(test)]` module at the bottom of `src/model/types.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn make_native(key: &str, size_bytes: Option<u64>, params_string: Option<String>) -> NativeModelData {
        NativeModelData {
            key: key.to_string(),
            model_type: "llm".to_string(),
            publisher: "test".to_string(),
            architecture: Some("llama".to_string()),
            format: Some("gguf".to_string()),
            quantization: Some(NativeQuantization { name: Some("Q4_K_M".to_string()) }),
            max_context_length: 4096,
            loaded_instances: vec![],
            capabilities: None,
            size_bytes,
            params_string,
        }
    }

    #[test]
    fn uses_real_size_bytes_when_present() {
        let native = make_native("mymodel", Some(4_200_000_000), None);
        let info = ModelInfo::from_native_data(&native);
        assert_eq!(info.calculate_estimated_size(), 4_200_000_000);
    }

    #[test]
    fn falls_back_to_heuristic_when_size_bytes_absent() {
        let native = make_native("llama-7b", None, None);
        let info = ModelInfo::from_native_data(&native);
        // heuristic for "7b" with Q4 gives ~3.85GB — just check it's non-zero and not exact
        assert!(info.calculate_estimated_size() > 0);
        assert_ne!(info.calculate_estimated_size(), 4_200_000_000);
    }

    #[test]
    fn uses_real_params_string_when_present() {
        let native = make_native("somemodel", None, Some("13B".to_string()));
        let info = ModelInfo::from_native_data(&native);
        assert_eq!(info.parse_parameters().size_string, "13B");
    }

    #[test]
    fn falls_back_to_inferred_params_when_absent() {
        let native = make_native("llama-7b-instruct", None, None);
        let info = ModelInfo::from_native_data(&native);
        assert_eq!(info.parse_parameters().size_string, "7B");
    }
}
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd /home/uwuclxdy/repos/rs/ollama-lmstudio-proxy
cargo test --lib 2>&1 | tail -20
```

Expected: compile error — `NativeModelData` has no `size_bytes` field, `ModelInfo` has no `size_bytes`/`params_string`, and `calculate_estimated_size`/`parse_parameters` are private.

- [ ] **Step 3: Make `calculate_estimated_size` and `parse_parameters` pub(crate) for testing**

In `src/model/types.rs`, change:
```rust
fn calculate_estimated_size(&self) -> u64 {
```
to:
```rust
pub(crate) fn calculate_estimated_size(&self) -> u64 {
```
And:
```rust
fn parse_parameters(&self) -> ModelParameters {
```
to:
```rust
pub(crate) fn parse_parameters(&self) -> ModelParameters {
```

- [ ] **Step 4: Add `size_bytes` to `NativeModelData`**

In the `NativeModelData` struct, after `capabilities`:
```rust
#[serde(default)]
pub size_bytes: Option<u64>,
```

- [ ] **Step 5: Add `params_string` to `NativeModelData`**

```rust
#[serde(default)]
pub params_string: Option<String>,
```

- [ ] **Step 6: Add `size_bytes` and `params_string` to `ModelInfo`**

In the `ModelInfo` struct, after `supports_tools`:
```rust
pub size_bytes: Option<u64>,
pub params_string: Option<String>,
```

- [ ] **Step 7: Transfer values in `from_native_data`**

In `ModelInfo::from_native_data`, add to the `Self { ... }` block:
```rust
size_bytes: native_data.size_bytes,
params_string: native_data.params_string.clone(),
```

- [ ] **Step 8: Update `calculate_estimated_size` to use real value first**

Replace the first line of the function body:
```rust
pub(crate) fn calculate_estimated_size(&self) -> u64 {
    if let Some(bytes) = self.size_bytes {
        return bytes;
    }
    let lower_id = self.id.to_lowercase();
    // ... existing heuristic unchanged ...
```

- [ ] **Step 9: Update `parse_parameters` to use real value first**

```rust
pub(crate) fn parse_parameters(&self) -> ModelParameters {
    if let Some(ref s) = self.params_string {
        return ModelParameters { size_string: s.clone() };
    }
    let lower_id = self.id.to_lowercase();
    // ... existing inference unchanged ...
```

- [ ] **Step 10: Fix `with_alias_name` to propagate new fields**

`with_alias_name` clones `self` and changes only `ollama_name`, so `size_bytes` and `params_string` are automatically propagated. Verify by reading the method — no change needed.

- [ ] **Step 11: Run tests**

```bash
cargo test --lib model::types 2>&1
```

Expected: 4 tests pass.

- [ ] **Step 12: Compile check**

```bash
cargo check 2>&1 | tail -10
```

Expected: no errors (fields are only added, no existing code broken).

- [ ] **Step 13: Commit**

```bash
git add src/model/types.rs
git commit -m "feat: use real size_bytes and params_string from LM Studio model data"
```

---

## Task 2: `GET /` Root Endpoint

**Spec:** Section 3a

**Files:**
- Modify: `src/handlers/ollama/health.rs`
- Modify: `src/server/routes.rs`

### What & why
Ollama responds to `GET /` with the plain-text string `"Ollama is running"`. Many clients (Open WebUI, LiteLLM) probe this to detect Ollama presence. Currently the proxy returns 404.

### Steps

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)]` module in `src/handlers/ollama/health.rs` (create the module if absent):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use http_body_util::BodyExt;

    #[tokio::test]
    async fn root_returns_ollama_is_running() {
        let response = handle_ollama_root().await.unwrap();
        assert_eq!(response.status(), 200);
        // Verify Content-Type header
        let ct = response.headers()
            .get("Content-Type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(ct.contains("text/plain"), "expected text/plain, got: {}", ct);
        // Verify body
        let body = response.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(body.as_ref(), b"Ollama is running");
    }
}
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cargo test --lib handlers::ollama::health 2>&1 | tail -10
```

Expected: compile error — `handle_ollama_root` does not exist.

- [ ] **Step 3: Implement `handle_ollama_root`**

Add to `src/handlers/ollama/health.rs`, after the existing `handle_ollama_version` function. Use the same `warp::http::Response::builder().body("...".into())` pattern as `json_response` in `src/http/response.rs` — the `.into()` converts a `String`/`&'static str` to the `warp::reply::Response` body type:

```rust
pub async fn handle_ollama_root() -> Result<warp::reply::Response, ProxyError> {
    Ok(warp::http::Response::builder()
        .status(warp::http::StatusCode::OK)
        .header("Content-Type", "text/plain; charset=utf-8")
        .body("Ollama is running".into())
        .unwrap_or_else(|_| {
            warp::http::Response::builder()
                .status(warp::http::StatusCode::INTERNAL_SERVER_ERROR)
                .body("Internal Server Error".into())
                .unwrap()
        }))
}
```

- [ ] **Step 4: Run test**

```bash
cargo test --lib handlers::ollama::health 2>&1 | tail -10
```

Expected: 1 test passes.

- [ ] **Step 5: Register the route**

In `src/server/routes.rs`, add a new route before the `health_route` definition:

```rust
let root_route = warp::path::end()
    .and(warp::get())
    .and_then(|| async move {
        ollama::handle_ollama_root()
            .await
            .map_err(warp::reject::custom)
    });
```

Add `root_route` to the `or` chain first (conventional placement; `warp::path::end()` is unambiguous and won't conflict with any other route since no other route matches an empty path):

```rust
root_route
    .or(health_route)
    .or(ollama_tags_route)
    // ... rest unchanged
```

Also add `handle_ollama_root` to the re-export in `src/handlers/ollama/mod.rs`. Find the existing line:
```rust
pub use health::{handle_health_check, handle_ollama_version};
```
and extend it to:
```rust
pub use health::{handle_health_check, handle_ollama_root, handle_ollama_version};
```

- [ ] **Step 6: Compile check**

```bash
cargo check 2>&1 | tail -10
```

- [ ] **Step 7: Commit**

```bash
git add src/handlers/ollama/health.rs src/server/routes.rs src/handlers/ollama/mod.rs
git commit -m "feat: add GET / root endpoint returning 'Ollama is running'"
```

---

## Task 3: Direct Option Forwarding + Unsupported Options Logging

**Spec:** Section 1a, 1d

**Files:**
- Modify: `src/http/request.rs`

### What & why
Three Ollama `options.*` parameters (`presence_penalty`, `frequency_penalty`, `min_p`) are forwarded to LM Studio's OpenAI-compat endpoint directly. Sixteen unsupported parameters (mirostat, num_ctx, etc.) are currently silently dropped; they should emit a debug-level log line so operators can see what's being ignored.

### Steps

- [ ] **Step 1: Write the failing tests**

Add a `#[cfg(test)]` module to `src/http/request.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn forwards_presence_penalty() {
        let options = json!({ "presence_penalty": 0.5 });
        let params = map_ollama_to_lmstudio_params(Some(&options), None);
        assert_eq!(params.get("presence_penalty"), Some(&json!(0.5)));
    }

    #[test]
    fn forwards_frequency_penalty() {
        let options = json!({ "frequency_penalty": 0.3 });
        let params = map_ollama_to_lmstudio_params(Some(&options), None);
        assert_eq!(params.get("frequency_penalty"), Some(&json!(0.3)));
    }

    #[test]
    fn forwards_min_p() {
        let options = json!({ "min_p": 0.05 });
        let params = map_ollama_to_lmstudio_params(Some(&options), None);
        assert_eq!(params.get("min_p"), Some(&json!(0.05)));
    }

    #[test]
    fn collects_unsupported_options() {
        let options = json!({ "num_ctx": 4096, "mirostat": 1, "temperature": 0.7 });
        let unsupported = collect_unsupported_keys(&options);
        assert!(unsupported.contains(&"num_ctx"), "expected num_ctx in {:?}", unsupported);
        assert!(unsupported.contains(&"mirostat"), "expected mirostat in {:?}", unsupported);
        // temperature is supported — must NOT appear
        assert!(!unsupported.contains(&"temperature"), "temperature should not appear in {:?}", unsupported);
    }

    #[test]
    fn log_unsupported_options_does_not_panic() {
        // Smoke test: calling with unsupported keys must not panic (log output not asserted)
        let options = json!({ "num_ctx": 4096, "mirostat": 1 });
        log_unsupported_options(&options);

        // No keys: also must not panic
        let empty = json!({ "temperature": 0.7 });
        log_unsupported_options(&empty);
    }
}
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cargo test --lib http::request 2>&1 | tail -10
```

- [ ] **Step 3: Add the three params to `DIRECT_MAPPINGS`**

In `map_direct_params`, change:
```rust
const DIRECT_MAPPINGS: &[&str] = &[
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "stop",
    "truncate",
    "dimensions",
];
```
to:
```rust
const DIRECT_MAPPINGS: &[&str] = &[
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "stop",
    "truncate",
    "dimensions",
    "presence_penalty",
    "frequency_penalty",
    "min_p",
];
```

- [ ] **Step 4: Add `collect_unsupported_keys` and `log_unsupported_options`**

Add after `map_format_params`:

```rust
const UNSUPPORTED_OPTION_KEYS: &[&str] = &[
    "num_ctx",
    "repeat_last_n",
    "tfs_z",
    "typical_p",
    "mirostat",
    "mirostat_tau",
    "mirostat_eta",
    "penalize_newline",
    "num_keep",
    "num_batch",
    "num_gpu",
    "num_thread",
    "numa",
    "use_mmap",
    "use_mlock",
    "vocab_only",
];

pub(crate) fn collect_unsupported_keys(options: &Value) -> Vec<&'static str> {
    UNSUPPORTED_OPTION_KEYS
        .iter()
        .copied()
        .filter(|key| options.get(key).is_some())
        .collect()
}

pub fn log_unsupported_options(options: &Value) {
    let keys = collect_unsupported_keys(options);
    if !keys.is_empty() {
        log::debug!("unsupported options ignored: {}", keys.join(", "));
    }
}
```

- [ ] **Step 5: Call `log_unsupported_options` from `map_ollama_to_lmstudio_params`**

In `map_ollama_to_lmstudio_params`, after the four existing function calls:
```rust
pub fn map_ollama_to_lmstudio_params(
    ollama_options: Option<&Value>,
    structured_format: Option<&Value>,
) -> serde_json::Map<String, Value> {
    let mut params = serde_json::Map::new();

    map_direct_params(ollama_options, &mut params);
    map_token_limits(ollama_options, &mut params);
    map_penalties(ollama_options, &mut params);
    map_format_params(ollama_options, structured_format, &mut params);

    if let Some(options) = ollama_options {
        log_unsupported_options(options);
    }

    params
}
```

- [ ] **Step 6: Run tests**

```bash
cargo test --lib http::request 2>&1 | tail -15
```

Expected: 4 tests pass.

- [ ] **Step 7: Compile check**

```bash
cargo check 2>&1 | tail -10
```

- [ ] **Step 8: Commit**

```bash
git add src/http/request.rs
git commit -m "feat: forward presence_penalty, frequency_penalty, min_p; log unsupported options"
```

---

## Task 4: `TopLevelParams` Struct + `build_lm_studio_request` Signature

**Spec:** Section 1b (passing mechanism)

**Files:**
- Modify: `src/http/request.rs`
- Modify: `src/handlers/ollama/chat.rs` (call site only — pass `None`)
- Modify: `src/handlers/ollama/generate.rs` (call site only — pass `None`)
- Modify: `src/handlers/ollama/embeddings.rs` (call site only — pass `None`)

### What & why
`build_lm_studio_request` currently has no way to receive top-level body parameters (`think`, `logprobs`, `top_logprobs`). This task adds the `TopLevelParams<'a>` struct and wires it into the builder. All existing call sites pass `None` for now — Tasks 5 and 6 fill in the real values.

> **Important:** All three call sites must be updated in the same commit as the signature change, or the codebase will not compile.

### Steps

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)]` module in `src/http/request.rs`:

```rust
#[test]
fn top_level_params_think_true_emits_reasoning_on() {
    let think_val = json!(true);
    let top = TopLevelParams {
        think: Some(&think_val),
        logprobs: None,
        top_logprobs: None,
    };
    let request = build_lm_studio_request(
        "mymodel",
        LMStudioRequestType::Completion {
            prompt: std::borrow::Cow::Borrowed("hello"),
            stream: false,
        },
        None,
        None,
        None,
        Some(&top),
    );
    assert_eq!(request.get("reasoning"), Some(&json!("on")));
}

#[test]
fn top_level_params_think_false_emits_reasoning_off() {
    let think_val = json!(false);
    let top = TopLevelParams {
        think: Some(&think_val),
        logprobs: None,
        top_logprobs: None,
    };
    let request = build_lm_studio_request(
        "mymodel",
        LMStudioRequestType::Completion {
            prompt: std::borrow::Cow::Borrowed("hello"),
            stream: false,
        },
        None,
        None,
        None,
        Some(&top),
    );
    assert_eq!(request.get("reasoning"), Some(&json!("off")));
}

#[test]
fn top_level_params_think_string_passes_through() {
    let think_val = json!("high");
    let top = TopLevelParams {
        think: Some(&think_val),
        logprobs: None,
        top_logprobs: None,
    };
    let request = build_lm_studio_request(
        "mymodel",
        LMStudioRequestType::Completion {
            prompt: std::borrow::Cow::Borrowed("hello"),
            stream: false,
        },
        None,
        None,
        None,
        Some(&top),
    );
    assert_eq!(request.get("reasoning"), Some(&json!("high")));
}

#[test]
fn top_level_params_absent_think_emits_no_reasoning() {
    let top = TopLevelParams { think: None, logprobs: None, top_logprobs: None };
    let request = build_lm_studio_request(
        "mymodel",
        LMStudioRequestType::Completion {
            prompt: std::borrow::Cow::Borrowed("hello"),
            stream: false,
        },
        None,
        None,
        None,
        Some(&top),
    );
    assert!(request.get("reasoning").is_none());
}

#[test]
fn top_level_params_logprobs_forwarded() {
    let lp = json!(true);
    let top = TopLevelParams { think: None, logprobs: Some(&lp), top_logprobs: None };
    let request = build_lm_studio_request(
        "mymodel",
        LMStudioRequestType::Completion {
            prompt: std::borrow::Cow::Borrowed("hello"),
            stream: false,
        },
        None,
        None,
        None,
        Some(&top),
    );
    assert_eq!(request.get("logprobs"), Some(&json!(true)));
}

#[test]
fn top_level_params_work_on_chat_type_too() {
    // think applies to both chat and generate paths
    let think_val = json!("medium");
    let top = TopLevelParams { think: Some(&think_val), logprobs: None, top_logprobs: None };
    let messages = json!([{"role": "user", "content": "hi"}]);
    let request = build_lm_studio_request(
        "mymodel",
        LMStudioRequestType::Chat { messages: &messages, stream: false },
        None,
        None,
        None,
        Some(&top),
    );
    assert_eq!(request.get("reasoning"), Some(&json!("medium")));
}
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cargo test --lib http::request 2>&1 | tail -10
```

- [ ] **Step 3: Add `TopLevelParams<'a>` struct**

In `src/http/request.rs`, after the existing `LMStudioRequestType` enum:

```rust
pub struct TopLevelParams<'a> {
    pub think: Option<&'a Value>,
    pub logprobs: Option<&'a Value>,
    pub top_logprobs: Option<&'a Value>,
}
```

- [ ] **Step 4: Add `apply_top_level_params` helper**

```rust
fn apply_top_level_params(top: &TopLevelParams<'_>, request_obj: &mut serde_json::Map<String, Value>) {
    if let Some(think_val) = top.think {
        let reasoning: Value = match think_val {
            Value::Bool(true) => json!("on"),
            Value::Bool(false) => json!("off"),
            Value::String(s) => json!(s),
            other => other.clone(),
        };
        request_obj.insert("reasoning".to_string(), reasoning);
    }
    if let Some(lp) = top.logprobs {
        request_obj.insert("logprobs".to_string(), lp.clone());
    }
    if let Some(tlp) = top.top_logprobs {
        request_obj.insert("top_logprobs".to_string(), tlp.clone());
    }
}
```

- [ ] **Step 5: Update `build_lm_studio_request` signature**

Add `top_level: Option<&TopLevelParams<'_>>` as the last parameter:

```rust
pub fn build_lm_studio_request(
    model_lm_studio_id: &str,
    request_type: LMStudioRequestType,
    ollama_options: Option<&Value>,
    ollama_tools: Option<&Value>,
    structured_format: Option<&Value>,
    top_level: Option<&TopLevelParams<'_>>,
) -> Value {
```

At the end of the function, before `request_json`, add:

```rust
if let Some(top) = top_level {
    if let Some(request_obj) = request_json.as_object_mut() {
        apply_top_level_params(top, request_obj);
    }
}
```

- [ ] **Step 6: Update all three call sites to pass `None`**

In `src/handlers/ollama/chat.rs`, change the `build_lm_studio_request` call to add `, None` at the end:
```rust
let mut lm_request = crate::http::request::build_lm_studio_request(
    &resolution_ctx.lm_studio_model_id,
    LMStudioRequestType::Chat {
        messages: &messages_with_images,
        stream,
    },
    resolution_ctx.effective_options.as_ref(),
    ollama_tools,
    resolution_ctx.effective_format.as_ref(),
    None,  // TopLevelParams — wired in Task 5
);
```

In `src/handlers/ollama/generate.rs`, same change:
```rust
let mut lm_request = crate::http::request::build_lm_studio_request(
    &resolution_ctx.lm_studio_model_id,
    lm_request_type,
    resolution_ctx.effective_options.as_ref(),
    None,
    resolution_ctx.effective_format.as_ref(),
    None,  // TopLevelParams — wired in Task 6
);
```

In `src/handlers/ollama/embeddings.rs`:
```rust
let mut lm_request = crate::http::request::build_lm_studio_request(
    &resolution_ctx.lm_studio_model_id,
    LMStudioRequestType::Embeddings {
        input: &input_value,
    },
    resolution_ctx.effective_options.as_ref(),
    None,
    None,
    None,  // TopLevelParams — not applicable for embeddings
);
```

- [ ] **Step 7: Run tests**

```bash
cargo test --lib http::request 2>&1 | tail -15
```

Expected: all tests in `http::request` pass (9 tests now).

- [ ] **Step 8: Compile check**

```bash
cargo check 2>&1 | tail -10
```

- [ ] **Step 9: Commit**

```bash
git add src/http/request.rs src/handlers/ollama/chat.rs src/handlers/ollama/generate.rs src/handlers/ollama/embeddings.rs
git commit -m "feat: add TopLevelParams struct and extend build_lm_studio_request"
```

---

## Task 5: Wire `think`/`logprobs`/`top_logprobs` in Chat Handler

**Spec:** Section 1b

**Files:**
- Modify: `src/handlers/ollama/chat.rs`

### What & why
Extract `think`, `logprobs`, and `top_logprobs` from the Ollama chat request body and forward them to LM Studio via `TopLevelParams`. The extraction is done in a small private helper function so it can be unit-tested without invoking the full async handler.

### Steps

- [ ] **Step 1: Write the failing test**

Add a `#[cfg(test)]` module to `src/handlers/ollama/chat.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extracts_think_from_body() {
        let body = json!({ "think": true, "model": "x", "messages": [] });
        let top = make_top_level_params(&body);
        assert!(top.think.is_some());
        assert_eq!(top.think, Some(&json!(true)));
    }

    #[test]
    fn absent_think_gives_none() {
        let body = json!({ "model": "x", "messages": [] });
        let top = make_top_level_params(&body);
        assert!(top.think.is_none());
    }

    #[test]
    fn extracts_logprobs_and_top_logprobs() {
        let body = json!({ "logprobs": true, "top_logprobs": 3 });
        let top = make_top_level_params(&body);
        assert_eq!(top.logprobs, Some(&json!(true)));
        assert_eq!(top.top_logprobs, Some(&json!(3)));
    }
}
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cargo test --lib handlers::ollama::chat 2>&1 | tail -10
```

Expected: compile error — `make_top_level_params` does not exist.

- [ ] **Step 3: Add `make_top_level_params` production helper and use it in the handler**

In `src/handlers/ollama/chat.rs`, add a private helper function (outside any `impl` block):

```rust
use crate::http::request::TopLevelParams;

fn make_top_level_params(body: &serde_json::Value) -> TopLevelParams<'_> {
    TopLevelParams {
        think: body.get("think"),
        logprobs: body.get("logprobs"),
        top_logprobs: body.get("top_logprobs"),
    }
}
```

Inside the `operation` closure, before the `build_lm_studio_request` call, add:

```rust
let top_level_params = make_top_level_params(&body_clone);
```

Then change the `build_lm_studio_request` call's last argument from `None` to `Some(&top_level_params)`.

- [ ] **Step 4: Run tests**

```bash
cargo test --lib handlers::ollama::chat 2>&1 | tail -10
```

- [ ] **Step 5: Compile check**

```bash
cargo check 2>&1 | tail -10
```

- [ ] **Step 6: Commit**

```bash
git add src/handlers/ollama/chat.rs
git commit -m "feat: forward think, logprobs, top_logprobs from chat requests"
```

---

## Task 6: Wire `think`/`logprobs`/`top_logprobs`/`suffix` in Generate Handler

**Spec:** Section 1b, 1c

**Files:**
- Modify: `src/handlers/ollama/generate.rs`

### What & why
Same as Task 5 but for `/api/generate`. Extract `think`/`logprobs`/`top_logprobs` using the same private helper pattern. Additionally, `suffix` must be forwarded to `/v1/completions` on the text path, and logged+dropped on the vision path.

### Steps

- [ ] **Step 1: Write the failing tests**

Add a `#[cfg(test)]` module to `src/handlers/ollama/generate.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extracts_think_from_generate_body() {
        let body = json!({ "think": "high", "model": "x", "prompt": "hi" });
        let top = make_top_level_params(&body);
        assert_eq!(top.think, Some(&json!("high")));
    }

    #[test]
    fn suffix_inserted_into_lm_request() {
        use crate::http::request::{build_lm_studio_request, LMStudioRequestType, TopLevelParams};
        use std::borrow::Cow;

        let body = json!({ "suffix": "world", "model": "test", "prompt": "hello" });
        let suffix_val = body.get("suffix");
        let top_level = TopLevelParams { think: None, logprobs: None, top_logprobs: None };

        let mut lm_request = build_lm_studio_request(
            "test",
            LMStudioRequestType::Completion { prompt: Cow::Borrowed("hello"), stream: false },
            None,           // ollama_options
            None,           // ollama_tools
            None,           // structured_format
            Some(&top_level), // top_level (new param added in Task 4)
        );

        // Mirror the insertion logic from generate.rs
        if let Some(s) = suffix_val {
            if let Some(obj) = lm_request.as_object_mut() {
                obj.insert("suffix".to_string(), s.clone());
            }
        }

        assert_eq!(lm_request.get("suffix"), Some(&json!("world")));
    }

    #[test]
    fn suffix_not_inserted_on_vision_path() {
        // When images are present (vision path), suffix must NOT be inserted
        let body = json!({ "suffix": "world", "model": "test", "prompt": "hello",
                           "images": ["base64data"] });
        let current_images = body.get("images");
        let suffix_val = body.get("suffix");
        let mut lm_request = json!({ "model": "test" });

        // Mirror the guard from generate.rs
        if current_images.is_none() {
            if let Some(s) = suffix_val {
                if let Some(obj) = lm_request.as_object_mut() {
                    obj.insert("suffix".to_string(), s.clone());
                }
            }
        }

        assert!(lm_request.get("suffix").is_none(), "suffix must be absent on vision path");
    }

    #[test]
    fn absent_think_gives_none_in_generate() {
        let body = json!({ "model": "x", "prompt": "hi" });
        let top = make_top_level_params(&body);
        assert!(top.think.is_none());
    }
}
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cargo test --lib handlers::ollama::generate 2>&1 | tail -10
```

Expected: compile error — `make_top_level_params` does not exist.

- [ ] **Step 3: Add `make_top_level_params` production helper and wire it into the handler**

In `src/handlers/ollama/generate.rs`, add (mirroring chat.rs):

```rust
use crate::http::request::TopLevelParams;

fn make_top_level_params(body: &serde_json::Value) -> TopLevelParams<'_> {
    TopLevelParams {
        think: body.get("think"),
        logprobs: body.get("logprobs"),
        top_logprobs: body.get("top_logprobs"),
    }
}
```

Inside the `operation` closure, before the `build_lm_studio_request` call, add:

```rust
let top_level_params = make_top_level_params(&body_clone);
let suffix_val = body_clone.get("suffix");

// Log suffix if it cannot be used (vision path)
if current_images.is_some() && suffix_val.is_some() {
    log::debug!("unsupported on vision path: suffix");
}
```

Update `build_lm_studio_request` call's last argument from `None` to `Some(&top_level_params)`.

After the `build_lm_studio_request` call, on the completions (non-vision) branch, apply `suffix`:

```rust
// Apply suffix only on the completions path
if current_images.is_none() {
    if let Some(s) = suffix_val {
        if let Some(obj) = lm_request.as_object_mut() {
            obj.insert("suffix".to_string(), s.clone());
        }
    }
}
```

The right place to insert this is immediately after `let mut lm_request = ...`, before the `apply_keep_alive_ttl` call. Since `lm_request` is always mutable here, the insertion is safe regardless of path.

- [ ] **Step 4: Run tests**

```bash
cargo test --lib handlers::ollama::generate 2>&1 | tail -10
```

- [ ] **Step 5: Compile check**

```bash
cargo check 2>&1 | tail -10
```

- [ ] **Step 6: Commit**

```bash
git add src/handlers/ollama/generate.rs
git commit -m "feat: forward think, logprobs, top_logprobs, suffix from generate requests"
```

---

## Task 7: Fix Reasoning Content in Non-Streaming Responses

**Spec:** Section 2a, 2c

**Files:**
- Modify: `src/handlers/transform.rs`

### What & why
Currently `extract_chat_content_with_reasoning` merges reasoning into content as markdown. Ollama expects reasoning in `message.thinking` (for chat) and `thinking` (top-level, for generate). This task fixes both non-streaming shapes.

### Steps

- [ ] **Step 1: Write the failing tests**

Add a `#[cfg(test)]` module to `src/handlers/transform.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::Instant;

    fn lm_chat_response(content: &str, reasoning: Option<&str>) -> serde_json::Value {
        let mut msg = json!({ "content": content });
        if let Some(r) = reasoning {
            msg.as_object_mut().unwrap().insert("reasoning".to_string(), json!(r));
        }
        json!({
            "choices": [{ "message": msg, "finish_reason": "stop" }],
            "usage": { "prompt_tokens": 10, "completion_tokens": 5 }
        })
    }

    fn lm_completion_response(text: &str, reasoning: Option<&str>) -> serde_json::Value {
        let mut choice = json!({ "text": text, "finish_reason": "stop" });
        if let Some(r) = reasoning {
            choice.as_object_mut().unwrap().insert("reasoning".to_string(), json!(r));
        }
        json!({
            "choices": [choice],
            "usage": { "prompt_tokens": 10, "completion_tokens": 5 }
        })
    }

    #[test]
    fn chat_response_thinking_in_message_not_content() {
        let lm = lm_chat_response("The answer is 42", Some("Let me think..."));
        let result = ResponseTransformer::convert_to_ollama_chat(&lm, "mymodel", 2, Instant::now(), false);
        let msg = result.get("message").unwrap();
        assert_eq!(msg.get("content").and_then(|v| v.as_str()), Some("The answer is 42"));
        assert_eq!(msg.get("thinking").and_then(|v| v.as_str()), Some("Let me think..."));
        // must not be merged into content
        assert!(!msg.get("content").unwrap().as_str().unwrap().contains("Reasoning"));
    }

    #[test]
    fn chat_response_no_thinking_field_when_absent() {
        let lm = lm_chat_response("The answer is 42", None);
        let result = ResponseTransformer::convert_to_ollama_chat(&lm, "mymodel", 2, Instant::now(), false);
        let msg = result.get("message").unwrap();
        assert!(msg.get("thinking").is_none());
    }

    #[test]
    fn generate_response_thinking_top_level() {
        let lm = lm_completion_response("42", Some("Let me reason"));
        let result = ResponseTransformer::convert_to_ollama_generate(&lm, "mymodel", "what is the answer?", Instant::now(), false);
        assert_eq!(result.get("response").and_then(|v| v.as_str()), Some("42"));
        assert_eq!(result.get("thinking").and_then(|v| v.as_str()), Some("Let me reason"));
    }

    #[test]
    fn generate_response_no_thinking_field_when_absent() {
        let lm = lm_completion_response("42", None);
        let result = ResponseTransformer::convert_to_ollama_generate(&lm, "mymodel", "q", Instant::now(), false);
        assert!(result.get("thinking").is_none());
    }
}
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cargo test --lib handlers::transform 2>&1 | tail -15
```

Expected failure breakdown:
- `chat_response_thinking_in_message_not_content` — **FAILS**: current code merges reasoning into `content`, so `message.thinking` is absent
- `generate_response_thinking_top_level` — **FAILS**: current `convert_to_ollama_generate` never sets `thinking`
- `chat_response_no_thinking_field_when_absent` — **PASSES** under current code (no reasoning → no thinking field already)
- `generate_response_no_thinking_field_when_absent` — **PASSES** under current code (generate never sets thinking)

Confirm at least 2 tests fail before proceeding.

- [ ] **Step 3: Add `extract_reasoning_content` function**

In `src/handlers/transform.rs`:
1. **Delete** the existing `fn extract_chat_content_with_reasoning(...)` method from the `impl ResponseTransformer` block entirely.
2. Add the following three free functions (private, outside the `impl` block — same as other small helpers in this file):

```rust
fn extract_chat_content(lm_response: &Value) -> String {
    lm_response
        .get("choices")
        .and_then(|c| c.as_array()?.first())
        .and_then(|choice| choice.get("message")?.get("content")?.as_str())
        .unwrap_or("")
        .to_string()
}

fn extract_reasoning_content(lm_response: &Value) -> Option<String> {
    let s = lm_response
        .get("choices")
        .and_then(|c| c.as_array()?.first())
        .and_then(|choice| choice.get("message")?.get("reasoning")?.as_str())?;
    if s.is_empty() { None } else { Some(s.to_string()) }
}

fn extract_completion_thinking(lm_response: &Value) -> Option<String> {
    // completions path: check choices[0].reasoning then choices[0].thinking
    let choice = lm_response
        .get("choices")
        .and_then(|c| c.as_array()?.first())?;
    let s = choice.get("reasoning")
        .or_else(|| choice.get("thinking"))
        .and_then(|v| v.as_str())?;
    if s.is_empty() { None } else { Some(s.to_string()) }
}
```

- [ ] **Step 4: Update `convert_to_ollama_chat`**

Replace the call to `extract_chat_content_with_reasoning` with the two new functions, and build the message accordingly:

```rust
let content = extract_chat_content(lm_response);
let thinking = extract_reasoning_content(lm_response);

// ...existing timing code (note: timing now estimates output tokens from `content` only;
// reasoning tokens are excluded, causing a slight under-count when think is active.
// This is acceptable — the proxy has no way to count reasoning tokens separately.)...

let mut ollama_message = json!({
    "role": "assistant",
    "content": content
});

// Add thinking field only when present
if let Some(ref thinking_str) = thinking {
    if let Some(msg_obj) = ollama_message.as_object_mut() {
        msg_obj.insert("thinking".to_string(), json!(thinking_str));
    }
}

// ...existing tool_calls code unchanged...
```

- [ ] **Step 5: Update `convert_to_ollama_generate`**

After extracting `content`, add thinking extraction and include in response:

```rust
let content = Self::extract_completion_content(lm_response);
let thinking = extract_completion_thinking(lm_response);

// ...existing timing code...

let done_reason = extract_finish_reason(lm_response).unwrap_or("stop");
let mut response_obj = json!({
    "model": model_ollama_name,
    "created_at": chrono::Utc::now().to_rfc3339(),
    "response": content,
    "done": true,
    "done_reason": done_reason,
    "context": [],
    "total_duration": timing.total_duration,
    "load_duration": timing.load_duration,
    "prompt_eval_count": timing.prompt_eval_count,
    "prompt_eval_duration": timing.prompt_eval_duration,
    "eval_count": timing.eval_count,
    "eval_duration": timing.eval_duration
});

if let Some(ref t) = thinking {
    if let Some(obj) = response_obj.as_object_mut() {
        obj.insert("thinking".to_string(), json!(t));
    }
}

response_obj
```

- [ ] **Step 6: Run tests**

```bash
cargo test --lib handlers::transform 2>&1 | tail -15
```

Expected: 4 tests pass.

- [ ] **Step 7: Compile check**

```bash
cargo check 2>&1 | tail -10
```

- [ ] **Step 8: Commit**

```bash
git add src/handlers/transform.rs
git commit -m "fix: put reasoning in message.thinking / thinking, not merged into content"
```

---

## Task 8: Fix Reasoning Content in Streaming Responses

**Spec:** Section 2b, 2d

**Files:**
- Modify: `src/streaming/chunks.rs`
- Modify: `src/streaming/sse.rs`

### What & why
Streaming responses have the same problem as non-streaming: `delta.reasoning` is currently appended into the content buffer. This task splits it out into `message.thinking` (for chat chunks) and `thinking` (for generate chunks), and updates the `None`-return guard so reasoning-only chunks are not dropped.

> Both files must be updated together — `sse.rs` calls functions whose signatures change.

### Steps

- [ ] **Step 1: Write the failing tests**

Add a `#[cfg(test)]` module to `src/streaming/chunks.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn choice_with_delta(content: Option<&str>, reasoning: Option<&str>) -> serde_json::Value {
        let mut delta = json!({});
        if let Some(c) = content {
            delta.as_object_mut().unwrap().insert("content".to_string(), json!(c));
        }
        if let Some(r) = reasoning {
            delta.as_object_mut().unwrap().insert("reasoning".to_string(), json!(r));
        }
        json!({ "delta": delta })
    }

    #[test]
    fn reasoning_goes_to_thinking_not_content() {
        let choice = choice_with_delta(Some("answer"), Some("my thinking"));
        let mut state = ChunkProcessingState::default();
        let payload = process_choice_delta(&choice, &mut state).unwrap();
        assert_eq!(payload.content, "answer");
        assert_eq!(payload.thinking, "my thinking");
    }

    #[test]
    fn reasoning_only_chunk_is_not_dropped() {
        // No content, only reasoning — must return Some
        let choice = choice_with_delta(None, Some("reasoning only"));
        let mut state = ChunkProcessingState::default();
        let payload = process_choice_delta(&choice, &mut state);
        assert!(payload.is_some());
        let p = payload.unwrap();
        assert_eq!(p.content, "");
        assert_eq!(p.thinking, "reasoning only");
    }

    #[test]
    fn chat_chunk_thinking_in_message() {
        let chunk = create_ollama_streaming_chunk("m", "hi", true, false, None, "my thought");
        let msg = chunk.get("message").unwrap();
        assert_eq!(msg.get("thinking").and_then(|v| v.as_str()), Some("my thought"));
        assert_eq!(msg.get("content").and_then(|v| v.as_str()), Some("hi"));
    }

    #[test]
    fn chat_chunk_no_thinking_field_when_empty() {
        let chunk = create_ollama_streaming_chunk("m", "hi", true, false, None, "");
        let msg = chunk.get("message").unwrap();
        assert!(msg.get("thinking").is_none());
    }

    #[test]
    fn generate_chunk_thinking_top_level() {
        let chunk = create_ollama_streaming_chunk("m", "response", false, false, None, "thought");
        assert_eq!(chunk.get("thinking").and_then(|v| v.as_str()), Some("thought"));
        // must NOT be nested inside message
        assert!(chunk.get("message").is_none());
    }
}
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cargo test --lib streaming::chunks 2>&1 | tail -15
```

- [ ] **Step 3: Add `thinking` field to `ChoiceDeltaPayload`**

> ⚠️ **Steps 3 and 4 must be applied together.** After adding `thinking` to the struct, the existing `ChoiceDeltaPayload { content, tool_calls_delta }` struct literal in `process_choice_delta` will be missing the field and will not compile. Do not run `cargo check` between Steps 3 and 4 — the codebase is intentionally broken until Step 4 completes.

```rust
pub struct ChoiceDeltaPayload {
    pub content: String,
    pub thinking: String,
    pub tool_calls_delta: Option<Value>,
}
```

- [ ] **Step 4: Update `process_choice_delta`**

Replace the reasoning append-to-content logic with a separate `thinking` buffer:

```rust
pub fn process_choice_delta(
    choice: &Value,
    state: &mut ChunkProcessingState,
) -> Option<ChoiceDeltaPayload> {
    state.update_finish_reason(choice);

    let mut content = String::new();
    let mut thinking = String::new();
    let mut tool_calls_delta: Option<Value> = None;

    if let Some(delta) = choice.get("delta") {
        if let Some(content_value) = delta.get("content") {
            append_stream_content(content_value, &mut content);
        }
        if let Some(reasoning_value) = delta.get("reasoning") {
            append_stream_content(reasoning_value, &mut thinking);
        }
        if let Some(new_tool_calls) = delta.get("tool_calls").and_then(|v| v.as_array())
            && !new_tool_calls.is_empty()
        {
            tool_calls_delta = Some(json!(new_tool_calls));
        }
    }

    if content.is_empty() {
        if let Some(text_value) = choice.get("text") {
            append_stream_content(text_value, &mut content);
        } else if let Some(message_content) = choice
            .get("message")
            .and_then(|m| m.get("content"))
        {
            append_stream_content(message_content, &mut content);
        }
    }

    // Updated guard: return Some if any of the three fields is non-empty
    if content.is_empty() && thinking.is_empty() && tool_calls_delta.is_none() {
        None
    } else {
        Some(ChoiceDeltaPayload { content, thinking, tool_calls_delta })
    }
}
```

- [ ] **Step 5: Update `create_ollama_streaming_chunk` signature**

Add `thinking: &str` as the last parameter:

```rust
pub fn create_ollama_streaming_chunk(
    model_ollama_name: &str,
    content: &str,
    is_chat_endpoint: bool,
    done: bool,
    tool_calls_delta: Option<&Value>,
    thinking: &str,
) -> Value {
```

In the chat branch, add `thinking` to the message object when non-empty:

```rust
if is_chat_endpoint {
    let mut message_obj = json!({
        "role": "assistant",
        "content": content
    });
    if !thinking.is_empty() {
        message_obj.as_object_mut().unwrap().insert("thinking".to_string(), json!(thinking));
    }
    if let Some(tc_delta) = tool_calls_delta
        && let Some(msg_map) = message_obj.as_object_mut()
    {
        msg_map.insert("tool_calls".to_string(), tc_delta.clone());
    }
    json!({
        "model": model_ollama_name,
        "created_at": timestamp,
        "message": message_obj,
        "done": done
    })
} else {
    let mut chunk = json!({
        "model": model_ollama_name,
        "created_at": timestamp,
        "response": content,
        "done": done
    });
    if !thinking.is_empty() {
        chunk.as_object_mut().unwrap().insert("thinking".to_string(), json!(thinking));
    }
    chunk
}
```

- [ ] **Step 6: Update the three internal callers in `chunks.rs`**

`create_error_chunk`, `create_cancellation_chunk`, and `create_final_chunk` all call `create_ollama_streaming_chunk`. Add `""` as the last argument to each:

```rust
// In create_error_chunk:
create_ollama_streaming_chunk(model_ollama_name, "", is_chat_endpoint, true, None, "");

// In create_cancellation_chunk:
create_ollama_streaming_chunk(model_ollama_name, "", is_chat_endpoint, true, None, "");

// In create_final_chunk:
create_ollama_streaming_chunk(params.model_name, "", params.is_chat, true, None, "");
```

- [ ] **Step 7: Update `sse.rs` — 3 call sites for `create_ollama_streaming_chunk` and 3 emit guards**

In each of the three `process_choice_delta` + `create_ollama_streaming_chunk` patterns in `sse.rs`:

Replace this pattern (appears 3 times):
```rust
if let Some(choice) = extract_first_choice(&lm_studio_json_chunk)
    && let Some(delta_payload) = process_choice_delta(choice, &mut chunk_state) {
        content_to_send = delta_payload.content;
        tool_calls_delta = delta_payload.tool_calls_delta;
    }

if !content_to_send.is_empty() || tool_calls_delta.is_some() {
    let ollama_chunk = create_ollama_streaming_chunk(
        &model_clone_for_task,
        &content_to_send,
        is_chat_endpoint,
        false,
        tool_calls_delta.as_ref()
    );
```

With (the `let mut thinking_to_send` declaration must be at the top of the same scope as `content_to_send`, before the `if let Some(choice)` guard that assigns into it):
```rust
let mut content_to_send = String::new();
let mut thinking_to_send = String::new();

if let Some(choice) = extract_first_choice(&lm_studio_json_chunk)
    && let Some(delta_payload) = process_choice_delta(choice, &mut chunk_state) {
        content_to_send = delta_payload.content;
        thinking_to_send = delta_payload.thinking;
        tool_calls_delta = delta_payload.tool_calls_delta;
    }

if !content_to_send.is_empty() || !thinking_to_send.is_empty() || tool_calls_delta.is_some() {
    let ollama_chunk = create_ollama_streaming_chunk(
        &model_clone_for_task,
        &content_to_send,
        is_chat_endpoint,
        false,
        tool_calls_delta.as_ref(),
        &thinking_to_send,
    );
```

The three inner scopes where this change must be applied are:

1. **Primary happy-path block** — the `Ok(lm_studio_json_chunk) => { ... }` match arm that processes normal SSE data lines. Add `let mut thinking_to_send = String::new();` at the top of this block, alongside `let mut content_to_send`.

2. **Chunk-recovery block** — the `if enable_chunk_recovery { ... }` block nested inside the primary arm, which handles partial/malformed chunks. Add `let mut thinking_to_send = String::new();` at the top of this inner block.

3. **End-of-stream recovery block** — the `Ok(None) => { ... }` arm that emits a final chunk when the stream ends without an explicit `[DONE]`. Add `let mut thinking_to_send = String::new();` at the top of this block.

Each variable is scoped to its own block and does not need resetting — it is declared fresh each time.

- [ ] **Step 8: Run tests**

```bash
cargo test --lib streaming::chunks 2>&1 | tail -15
```

Expected: 6 tests pass.

- [ ] **Step 9: Full compile check and test suite**

```bash
cargo check 2>&1 | tail -10
cargo test --lib 2>&1 | tail -20
```

Expected: all tests pass, no errors.

- [ ] **Step 10: Commit**

```bash
git add src/streaming/chunks.rs src/streaming/sse.rs
git commit -m "fix: emit reasoning as thinking field in streaming chunks, not merged into content"
```

---

## Final Verification

- [ ] **Full build**

```bash
cargo build 2>&1 | tail -10
```

Expected: clean build, no warnings (or only pre-existing warnings).

- [ ] **Full test suite**

```bash
cargo test 2>&1
```

Expected: all tests pass.

- [ ] **Spot-check the spec**

Re-read `docs/superpowers/specs/2026-03-20-ollama-api-compliance-design.md` and confirm each section has a corresponding task that was completed.
