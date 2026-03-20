# Ollama API Compliance — Design Spec

**Date:** 2026-03-20
**Scope:** Close the gaps between what the Ollama API specifies and what the proxy currently implements, based on the `api docs/lmstudio_vs_ollama.md` mapping reference.

---

## Background

The proxy translates Ollama API requests into LM Studio API calls (primarily the OpenAI-compatible `/v1/chat/completions` and `/v1/completions` endpoints) and converts responses back to Ollama format. A systematic comparison against the mapping reference revealed the following categories of gaps:

1. Parameters in Ollama requests that aren't forwarded to LM Studio when they could be
2. Parameters in Ollama requests that have no LM Studio equivalent and should be logged at debug level instead of silently dropped
3. Response fields that are populated incorrectly (reasoning content merged into `content` instead of `message.thinking`)
4. A missing root endpoint (`GET /`) required by many Ollama clients
5. Model metadata fields (`size_bytes`, `params_string`) available from LM Studio but not parsed

---

## Section 1: Parameter Mapping

### 1a. New directly-forwarded options

Add to the `DIRECT_MAPPINGS` constant in `map_direct_params` (`src/http/request.rs`):

| Ollama `options.*` key | LM Studio field | Notes |
|---|---|---|
| `presence_penalty` | `presence_penalty` | Direct forward via OpenAI-compat |
| `frequency_penalty` | `frequency_penalty` | Direct forward via OpenAI-compat |
| `min_p` | `min_p` | Direct forward; LM Studio may or may not use it |

The existing `map_penalties` function has three branches for `repeat_penalty`:
1. If neither `frequency_penalty` nor `presence_penalty` is already in `params` → insert as `repeat_penalty`
2. If `presence_penalty` is set but `frequency_penalty` is not → insert `repeat_penalty` value as `frequency_penalty`
3. If `frequency_penalty` is already set → do nothing

After the new direct forwarding of `frequency_penalty` and `presence_penalty` from `options`, case 2 and 3 become unreachable when a client provides those fields explicitly. All three branches are preserved unchanged; the new mappings simply populate `params` earlier, causing the guards to short-circuit correctly.

### 1b. New top-level body parameters

These are not inside `options` and are not currently extracted from the request body. They must be read before `build_lm_studio_request` is called and merged into the outgoing request.

**`think`** (chat and generate endpoints):
- Ollama: `think: bool | "high" | "medium" | "low"` (optional; absent = do not emit `reasoning`)
- Mapping: `true` → `"on"`, `false` → `"off"`, `"high"/"medium"/"low"` pass through unchanged, any other string value passes through unchanged (let LM Studio accept or reject)
- Forward as `reasoning` in the LM Studio request body (best-effort; behaviour depends on LM Studio version)
- When `think` is absent from the Ollama request, no `reasoning` field is emitted

**`logprobs`** (chat and generate endpoints):
- Ollama: `logprobs: bool`
- Forward as-is; let LM Studio accept or ignore

**`top_logprobs`** (chat and generate endpoints):
- Ollama: `top_logprobs: integer`
- Forward as-is

**Passing mechanism:** Introduce a `TopLevelParams<'a>` struct in `src/http/request.rs`:
```rust
pub struct TopLevelParams<'a> {
    pub think: Option<&'a Value>,
    pub logprobs: Option<&'a Value>,
    pub top_logprobs: Option<&'a Value>,
}
```
Add a `top_level: Option<&TopLevelParams>` parameter to `build_lm_studio_request`. When present, merge these fields directly into the request object after the options map. `chat.rs` and `generate.rs` extract these values from the body and construct `TopLevelParams` before calling the builder.

### 1c. Generate-specific: `suffix`

When `suffix` is present in an `/api/generate` request, forward it to `/v1/completions`. This enables fill-in-the-middle completion. When the request routes through chat (vision path — i.e. `images` is also present), `suffix` cannot be forwarded; log it at debug level with "unsupported on vision path: suffix" and proceed without it.

### 1d. Full options audit — unsupported parameter logging

Introduce `log_unsupported_options(options: &Value)` in `src/http/request.rs`. Called at the end of `map_ollama_to_lmstudio_params` when debug logging is enabled. Logs a single line listing any `options.*` keys that were not forwarded.

Exhaustive list of keys that trigger the log:
`num_ctx`, `repeat_last_n`, `tfs_z`, `typical_p`, `mirostat`, `mirostat_tau`, `mirostat_eta`, `penalize_newline`, `num_keep`, `num_batch`, `num_gpu`, `num_thread`, `numa`, `use_mmap`, `use_mlock`, `vocab_only`

Log format (debug level):
```
unsupported options ignored: num_ctx, mirostat, num_gpu
```

No error is returned; the request proceeds normally.

---

## Section 2: Reasoning/Thinking Content in Responses

LM Studio returns reasoning content in `choices[0].message.reasoning` (non-streaming) and `choices[0].delta.reasoning` (streaming). The current proxy merges this into the `content` field as markdown. Ollama clients expect it in a dedicated field.

### 2a. Non-streaming chat (`src/handlers/transform.rs`)

`extract_chat_content_with_reasoning` currently returns a combined string. Split into two functions:
- `extract_chat_content(lm_response) -> String` — extracts only `choices[0].message.content`
- `extract_reasoning_content(lm_response) -> Option<String>` — extracts `choices[0].message.reasoning`, returns `None` if absent or empty

In `convert_to_ollama_chat`, build `message` as:
```json
{
  "role": "assistant",
  "content": "<answer only>",
  "thinking": "<reasoning, if present>"
}
```
The `thinking` key is only included when reasoning is non-empty.

### 2b. Streaming chat (`src/streaming/chunks.rs`)

`process_choice_delta` currently appends `delta.reasoning` into the content buffer. Changes:

1. Add `thinking: String` field to `ChoiceDeltaPayload`; populate it from `delta.reasoning` instead of appending to `content`
2. Update the `None`-return guard: return `Some` when `content` is non-empty **or** `thinking` is non-empty **or** `tool_calls_delta` is `Some`. Reasoning-only chunks (empty content, non-empty thinking) must not be dropped.
3. Update `create_ollama_streaming_chunk` to accept an optional `thinking: &str` parameter; include `message.thinking` in the JSON object when non-empty. All call sites must be updated accordingly.

### 2c. Non-streaming generate (`src/handlers/transform.rs`)

LM Studio's `/v1/completions` response may include reasoning content when active. Check `choices[0].message.reasoning` first (primary field name); fall back to `choices[0].message.thinking` for older LM Studio versions. Extract it and include as a top-level `thinking` field in the Ollama generate response:
```json
{
  "model": "...",
  "response": "...",
  "thinking": "...",
  "done": true,
  ...
}
```

### 2d. Streaming generate (`src/streaming/chunks.rs`)

Same pattern: emit `thinking` as a top-level field on streaming generate chunks when present.

---

## Section 3: New Endpoints & Model Data

### 3a. `GET /` endpoint

Add to `src/server/routes.rs`. Returns plain text `"Ollama is running"` with `Content-Type: text/plain`. This is the standard Ollama health/presence probe used by clients like Open WebUI, LiteLLM proxy, and others.

Handler lives in `src/handlers/ollama/health.rs` alongside the existing `handle_ollama_version` and `handle_health_check` — all three are server-presence/status concerns and the file is already imported at the routes level, so no new module boundary is needed.

### 3b. Accurate model `size_bytes` from LM Studio

Add to `NativeModelData` in `src/model/types.rs`:
```rust
pub size_bytes: Option<u64>,
```
LM Studio's `/api/v1/models` response includes this field. Deserialise it with `#[serde(default)]`.

In `ModelInfo::from_native_data`, copy the value to `ModelInfo`:
```rust
pub size_bytes: Option<u64>,  // transferred from NativeModelData.size_bytes
```

In `calculate_estimated_size`, check `self.size_bytes` first:
```rust
fn calculate_estimated_size(&self) -> u64 {
    if let Some(bytes) = self.size_bytes {
        return bytes;
    }
    // existing name-based heuristic follows ...
}
```
The heuristic is only reached when LM Studio omits `size_bytes`.

### 3c. Accurate model `params_string` from LM Studio

Add to `NativeModelData`:
```rust
pub params_string: Option<String>,
```
LM Studio includes this field (e.g. `"7B"`, `"70B"`).

In `ModelInfo::from_native_data`, copy the value:
```rust
pub params_string: Option<String>,  // transferred from NativeModelData.params_string
```

In `parse_parameters`:
```rust
fn parse_parameters(&self) -> ModelParameters {
    if let Some(ref s) = self.params_string {
        return ModelParameters { size_string: s.clone() };
    }
    // existing name-based inference follows ...
}
```
The existing inference from model id is only reached when LM Studio omits the field.

---

## Files Changed

| File | Changes |
|---|---|
| `src/http/request.rs` | Add `presence_penalty`, `frequency_penalty`, `min_p` to direct mappings; add `TopLevelParams` struct; add `top_level` parameter to `build_lm_studio_request`; add `log_unsupported_options` |
| `src/handlers/ollama/chat.rs` | Extract `think`, `logprobs`, `top_logprobs` from body; construct and pass `TopLevelParams` |
| `src/handlers/ollama/generate.rs` | Extract `think`, `logprobs`, `top_logprobs`, `suffix` from body; construct and pass `TopLevelParams`; forward `suffix` on completions path only |
| `src/handlers/transform.rs` | Split reasoning extraction into `extract_chat_content` + `extract_reasoning_content`; add `thinking` field to chat and generate responses |
| `src/streaming/chunks.rs` | Add `thinking: String` to `ChoiceDeltaPayload`; update `None`-return guard; add `thinking` param to `create_ollama_streaming_chunk`; update all call sites |
| `src/model/types.rs` | Add `size_bytes`, `params_string` to `NativeModelData` and `ModelInfo`; use actual values over heuristics |
| `src/server/routes.rs` | Register `GET /` route |
| `src/handlers/ollama/health.rs` | Add `handle_ollama_root` handler |

---

## Out of Scope

- LM Studio native API routing (`/api/v1/chat`, `/api/v1/models/load`, etc.) — the existing passthrough at `/api/v1/*` handles these
- `logprobs` in response bodies — LM Studio does not return them via OpenAI-compat; pass-through of the request field is sufficient
- `context` token array in generate responses — already returns `[]` which is correct for the proxy use case
- Digest algorithm change (MD5 → SHA256) — no real digest is available from LM Studio; MD5 of name is acceptable as a stable identifier
