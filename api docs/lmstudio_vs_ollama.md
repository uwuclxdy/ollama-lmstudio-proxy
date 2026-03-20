# LMStudio ↔ Ollama native API parameter mapping

LMStudio (v1 native API released with 0.4.0, legacy v0 from 0.3.6) and Ollama expose fundamentally different REST surfaces for the same underlying task: running local LLMs. **LMStudio's v1 API is opinionated and high-level**, collapsing chat, vision, tool use, and reasoning into a single `/api/v1/chat` endpoint. **Ollama's API is lower-level and more granular**, separating text generation (`/api/generate`) from chat (`/api/chat`) and offering extensive runtime tuning through an `options` object with 25+ parameters. This document maps every native endpoint and parameter between the two systems, identifies gaps, and notes semantic differences.

---

## Endpoint-level overview and correspondence

The table below shows every native endpoint from each system and its closest counterpart. Several endpoints have no equivalent on the other side.

| Category | LMStudio Endpoint | Method | Ollama Endpoint | Method | Notes |
|----------|-------------------|--------|-----------------|--------|-------|
| Chat | `/api/v1/chat` | POST | `/api/chat` | POST | Closest functional match; structural differences detailed below |
| Text generation | `/api/v0/completions` (legacy) | POST | `/api/generate` | POST | LMStudio v1 has no raw-completion endpoint |
| Model listing | `/api/v1/models` | GET | `/api/tags` | GET | Different response schemas |
| Running models | *(via `loaded_instances` in `/api/v1/models`)* | — | `/api/ps` | GET | LMStudio inlines this; Ollama has a dedicated endpoint |
| Model info | `/api/v0/models/{model}` | GET | `/api/show` | POST | Ollama uses POST with body; LMStudio uses path param |
| Load model | `/api/v1/models/load` | POST | *(implicit on first request)* | — | Ollama auto-loads; send empty `/api/generate` to preload |
| Unload model | `/api/v1/models/unload` | POST | *(via `keep_alive: 0`)* | — | Ollama unloads by setting `keep_alive` to `"0"` on a request |
| Download model | `/api/v1/models/download` | POST | `/api/pull` | POST | Different progress reporting models |
| Download status | `/api/v1/models/download/status/:job_id` | GET | *(inline in `/api/pull` stream)* | — | Ollama streams progress; no separate status endpoint |
| Embeddings | `/api/v0/embeddings` | POST | `/api/embed` | POST | Ollama also has legacy `/api/embeddings` |
| Create/customize model | *No equivalent* | — | `/api/create` | POST | Ollama-only: build models from Modelfiles |
| Copy model | *No equivalent* | — | `/api/copy` | POST | Ollama-only |
| Push model | *No equivalent* | — | `/api/push` | POST | Ollama-only: upload to registry |
| Delete model | *No equivalent* | — | `/api/delete` | DELETE | Ollama-only |
| Blob management | *No equivalent* | — | `/api/blobs/:digest` | HEAD/POST | Ollama-only |
| Health check | *No equivalent* | — | `GET /` | GET | Returns `"Ollama is running"` |
| Server version | *No equivalent* | — | `/api/version` | GET | Ollama-only |

---

## Chat endpoints: detailed parameter mapping

This is the primary endpoint for both systems. LMStudio's `/api/v1/chat` maps to Ollama's `/api/chat`, but they differ substantially in parameter naming, structure, and capabilities.

### Request parameters

| LMStudio `/api/v1/chat` | Type | Ollama `/api/chat` | Type | Mapping notes |
|--------------------------|------|--------------------|------|---------------|
| `model` | string | `model` | string | **Direct match.** Naming conventions differ (`publisher/model` vs `model:tag`) |
| `input` (string or array) | string \| object[] | `messages` | object[] | **Structural difference.** LMStudio accepts a plain string or typed input objects (`{type: "message", content}`, `{type: "image", data_url}`). Ollama requires an array of `{role, content, images?}` message objects. LMStudio's string shorthand has no Ollama equivalent. |
| `system_prompt` | string | *(first message with `role: "system"`)* | — | **Different mechanism.** LMStudio has a dedicated top-level field. In Ollama, prepend a `{role: "system", content: "..."}` message to the `messages` array. |
| `stream` | boolean (default `false`) | `stream` | boolean (default `true`) | **Opposite defaults.** LMStudio defaults to non-streaming; Ollama defaults to streaming. |
| `temperature` | number | `options.temperature` | float | **Nested in Ollama.** Ollama wraps all sampling params inside an `options` object. Default: Ollama 0.8, LMStudio unspecified. |
| `top_p` | number | `options.top_p` | float | Nested in Ollama. Ollama default: 0.9 |
| `top_k` | integer | `options.top_k` | int | Nested in Ollama. Ollama default: 40 |
| `min_p` | number | `options.min_p` | float | Nested in Ollama. Ollama default: 0.0 |
| `repeat_penalty` | number | `options.repeat_penalty` | float | Nested in Ollama. Ollama default: 1.1 |
| `max_output_tokens` | integer | `options.num_predict` | int | **Different name.** Same concept. Ollama uses -1 for unlimited (default). |
| `context_length` | integer | `options.num_ctx` | int | **Different name.** LMStudio is top-level; Ollama is nested in `options`. Ollama default: 2048. |
| `reasoning` | `"off"\|"low"\|"medium"\|"high"\|"on"` | `think` | boolean or `"high"\|"medium"\|"low"` | **Similar but different values.** LMStudio uses `"off"/"on"` plus granular levels. Ollama uses `true/false` plus string levels for select models (GPT-OSS). |
| `store` | boolean (default `true`) | *No equivalent* | — | **LMStudio-only.** Enables stateful chat with `response_id` tracking. |
| `previous_response_id` | string | *No equivalent* | — | **LMStudio-only.** Appends to a prior conversation by ID. Ollama achieves multi-turn by resending full `messages` array. |
| `integrations` | array | *No equivalent* | — | **LMStudio-only.** MCP server and plugin integrations. |
| *No equivalent* | — | `tools` | object[] | **Ollama-only on native API.** LMStudio v1 native chat does not support custom tool definitions (it uses `integrations` for MCP-based tools instead; custom tools require the OpenAI-compat `/v1/chat/completions`). |
| *No equivalent* | — | `format` | string \| object | **Ollama-only.** JSON mode (`"json"`) or JSON schema for structured outputs. No LMStudio native equivalent. |
| *No equivalent* | — | `keep_alive` | string \| number | **Ollama-only.** Controls how long the model stays loaded after the request. Default: `"5m"`. |
| *No equivalent* | — | `logprobs` | boolean | **Ollama-only.** Return token-level log probabilities. |
| *No equivalent* | — | `top_logprobs` | integer | **Ollama-only.** Number of top logprobs per token position. |
| *No equivalent* | — | `options.seed` | int | **Ollama-only.** Reproducible generation seed. |
| *No equivalent* | — | `options.tfs_z` | float | **Ollama-only.** Tail-free sampling. |
| *No equivalent* | — | `options.typical_p` | float | **Ollama-only.** Typical sampling. |
| *No equivalent* | — | `options.presence_penalty` | float | **Ollama-only** on native chat. LMStudio exposes this only on OpenAI-compat endpoints. |
| *No equivalent* | — | `options.frequency_penalty` | float | **Ollama-only** on native chat. Same caveat as above. |
| *No equivalent* | — | `options.repeat_last_n` | int | **Ollama-only.** Repetition lookback window (default 64). |
| *No equivalent* | — | `options.mirostat` | int | **Ollama-only.** Mirostat sampling mode (0/1/2). |
| *No equivalent* | — | `options.mirostat_tau` | float | **Ollama-only.** Mirostat target entropy. |
| *No equivalent* | — | `options.mirostat_eta` | float | **Ollama-only.** Mirostat learning rate. |
| *No equivalent* | — | `options.penalize_newline` | boolean | **Ollama-only.** |
| *No equivalent* | — | `options.stop` | string[] | **Ollama-only** on native chat. LMStudio exposes `stop` only on OpenAI-compat endpoints. |
| *No equivalent* | — | `options.num_keep` | int | **Ollama-only.** Tokens to keep from prompt when context overflows. |
| *No equivalent* | — | `options.num_batch` | int | **Ollama-only** on native API. LMStudio has `eval_batch_size` on model load, not per-request. |
| *No equivalent* | — | `options.num_gpu` | int | **Ollama-only.** GPU layer offloading count. |
| *No equivalent* | — | `options.num_thread` | int | **Ollama-only.** CPU thread count. |
| *No equivalent* | — | `options.numa` | boolean | **Ollama-only.** NUMA optimization. |

### Response field mapping

| LMStudio `/api/v1/chat` response | Ollama `/api/chat` response | Notes |
|-----------------------------------|-----------------------------|-------|
| `output[].content` (type `"message"`) | `message.content` | LMStudio returns an array of typed output items; Ollama returns a single message object |
| `output[].content` (type `"reasoning"`) | `message.thinking` | Different field names for reasoning/thinking content |
| `output[]` (type `"tool_call"`) | `message.tool_calls[]` | Different structures: LMStudio includes execution `output` inline; Ollama returns only the call, execution is client-side |
| `stats.input_tokens` | `prompt_eval_count` | Same concept, different field name |
| `stats.total_output_tokens` | `eval_count` | Same concept, different field name |
| `stats.reasoning_output_tokens` | *No equivalent* | **LMStudio-only.** Separate count for reasoning tokens |
| `stats.tokens_per_second` | *(computed: `eval_count / (eval_duration / 1e9)`)* | LMStudio provides directly; Ollama requires manual calculation |
| `stats.time_to_first_token_seconds` | *No direct equivalent* | **LMStudio-only** as a discrete field. Ollama provides `prompt_eval_duration` (nanoseconds) which approximates TTFT. |
| `stats.model_load_time_seconds` | `load_duration` | Same concept. LMStudio uses seconds; **Ollama uses nanoseconds** |
| `model_instance_id` | *No equivalent* | **LMStudio-only.** Identifies the specific loaded instance |
| `response_id` | *No equivalent* | **LMStudio-only.** For stateful conversations |
| *No equivalent* | `created_at` | **Ollama-only.** ISO 8601 timestamp |
| *No equivalent* | `done` / `done_reason` | **Ollama-only.** LMStudio uses SSE event types (`chat.end`) instead |
| *No equivalent* | `total_duration` | **Ollama-only.** End-to-end request time in nanoseconds |
| *No equivalent* | `prompt_eval_duration` | **Ollama-only.** Prompt processing time in nanoseconds |
| *No equivalent* | `eval_duration` | **Ollama-only.** Generation time in nanoseconds |

### Streaming format differences

LMStudio uses **Server-Sent Events (SSE)** with typed event names (`chat.start`, `message.delta`, `chat.end`, etc.), providing granular lifecycle events including model loading progress, prompt processing progress, and reasoning phases. Ollama uses **newline-delimited JSON (NDJSON)** where each line is a JSON object with `done: false` during streaming and `done: true` on the final object which carries aggregated statistics.

| LMStudio SSE event | Ollama NDJSON equivalent | Notes |
|---------------------|--------------------------|-------|
| `chat.start` | *(first NDJSON line)* | No explicit start event in Ollama |
| `model_load.start` / `model_load.progress` / `model_load.end` | *No equivalent* | **LMStudio-only.** Ollama loads silently; load time appears in final stats |
| `prompt_processing.start` / `prompt_processing.progress` / `prompt_processing.end` | *No equivalent* | **LMStudio-only.** |
| `message.delta` with `content` | Each NDJSON line with `message.content` fragment | Functionally equivalent content streaming |
| `reasoning.delta` with `content` | Each NDJSON line with `message.thinking` fragment | Thinking content streamed differently |
| `tool_call.start` / `tool_call.arguments` / `tool_call.success` / `tool_call.failure` | Single NDJSON line with `message.tool_calls` | LMStudio provides granular tool lifecycle; Ollama emits tool calls as complete objects |
| `chat.end` with full `result` | Final NDJSON line with `done: true` and stats | Both provide aggregated data at stream end |
| `error` event | HTTP error response or inline error | Different error propagation models |

---

## Text generation endpoints

LMStudio's legacy `/api/v0/completions` maps to Ollama's `/api/generate`. LMStudio v1 has **no dedicated text-completion endpoint** — all inference goes through `/api/v1/chat`.

### Request parameters

| LMStudio `/api/v0/completions` | Type | Ollama `/api/generate` | Type | Notes |
|-------------------------------|------|------------------------|------|-------|
| `model` | string | `model` | string | Direct match |
| `prompt` | string | `prompt` | string | Direct match |
| `temperature` | number | `options.temperature` | float | Nested in Ollama |
| `max_tokens` | integer | `options.num_predict` | int | Different name |
| `stream` | boolean | `stream` | boolean | LMStudio defaults false; Ollama defaults true |
| `stop` | string | `options.stop` | string[] | LMStudio accepts single string; Ollama accepts array |
| *No equivalent* | — | `suffix` | string | **Ollama-only.** Fill-in-the-middle completion |
| *No equivalent* | — | `images` | string[] | **Ollama-only.** Base64 images for vision models |
| *No equivalent* | — | `format` | string \| object | **Ollama-only.** JSON/schema structured output |
| *No equivalent* | — | `system` | string | **Ollama-only.** System message override |
| *No equivalent* | — | `template` | string | **Ollama-only.** Prompt template override |
| *No equivalent* | — | `raw` | boolean | **Ollama-only.** Skip all formatting/templating |
| *No equivalent* | — | `context` | int[] | **Ollama-only** (deprecated). Token context for conversation continuity |
| *No equivalent* | — | `think` | boolean \| string | **Ollama-only.** Thinking/reasoning output |
| *No equivalent* | — | `keep_alive` | string \| number | **Ollama-only.** Model keep-alive duration |
| *No equivalent* | — | `options.*` | various | **Ollama-only.** All 25+ options (top_k, top_p, min_p, seed, mirostat, etc.) |
| *No equivalent* | — | `logprobs` / `top_logprobs` | boolean / int | **Ollama-only.** Log probability output |

### Response field mapping

| LMStudio `/api/v0/completions` | Ollama `/api/generate` | Notes |
|-------------------------------|------------------------|-------|
| `choices[0].text` | `response` | Different structure: LMStudio uses OpenAI-style `choices` array |
| `usage.prompt_tokens` | `prompt_eval_count` | Same concept |
| `usage.completion_tokens` | `eval_count` | Same concept |
| `stats.tokens_per_second` | *(computed from `eval_count/eval_duration`)* | LMStudio provides directly |
| `stats.time_to_first_token` | *(approx. `prompt_eval_duration`)* | Different granularity |
| `stats.generation_time` | `eval_duration` | LMStudio in seconds; **Ollama in nanoseconds** |
| `model_info.arch` | *(via `/api/show`)* | LMStudio inlines model metadata in response |
| `model_info.quant` | *(via `/api/show`)* | LMStudio inlines; Ollama requires separate call |
| `model_info.context_length` | *(via `/api/show`)* | Same |
| `runtime.name` / `runtime.version` | *(via `/api/version`)* | LMStudio inlines runtime info |
| *No equivalent* | `thinking` | **Ollama-only.** Reasoning output |
| *No equivalent* | `context` | **Ollama-only** (deprecated). Token array for conversation |
| *No equivalent* | `total_duration` | **Ollama-only.** Full request time in ns |
| *No equivalent* | `load_duration` | **Ollama-only.** Model load time in ns |
| *No equivalent* | `done_reason` | **Ollama-only.** (`"stop"`, `"length"`) |

---

## Model listing and info endpoints

### List models

| LMStudio `/api/v1/models` (GET) | Ollama `/api/tags` (GET) | Notes |
|----------------------------------|--------------------------|-------|
| `models[].key` | `models[].name` / `models[].model` | Primary model identifier |
| `models[].type` (`"llm"` \| `"embedding"`) | *No equivalent* | **LMStudio-only.** Ollama does not distinguish model types in listing |
| `models[].publisher` | *No equivalent* | **LMStudio-only.** |
| `models[].display_name` | *No equivalent* | **LMStudio-only.** Human-readable name |
| `models[].architecture` | `models[].details.family` | Similar concept; different naming |
| `models[].quantization.name` | `models[].details.quantization_level` | Same concept |
| `models[].quantization.bits_per_weight` | *No equivalent* | **LMStudio-only.** |
| `models[].size_bytes` | `models[].size` | Direct match (both in bytes) |
| `models[].params_string` | `models[].details.parameter_size` | Same concept (e.g., `"7B"`) |
| `models[].format` (`"gguf"` \| `"mlx"`) | `models[].details.format` | Direct match (Ollama is always `"gguf"`) |
| `models[].max_context_length` | *(via `/api/show` → `model_info`)* | LMStudio includes in listing; Ollama requires separate call |
| `models[].capabilities.vision` | *No equivalent in listing* | **LMStudio-only** in model list response |
| `models[].capabilities.trained_for_tool_use` | *No equivalent* | **LMStudio-only.** |
| `models[].description` | *No equivalent in listing* | **LMStudio-only.** |
| `models[].loaded_instances[]` | *(via `/api/ps`)* | LMStudio inlines loaded status; Ollama uses separate `/api/ps` endpoint |
| *No equivalent* | `models[].modified_at` | **Ollama-only.** Last modification timestamp |
| *No equivalent* | `models[].digest` | **Ollama-only.** SHA256 digest |
| *No equivalent* | `models[].details.families` | **Ollama-only.** Array of model families |

### Running/loaded models

| LMStudio (inline in `/api/v1/models`) | Ollama `/api/ps` (GET) | Notes |
|----------------------------------------|------------------------|-------|
| `models[].loaded_instances[].id` | `models[].model` | LMStudio uses instance IDs; Ollama uses model names |
| `models[].loaded_instances[].config.context_length` | *No equivalent in `/api/ps`* | Available via `/api/show` |
| `models[].loaded_instances[].config.flash_attention` | *No equivalent* | **LMStudio-only.** |
| `models[].loaded_instances[].config.eval_batch_size` | *No equivalent* | **LMStudio-only.** |
| *No equivalent* | `models[].expires_at` | **Ollama-only.** When model will be auto-unloaded |
| *No equivalent* | `models[].size_vram` | **Ollama-only.** VRAM usage |

### Model details

| LMStudio `/api/v0/models/{model}` (GET) | Ollama `/api/show` (POST) | Notes |
|------------------------------------------|---------------------------|-------|
| Path parameter: `model` | Body parameter: `model` | **Different HTTP methods.** LMStudio uses GET with path param; Ollama uses POST with JSON body |
| Response: single model object | Response: `modelfile`, `parameters`, `template`, `details`, `model_info`, `license`, `system` | **Ollama returns far more detail** including the full Modelfile, template, license, and detailed model_info metadata |
| *No equivalent* | `verbose` parameter | **Ollama-only.** Request verbose output |

---

## Model loading, unloading, and lifecycle

This is where the two systems diverge most. **LMStudio uses explicit load/unload** operations. **Ollama uses implicit loading** with TTL-based unloading.

### Loading a model

| LMStudio `/api/v1/models/load` (POST) | Ollama equivalent | Notes |
|----------------------------------------|-------------------|-------|
| `model` (required) | `model` on any inference request | Ollama auto-loads on first inference request. To preload without inference, send `POST /api/generate` with `model` and empty `prompt`. |
| `context_length` | `options.num_ctx` | Set per-request in Ollama, not at load time |
| `eval_batch_size` | `options.num_batch` | Set per-request in Ollama |
| `flash_attention` | *No equivalent* | **LMStudio-only.** |
| `num_experts` | *No equivalent* | **LMStudio-only.** MoE expert count. |
| `offload_kv_cache_to_gpu` | *No equivalent* | **LMStudio-only.** Ollama has `options.num_gpu` for layer offloading but not KV cache control. |
| `echo_load_config` | *No equivalent* | **LMStudio-only.** |
| Response: `instance_id`, `load_time_seconds`, `status` | Response: standard generate response with `load_duration` | Different feedback mechanisms |

### Unloading a model

| LMStudio `/api/v1/models/unload` (POST) | Ollama equivalent | Notes |
|------------------------------------------|-------------------|-------|
| `instance_id` (required) | Send any request with `keep_alive: "0"` or `keep_alive: 0` | **Fundamentally different.** LMStudio unloads by instance ID. Ollama unloads by setting keep_alive to zero on the next request to that model. |

### Downloading a model

| LMStudio `/api/v1/models/download` (POST) | Ollama `/api/pull` (POST) | Notes |
|--------------------------------------------|---------------------------|-------|
| `model` | `model` | Direct match |
| `quantization` | *No equivalent* | **LMStudio-only.** Ollama embeds quantization in the model tag (e.g., `model:q4_K_M`) |
| *No equivalent* | `insecure` | **Ollama-only.** Allow insecure connections |
| *No equivalent* | `stream` (default `true`) | **Ollama-only.** LMStudio uses a separate status-polling endpoint instead |
| Response: `job_id`, `status`, `total_size_bytes` | Streaming: `status`, `digest`, `total`, `completed` | **Different progress models.** LMStudio returns a `job_id` and you poll `/api/v1/models/download/status/:job_id`. Ollama streams NDJSON progress inline. |

---

## Embeddings endpoints

| LMStudio `/api/v0/embeddings` (POST) | Ollama `/api/embed` (POST) | Notes |
|---------------------------------------|----------------------------|-------|
| `model` | `model` | Direct match |
| `input` (string) | `input` (string or string[]) | **Ollama supports batch input.** LMStudio accepts only a single string. |
| *No equivalent* | `truncate` (boolean, default `true`) | **Ollama-only.** Controls whether to truncate inputs exceeding context window |
| *No equivalent* | `options` | **Ollama-only.** Runtime model parameters |
| *No equivalent* | `keep_alive` | **Ollama-only.** Model TTL |

### Embeddings response mapping

| LMStudio response | Ollama response | Notes |
|-------------------|-----------------|-------|
| `data[].embedding` (float[]) | `embeddings` (float[][]) | **Different structure.** LMStudio wraps in OpenAI-style `data` array. Ollama returns a flat array of arrays. |
| `data[].index` | *(implicit array index)* | **LMStudio-only.** Ollama uses array position |
| `model` | `model` | Direct match |
| `usage.prompt_tokens` | `prompt_eval_count` | Same concept |
| *No equivalent* | `total_duration` | **Ollama-only.** |
| *No equivalent* | `load_duration` | **Ollama-only.** |

---

## Ollama-only endpoints with no LMStudio equivalent

Several Ollama endpoints have **no counterpart** in LMStudio's native API:

| Ollama endpoint | Method | Purpose | Why no LMStudio equivalent |
|-----------------|--------|---------|---------------------------|
| `/api/create` | POST | Create custom model from Modelfile, base model, or safetensors | LMStudio manages models through its GUI; no programmatic model creation API |
| `/api/copy` | POST | Duplicate a model under a new name | Not exposed in LMStudio API |
| `/api/push` | POST | Upload model to Ollama registry | LMStudio has no public model registry |
| `/api/delete` | DELETE | Delete a local model | Not exposed in LMStudio native API |
| `/api/blobs/:digest` | HEAD/POST | Check/upload binary blobs | Low-level Ollama internals |
| `/api/version` | GET | Return server version | Not exposed in LMStudio API |
| `GET /` | GET | Health check (`"Ollama is running"`) | Not exposed in LMStudio API |

---

## LMStudio-only features with no Ollama equivalent

| Feature / Parameter | Endpoint | Why Ollama lacks it |
|---------------------|----------|---------------------|
| `integrations` (MCP servers, plugins) | `/api/v1/chat` | Ollama has no built-in MCP/plugin system; tool calling is the closest analog |
| `store` / `previous_response_id` (stateful chat) | `/api/v1/chat` | Ollama is stateless — clients must resend full message history each request |
| `response_id` tracking | `/api/v1/chat` response | Same as above; no server-side conversation state |
| `model_instance_id` | `/api/v1/chat`, `/api/v1/models` | Ollama doesn't expose instance-level identifiers; models are addressed by name |
| Granular SSE lifecycle events | `/api/v1/chat` streaming | Model load progress, prompt processing progress events have no Ollama equivalent |
| `echo_load_config` | `/api/v1/models/load` | Ollama has no explicit load endpoint |
| `capabilities` (vision, tool_use flags) | `/api/v1/models` response | Ollama does not advertise model capabilities in its listing |
| `display_name`, `description`, `publisher` | `/api/v1/models` response | Ollama model metadata is more minimal |
| `bits_per_weight` | `/api/v1/models` response | Ollama only provides quantization level name |

---

## Architectural differences that affect mapping

Beyond individual parameter mapping, several **design-level differences** change how you translate between the two APIs.

**Sampling parameter location.** LMStudio places sampling parameters (`temperature`, `top_p`, `top_k`, etc.) as **top-level request fields**. Ollama nests them inside an **`options` object**. When translating, every LMStudio sampling parameter must be wrapped in `options: { ... }` for Ollama, and vice versa.

**Streaming protocol.** LMStudio uses **SSE** (`text/event-stream`) with named event types. Ollama uses **NDJSON** (`application/x-ndjson`) with `done` boolean flags. Client code must handle fundamentally different parsing logic. LMStudio's SSE provides richer lifecycle information (load progress, prompt processing) that Ollama does not expose during streaming.

**Model lifecycle philosophy.** LMStudio treats model loading as an **explicit, managed operation** — you load a specific model with specific configuration and get back an `instance_id` you use to reference it. Ollama treats model loading as **implicit and ephemeral** — models load automatically on first request and unload after a configurable TTL (`keep_alive`). This means LMStudio code that explicitly loads/unloads models needs to be restructured for Ollama: remove load calls, add `keep_alive` to inference requests, and use empty-prompt generate requests for preloading.

**Conversation state.** LMStudio supports **server-side conversation state** via `store`/`previous_response_id`, letting you continue conversations without resending history. Ollama is **fully stateless** — the client must maintain and resend the complete `messages` array on every request. The deprecated `context` token array in `/api/generate` was Ollama's older approach to this, but it is not recommended.

**Time units.** LMStudio reports durations in **seconds** (float). Ollama reports durations in **nanoseconds** (integer). Divide Ollama values by `1e9` to get seconds, or multiply LMStudio values by `1e9` to get nanoseconds.

---

## Quick-reference conversion cheat sheet

For developers porting between the two APIs, this table captures the most common parameter translations:

| Concept | LMStudio param | Ollama param | Transform needed |
|---------|---------------|--------------|-----------------|
| Model identifier | `model` | `model` | Adjust naming convention |
| Chat messages | `input` (string or typed array) | `messages` (role/content array) | Restructure message format |
| System prompt | `system_prompt` | First `messages` entry with `role: "system"`, or `system` on `/api/generate` | Move to messages array |
| Max tokens | `max_output_tokens` | `options.num_predict` | Rename + nest in options |
| Context window | `context_length` | `options.num_ctx` | Rename + nest in options |
| Temperature | `temperature` | `options.temperature` | Nest in options |
| Top-p | `top_p` | `options.top_p` | Nest in options |
| Top-k | `top_k` | `options.top_k` | Nest in options |
| Min-p | `min_p` | `options.min_p` | Nest in options |
| Repeat penalty | `repeat_penalty` | `options.repeat_penalty` | Nest in options |
| Reasoning toggle | `reasoning: "on"/"off"` | `think: true/false` | Map string to boolean |
| Streaming | `stream` (default false) | `stream` (default true) | Flip default assumption |
| Load model | `POST /api/v1/models/load` | `POST /api/generate` with empty prompt | Different mechanism entirely |
| Unload model | `POST /api/v1/models/unload` with `instance_id` | Any request with `keep_alive: "0"` | Different mechanism entirely |
| Download model | `POST /api/v1/models/download` → poll status | `POST /api/pull` with streaming | Poll vs. stream |
| List models | `GET /api/v1/models` | `GET /api/tags` | Different response schema |
| Embeddings | `POST /api/v0/embeddings` | `POST /api/embed` | Ollama supports batch; different response structure |

## Conclusion

The mapping between LMStudio and Ollama native APIs is **functional but not one-to-one**. The core inference parameters (temperature, top_p, top_k, etc.) translate directly, requiring only structural relocation into Ollama's `options` object. The fundamental gaps are architectural: LMStudio's stateful conversations, explicit model lifecycle management, MCP integrations, and rich SSE streaming have no Ollama equivalents. Conversely, Ollama's extensive low-level tuning options (mirostat, tail-free sampling, NUMA control, thread counts), structured output support, log probabilities, and model creation/registry APIs have no LMStudio native counterparts. A translation layer between the two must handle not just parameter renaming, but protocol differences (SSE vs. NDJSON), time unit conversion (seconds vs. nanoseconds), and fundamentally different approaches to model lifecycle and conversation state.
