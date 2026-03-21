# LMStudio ↔ Ollama OpenAI-compatible API mapping

**Both LMStudio and Ollama expose an OpenAI-compatible API layer at `/v1/*`, but they differ meaningfully in which parameters they actually support, which they silently ignore, and what proprietary extensions each adds.** LMStudio (port **1234**) provides broader extensions like `top_k`, `repeat_penalty`, speculative decoding, and MCP integration, while Ollama (port **11434**) hews closer to the OpenAI spec with features like `logprobs`, `suffix`, `n`, and image generation that LMStudio lacks. Neither system supports the full OpenAI parameter surface — both silently drop unrecognized fields rather than returning errors.

This document covers every endpoint, every parameter, and every behavioral difference between the two systems' OpenAI-compatibility layers, based on official documentation and source-code analysis as of March 2026.

---

## General configuration and server differences

| Property | LMStudio | Ollama |
|---|---|---|
| **Default port** | `1234` | `11434` |
| **Base URL** | `http://localhost:1234/v1` | `http://localhost:11434/v1` |
| **Authentication** | Opt-in; `Authorization: Bearer <token>` when enabled | Not required locally; `api_key` field ignored (convention: set to `"ollama"`) |
| **Model naming** | `publisher/model-name` (e.g., `ibm/granite-4-micro`); variant with `@`: `google/gemma-3-12b@q3_k_l` | `model:tag` (e.g., `llama3.2`, `qwen3:8b`); tag defaults to `latest`; namespace optional: `namespace/model:tag` |
| **Model aliasing** | Not needed; uses publisher/model format directly | `ollama cp llama3.2 gpt-3.5-turbo` to alias for OpenAI-expecting tools |
| **JIT model loading** | Yes — auto-loads models on demand when JIT is ON; `ttl` parameter controls auto-unload | No equivalent via OpenAI compat; `keep_alive` only available in native API |
| **Context size override** | Via `context_length` parameter (native API) or model settings | Cannot set via OpenAI compat layer; requires `Modelfile` with `PARAMETER num_ctx` |
| **`system_fingerprint`** | Not returned in OpenAI-compat responses | Fixed value `"fp_ollama"` in all responses |
| **Unsupported param behavior** | Silently ignored | Silently ignored (Go's `json.Unmarshal` drops unknown fields) |
| **Additional compat layers** | Anthropic-compatible (`/v1/messages`), native v1 (`/api/v1/*`), legacy v0 (`/api/v0/*`) | Native API (`/api/*`) only |

---

## Endpoint availability at a glance

| Endpoint | LMStudio | Ollama | Notes |
|---|---|---|---|
| `POST /v1/chat/completions` | ✅ | ✅ | Core endpoint, both fully support |
| `POST /v1/completions` | ✅ (labeled "legacy") | ✅ | Text completions |
| `GET /v1/models` | ✅ | ✅ | List models |
| `GET /v1/models/{model}` | ⚠️ Not confirmed on `/v1/`; available at `/api/v0/models/{model}` | ✅ | Ollama returns single model object |
| `POST /v1/embeddings` | ✅ | ✅ | Embedding generation |
| `POST /v1/responses` | ✅ | ✅ (added v0.13.3) | OpenAI Responses API format |
| `POST /v1/images/generations` | ❌ | ✅ (experimental) | Ollama only |

---

## POST /v1/chat/completions — parameter mapping

This is the primary endpoint for both systems. The table below covers every parameter from the OpenAI spec plus all extensions.

### Request parameters

| Parameter | Type | OpenAI Spec | LMStudio | Ollama | Notes |
|---|---|---|---|---|---|
| `model` | string | ✅ Required | ✅ Required | ✅ Required | Different naming conventions (see above) |
| `messages` | array[object] | ✅ Required | ✅ Required | ✅ Required | Both support `system`, `user`, `assistant`, `tool` roles |
| `messages[].content` (multimodal) | string \| array | ✅ | ✅ (VLM models) | ✅ (vision models) | Both support `image_url` content parts with base64 data URIs |
| `temperature` | number | ✅ (default 1.0) | ✅ (model default) | ✅ (default 1.0) | Range 0–2 |
| `top_p` | number | ✅ (default 1.0) | ✅ (model default) | ✅ (default 1.0) | Nucleus sampling |
| `n` | integer | ✅ (default 1) | ❌ Not supported | ✅ (recently added) | LMStudio always returns 1 completion |
| `stream` | boolean | ✅ (default false) | ✅ (default false) | ✅ (default false) | SSE format, terminated with `data: [DONE]` |
| `stream_options` | object | ✅ | ✅ | ✅ | `{include_usage: true}` for usage in streaming |
| `stop` | string \| array[string] | ✅ | ✅ | ✅ | Ollama converts string to single-element array internally |
| `max_tokens` | integer | ✅ | ✅ (use -1 for unlimited) | ✅ (maps to `num_predict`) | |
| `max_completion_tokens` | integer | ✅ | ❌ Ignored | ❌ Ignored | OpenAI's newer alias; neither supports |
| `presence_penalty` | number | ✅ (default 0) | ✅ (default 0) | ✅ (default 0) | Range -2.0 to 2.0 |
| `frequency_penalty` | number | ✅ (default 0) | ✅ (default 0) | ✅ (default 0) | Range -2.0 to 2.0 |
| `logit_bias` | object | ✅ | ✅ | ✅ (recently added) | Token ID → bias mapping |
| `logprobs` | boolean | ✅ | ❌ Not supported | ✅ | **Major difference**: LMStudio returns `null`; only supports logprobs on `/v1/responses` |
| `top_logprobs` | integer | ✅ (0–20) | ❌ Not supported | ✅ | Requires `logprobs=true` in Ollama |
| `seed` | integer | ✅ | ✅ | ✅ | Reproducible outputs |
| `response_format` | object | ✅ | ✅ (`json_schema`, `text` only) | ✅ (`json_object`, `json_schema`) | **LMStudio rejects `json_object`**; Ollama supports both |
| `tools` | array[object] | ✅ | ✅ | ✅ | Function tool definitions |
| `tool_choice` | string \| object | ✅ | ✅ (`auto`, `none`, `required`) | ✅ (recently added) | |
| `parallel_tool_calls` | boolean | ✅ | ❌ Ignored | ❌ Ignored | Neither supports |
| `reasoning_effort` | string | ✅ | ❌ Ignored on this endpoint | ✅ (`high`, `medium`, `low`, `none`) | LMStudio only works on `/v1/responses` |
| `reasoning` | object | ✅ | ❌ Ignored on this endpoint | ✅ (`reasoning.effort`) | Same limitation as above |
| `user` | string | ✅ | ❌ Not listed | ✅ (recently added) | Passed through in Ollama |
| `service_tier` | string | ✅ | ❌ Ignored | ❌ Ignored | Cloud-only OpenAI parameter |
| `metadata` | object | ✅ | ❌ Ignored | ❌ Ignored | |
| `store` | boolean | ✅ (OpenAI) | ❌ Ignored | ❌ Ignored | OpenAI's conversation storage flag |
| `modalities` | array | ✅ | ❌ Ignored | ❌ Ignored | Audio/text modality selection |
| `audio` | object | ✅ | ❌ Ignored | ❌ Ignored | Audio generation parameters |
| `prediction` | object | ✅ | ❌ Ignored | ❌ Ignored | Predicted output for caching |
| **`top_k`** | integer | ❌ Not in spec | **✅ Extension** | ❌ Not supported | **LMStudio-only extension** |
| **`repeat_penalty`** | number | ❌ Not in spec | **✅ Extension** | ❌ Not supported | **LMStudio-only extension** (1 = no penalty) |
| **`ttl`** | integer | ❌ Not in spec | **✅ Extension** | ❌ Not available | Idle TTL in seconds for JIT-loaded models |
| **`draft_model`** | string | ❌ Not in spec | **✅ Extension** | ❌ Not available | Speculative decoding model identifier |

### Response fields

| Field | OpenAI Spec | LMStudio | Ollama | Notes |
|---|---|---|---|---|
| `id` | ✅ `chatcmpl-xxx` | ✅ | ✅ | Both generate unique IDs |
| `object` | ✅ `chat.completion` | ✅ | ✅ | |
| `created` | ✅ Unix timestamp | ✅ | ✅ | |
| `model` | ✅ | ✅ | ✅ | Returns the model identifier used |
| `system_fingerprint` | ✅ | ❌ Not returned | ✅ `"fp_ollama"` (fixed) | Ollama always returns same value |
| `choices[].index` | ✅ | ✅ | ✅ | |
| `choices[].message.role` | ✅ | ✅ | ✅ | Always `"assistant"` |
| `choices[].message.content` | ✅ | ✅ | ✅ | |
| `choices[].message.tool_calls` | ✅ | ✅ | ✅ | |
| `choices[].message.reasoning` | ❌ Not in base spec | ✅ (reasoning models) | ✅ (thinking models) | Both use `.reasoning` field for chain-of-thought |
| `choices[].finish_reason` | ✅ | ✅ (`stop`, `length`, `tool_calls`) | ✅ (`stop`, `length`, `tool_calls`) | |
| `choices[].logprobs` | ✅ | ❌ Always `null` | ✅ Token-level logprobs | |
| `usage.prompt_tokens` | ✅ | ✅ | ✅ (from `PromptEvalCount`) | |
| `usage.completion_tokens` | ✅ | ✅ | ✅ (from `EvalCount`) | |
| `usage.total_tokens` | ✅ | ✅ | ✅ | |
| **`stats`** | ❌ | **✅ LMStudio extension** | ❌ | Object with `tokens_per_second`, `time_to_first_token`, `generation_time`, `stop_reason`; speculative decoding stats when using `draft_model` |
| **`model_info`** | ❌ | **✅ LMStudio extension** | ❌ | Object with `arch`, `quant`, `format`, `context_length` |
| **`runtime`** | ❌ | **✅ LMStudio extension** | ❌ | Object with engine `name`, `version`, `supported_formats` |

---

## POST /v1/completions — parameter mapping

The legacy text completions endpoint. LMStudio labels it "legacy" but still supports it. No chat template is applied — raw prompt in, text out.

### Request parameters

| Parameter | Type | OpenAI Spec | LMStudio | Ollama | Notes |
|---|---|---|---|---|---|
| `model` | string | ✅ Required | ✅ Required | ✅ Required | |
| `prompt` | string \| array | ✅ Required | ✅ (string only) | ✅ (string only) | Neither supports array-of-tokens input |
| `suffix` | string | ✅ | ❌ Not supported | ✅ (maps to Ollama's `Suffix`) | Fill-in-the-middle; **Ollama-only** |
| `temperature` | number | ✅ | ✅ | ✅ | |
| `top_p` | number | ✅ | ✅ | ✅ | |
| `n` | integer | ✅ | ❌ Not supported | ✅ (recently added) | |
| `stream` | boolean | ✅ | ✅ | ✅ | |
| `stream_options` | object | ✅ | ✅ | ✅ | |
| `max_tokens` | integer | ✅ | ✅ | ✅ (→ `num_predict`) | |
| `stop` | string \| array | ✅ | ✅ | ✅ | |
| `presence_penalty` | number | ✅ | ✅ | ✅ | |
| `frequency_penalty` | number | ✅ | ✅ | ✅ | |
| `logit_bias` | object | ✅ | ✅ | ✅ (recently added) | |
| `logprobs` | integer | ✅ | ❌ Not supported | ✅ (int → `Logprobs=true, TopLogprobs=N`) | In completions API, OpenAI spec uses integer not boolean |
| `echo` | boolean | ✅ | ❌ Not supported | ✅ (recently added) | Echo prompt back with completion |
| `best_of` | integer | ✅ (deprecated) | ❌ Not supported | ✅ (recently added) | Generate N, return best |
| `seed` | integer | ✅ | ✅ | ✅ | |
| `user` | string | ✅ | ❌ Not listed | ✅ (recently added) | |
| **`top_k`** | integer | ❌ Not in spec | **✅ Extension** | ❌ | LMStudio-only |
| **`repeat_penalty`** | number | ❌ Not in spec | **✅ Extension** | ❌ | LMStudio-only |

### Response fields

| Field | LMStudio | Ollama | Notes |
|---|---|---|---|
| `id` | ✅ `cmpl-xxx` | ✅ | |
| `object` | ✅ `text_completion` | ✅ `text_completion` | |
| `created` | ✅ | ✅ | |
| `model` | ✅ | ✅ | |
| `system_fingerprint` | ❌ | ✅ `"fp_ollama"` | |
| `choices[].text` | ✅ | ✅ | |
| `choices[].index` | ✅ | ✅ | |
| `choices[].finish_reason` | ✅ (`stop`, `length`) | ✅ (`stop`, `length`) | |
| `choices[].logprobs` | ❌ `null` | ✅ | Token-level log probabilities |
| `usage` | ✅ | ✅ | Standard `prompt_tokens`, `completion_tokens`, `total_tokens` |
| **`stats`** | **✅ Extension** | ❌ | Same performance stats as chat endpoint |
| **`model_info`** | **✅ Extension** | ❌ | |
| **`runtime`** | **✅ Extension** | ❌ | |

---

## GET /v1/models — response mapping

A straightforward listing endpoint. Both return the standard OpenAI format, but the data inside differs because of each system's model management approach.

### Response fields

| Field | OpenAI Spec | LMStudio | Ollama | Notes |
|---|---|---|---|---|
| `object` | ✅ `"list"` | ✅ | ✅ | |
| `data[].id` | ✅ | ✅ (e.g., `ibm/granite-4-micro`) | ✅ (e.g., `llama3.2:latest`) | Naming convention difference |
| `data[].object` | ✅ `"model"` | ✅ | ✅ | |
| `data[].created` | ✅ Unix timestamp | ✅ (may be `0`) | ✅ (model's **last modified** time, not creation) | Semantic difference in `created` |
| `data[].owned_by` | ✅ | ✅ (publisher name) | ✅ (Ollama namespace, defaults to `"library"`) | |

**Behavioral difference**: With JIT enabled, LMStudio's `/v1/models` returns **all downloaded models** (loaded or not). With JIT disabled, it returns only **currently loaded** models. Ollama always returns all locally available models (equivalent to `ollama list`).

---

## GET /v1/models/{model} — single model retrieval

| Aspect | LMStudio | Ollama |
|---|---|---|
| **Availability** | ⚠️ Not confirmed on `/v1/` path; use `/api/v0/models/{model}` instead | ✅ Supported |
| **Response format** | Via v0: includes `type`, `publisher`, `arch`, `quantization`, `state`, `max_context_length` | Same fields as a single `/v1/models` entry: `id`, `object`, `created`, `owned_by` |

Ollama's response for this endpoint is minimal — identical to a single element from the `/v1/models` list. LMStudio's native v0 endpoint provides significantly richer metadata.

---

## POST /v1/embeddings — parameter mapping

### Request parameters

| Parameter | Type | OpenAI Spec | LMStudio | Ollama | Notes |
|---|---|---|---|---|---|
| `model` | string | ✅ Required | ✅ Required | ✅ Required | Must be an embedding model |
| `input` | string \| array | ✅ Required | ✅ (string or array[string]) | ✅ (string, array[string], array[int], array[array[int]]) | Ollama also accepts token arrays |
| `encoding_format` | string | ✅ (`float`, `base64`) | ❌ Ignored (always returns float) | ✅ (`float` default, `base64` with little-endian float32) | **Key difference**: LMStudio always returns float arrays regardless |
| `dimensions` | integer | ✅ | ❌ Not supported | ✅ | Truncation to desired dimensions; **Ollama-only** |
| `user` | string | ✅ | ❌ Not listed | ✅ | |

### Response fields

| Field | LMStudio | Ollama | Notes |
|---|---|---|---|
| `object` | ✅ `"list"` | ✅ `"list"` | |
| `data[].object` | ✅ `"embedding"` | ✅ `"embedding"` | |
| `data[].embedding` | ✅ float array | ✅ float array or base64 string | Depends on `encoding_format` |
| `data[].index` | ✅ | ✅ | |
| `model` | ✅ (includes `@quantization`) | ✅ | LMStudio appends quantization to model name |
| `usage.prompt_tokens` | ✅ | ✅ | |
| `usage.total_tokens` | ✅ | ✅ | |

---

## POST /v1/responses — the newer Responses API

Both systems support OpenAI's Responses API, though with different levels of feature completeness.

### Request parameters

| Parameter | Type | OpenAI Spec | LMStudio | Ollama | Notes |
|---|---|---|---|---|---|
| `model` | string | ✅ Required | ✅ Required | ✅ Required | |
| `input` | string \| array | ✅ Required | ✅ | ✅ | Text or structured message array |
| `instructions` | string | ✅ | ❌ Not listed (uses system prompt in messages) | ✅ | Ollama maps to system message |
| `stream` | boolean | ✅ | ✅ | ✅ | |
| `temperature` | number | ✅ | ❌ Not listed for this endpoint | ✅ | |
| `top_p` | number | ✅ | ❌ Not listed for this endpoint | ✅ | |
| `max_output_tokens` | integer | ✅ | ❌ Not listed for this endpoint | ✅ | |
| `tools` | array | ✅ | ✅ (includes `type: "mcp"` for remote MCP) | ✅ (`type: "function"` only) | **LMStudio extends with MCP server tools** |
| `reasoning` | object | ✅ | ✅ (`effort`: `low`/`medium`/`high`/`xhigh`) | ✅ (`effort`: `high`/`medium`/`low`/`none`) | LMStudio adds `xhigh`; Ollama adds `none` |
| `previous_response_id` | string | ✅ | ✅ (stateful continuation) | ❌ Accepted but **not functional** | **Major difference**: LMStudio supports stateful conversations; Ollama does not |
| `conversation` | object | ✅ | ❌ Not listed | ❌ Accepted but not functional | |
| `truncation` | string | ✅ | ❌ Not listed | ✅ | |
| `include` | array | ✅ | ❌ Not listed | ❌ Ignored | |
| `logprobs` / `top_logprobs` | ✅ | ✅ (only endpoint with logprob support) | ❌ Not listed | **LMStudio supports logprobs here but not on chat/completions** |

### Streaming event differences

LMStudio emits `response.created`, `response.output_text.delta`, and `response.completed` events. Ollama follows the same OpenAI Responses streaming event format. Both terminate with `data: [DONE]`.

### LMStudio-exclusive MCP tool integration

LMStudio's `/v1/responses` endpoint uniquely supports **remote MCP server tools** defined inline:
```json
{
  "type": "mcp",
  "server_label": "huggingface",
  "server_url": "https://huggingface.co/mcp",
  "allowed_tools": ["model_search"]
}
```
This is not available in Ollama's OpenAI-compatible layer.

---

## POST /v1/images/generations — Ollama only

This experimental endpoint exists **only in Ollama**. LMStudio does not support image generation (it supports image input via VLMs, not output).

| Parameter | Type | Required | Default | Notes |
|---|---|---|---|---|
| `model` | string | ✅ | — | Image generation model (e.g., `x/z-image-turbo`) |
| `prompt` | string | ✅ | — | Text description |
| `size` | string | Optional | — | `"WxH"` format (e.g., `"1024x1024"`) |
| `response_format` | string | Optional | — | **Only `"b64_json"` supported** |
| `n` | integer | Optional | 1 | Number of images |
| `quality` | string | Optional | — | Quality setting |
| `style` | string | Optional | — | Style setting |
| `seed` | integer | Optional | — | Ollama extension |
| `user` | string | Optional | — | User identifier |

Response returns `created` timestamp and `data[]` array with `b64_json` (base64-encoded image) fields. URL-based responses are not supported.

---

## Behavioral differences that affect migration

### Streaming protocol

Both systems use identical SSE formatting: `data: {json}\n\n` chunks terminated by `data: [DONE]`. Chat streaming uses `chat.completion.chunk` objects with `delta` fields; completions streaming uses `text_completion` objects with `text` fields. The `stream_options.include_usage` flag works identically. **No migration changes needed for streaming code.**

### The `response_format` divergence matters

LMStudio supports **only** `json_schema` and `text` types, and actively **rejects** `json_object` with an error. Ollama supports `json_object` (mapped to simple JSON mode) and `json_schema` (schema passed through). Code using `{"type": "json_object"}` will work on Ollama but **fail on LMStudio** — switch to `json_schema` with an explicit schema for cross-compatibility.

### Logprobs availability is asymmetric

Ollama provides full `logprobs` and `top_logprobs` support on both `/v1/chat/completions` and `/v1/completions`. LMStudio supports logprobs **only** on the `/v1/responses` endpoint (using `include: ["message.output_text.logprobs"]`). Any code relying on token-level probabilities from chat completions must use Ollama or switch to LMStudio's Responses API.

### Model auto-loading vs. pre-loading

LMStudio's JIT loading means sending a request with any downloaded model name will automatically load it, with the `ttl` parameter controlling when it unloads. Ollama also auto-loads models on request but provides no TTL mechanism through the OpenAI compat layer — model lifetime is controlled via the native API's `keep_alive` parameter. **Neither exposes context window configuration through the OpenAI layer**; LMStudio uses its native API's `context_length`, while Ollama requires a Modelfile.

### Extra response metadata from LMStudio

LMStudio enriches every inference response with `stats` (tokens/second, TTFT, generation time), `model_info` (architecture, quantization, format, context length), and `runtime` (engine name and version). Ollama provides none of these — its responses are leaner and closer to the bare OpenAI spec. Code that consumes these extra fields must handle their absence when switching to Ollama.

---

## Comprehensive unsupported parameter reference

The following OpenAI spec parameters are **not supported by either system**:

| OpenAI Parameter | LMStudio | Ollama | Notes |
|---|---|---|---|
| `max_completion_tokens` | ❌ Ignored | ❌ Ignored | Use `max_tokens` instead |
| `parallel_tool_calls` | ❌ Ignored | ❌ Ignored | |
| `service_tier` | ❌ Ignored | ❌ Ignored | Cloud-only |
| `metadata` | ❌ Ignored | ❌ Ignored | |
| `store` (OpenAI version) | ❌ Ignored | ❌ Ignored | Not the same as LMStudio's native `store` |
| `modalities` | ❌ Ignored | ❌ Ignored | Audio/multimodal selection |
| `audio` | ❌ Ignored | ❌ Ignored | Audio generation params |
| `prediction` | ❌ Ignored | ❌ Ignored | Predicted output |

---

## Extension parameters beyond the OpenAI spec

### LMStudio-only extensions

| Parameter | Endpoint(s) | Type | Description |
|---|---|---|---|
| `top_k` | `/v1/chat/completions`, `/v1/completions` | integer | Top-k token sampling limit |
| `repeat_penalty` | `/v1/chat/completions`, `/v1/completions` | number | Repetition penalty (1.0 = none) |
| `ttl` | `/v1/chat/completions` | integer | Seconds before auto-unloading JIT-loaded model |
| `draft_model` | `/v1/chat/completions` | string | Model identifier for speculative decoding |
| MCP tools in `tools[]` | `/v1/responses` | object | `type: "mcp"` with `server_url` for remote MCP |

LMStudio also adds response-only extensions: `stats`, `model_info`, and `runtime` objects on all inference responses.

### Ollama-only extensions (via OpenAI compat)

| Parameter | Endpoint(s) | Type | Description |
|---|---|---|---|
| `suffix` | `/v1/completions` | string | Fill-in-the-middle text completion |
| `echo` | `/v1/completions` | boolean | Return prompt with completion |
| `best_of` | `/v1/completions` | integer | Generate N completions, return best |
| `seed` (images) | `/v1/images/generations` | integer | Random seed for image generation |

Ollama's response-only extension is the fixed `system_fingerprint: "fp_ollama"` field.

---

## Conclusion

The two systems are largely interchangeable for basic chat completions, embeddings, and model listing — a simple base URL swap covers most use cases. The critical migration pitfalls are **`response_format` type support** (avoid `json_object` on LMStudio), **logprobs availability** (Ollama has it on chat; LMStudio only on responses), and **stateful conversations** (LMStudio supports `previous_response_id` on `/v1/responses`; Ollama does not). LMStudio's extensions skew toward inference control and observability (`top_k`, `repeat_penalty`, performance `stats`, speculative decoding), while Ollama's skew toward spec completeness (`logprobs`, `suffix`, `n`, `encoding_format`, image generation). For maximum cross-compatibility, stick to the shared parameter subset: `model`, `messages`/`prompt`, `temperature`, `top_p`, `max_tokens`, `stop`, `seed`, `presence_penalty`, `frequency_penalty`, `stream`, `tools`, `response_format` with `json_schema` type, and `logit_bias`.