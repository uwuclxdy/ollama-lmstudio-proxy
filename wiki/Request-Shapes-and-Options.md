# 🧬 Request shapes and options

The proxy understands two payload styles:

- **Ollama-style:** model parameters belong in `options`. Use this for `/api/*` endpoints.
- **OpenAI-style passthrough:** send OpenAI-compatible JSON to `/v1/*`, forwarded untouched.

For Ollama endpoints, put fields such as `temperature`, `num_predict`, `max_tokens`,
`logit_bias`, and structured `format` values inside the `options` object (or
top-level `format`). Set `"format": "json"` for quick JSON, or pass a JSON Schema
object (also accepted inside `options.format`) to use LM Studio's structured output
enforcement.

## Options that go inside `options`

| Ollama option | LM Studio parameter | Notes |
|---------------|---------------------|-------|
| `temperature`, `top_p` | Same name | Direct passthrough |
| `top_k` | `top_k` | Direct passthrough |
| `min_p` | `min_p` | Direct passthrough |
| `presence_penalty` | `presence_penalty` | Direct passthrough |
| `frequency_penalty` | `frequency_penalty` | Direct passthrough |
| `repeat_penalty` | `repeat_penalty` / `frequency_penalty` | Mapped depending on what is already set |
| `max_tokens` / `num_predict` | `max_tokens` | Picks whichever you set; `max_tokens` takes priority |
| `num_ctx` | `context_length` | Reloads the model at the requested context length before inference (LM Studio treats this as a load-time setting). No-op when absent/zero or already loaded at that size. Clamped to the model's max. Two concurrent requests with different `num_ctx` to the same model can race. When absent, falls back to `--default-context-length` / `OLLAMA_CONTEXT_LENGTH` if set. Also honored on `/api/embed` |
| `logit_bias` | `logit_bias` | Accepts JSON object or map notation |
| `system` (in `options`) | `system` | Injected as LM Studio system prompt |
| `stop`, `seed` | Same name | Direct passthrough |
| `truncate` | `truncate` | Direct passthrough; defaults to `true` on `/api/embed` when omitted (matches Ollama) so overlong inputs truncate instead of erroring |
| `dimensions` | `dimensions` | Direct passthrough (embeddings) |

### Accepted but ignored

Ollama-only sampling/runtime options with no LM Studio equivalent (e.g. `mirostat*`,
`tfs_z`, `typical_p`, `repeat_last_n`, `num_keep`, `num_gpu`, `num_thread`, `numa`) are
accepted without error and dropped, surfaced once per request in a warn-log so they are
not silently swallowed.

`draft_num_predict` (max speculative draft tokens per step) falls here too: LM Studio
configures speculative decoding at model-load time via a draft model, with no per-request
knob in its API, so the option is dropped — a no-op, matching Ollama itself when no draft
model is loaded.

## Fields that go at the top level

| Ollama field | LM Studio parameter | Notes |
|--------------|---------------------|-------|
| `think` / `reasoning_effort` | `reasoning` | `true`→`"on"`, `false`→`"off"`, `"none"`→`"off"`; levels `low\|medium\|high\|on\|off` pass through; `reasoning_effort` is an alias used only when `think` is absent. When `think` is omitted and the model is reasoning-capable (LM Studio reports a `reasoning` capability), defaults to `"on"` to match Ollama; explicit `think:false` always wins |
| `logprobs`, `top_logprobs` | Same name | Direct passthrough |
| `suffix` | `suffix` | Forwarded on non-vision generate requests only |
| `raw` | _none_ | Disables system-prompt injection in generate requests |
| `keep_alive` | `ttl` | Seconds (int) or duration string (`"5m"`); `0` unloads the model immediately |
| `tool_choice` | `tool_choice` | Forwarded on `/api/chat` (OpenAI-compat path) when `tools` is also present. Not forwarded without tools, and not on the `--use-native-chat` path |
| `integrations` | `integrations` | **Native path only** (`--use-native-chat`). Array of MCP tool specs forwarded verbatim. See [MCP Integrations](MCP-Integrations). |
