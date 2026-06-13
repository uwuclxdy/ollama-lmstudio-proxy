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
| `num_ctx` | `context_length` | Context window size |
| `logit_bias` | `logit_bias` | Accepts JSON object or map notation |
| `system` (in `options`) | `system` | Injected as LM Studio system prompt |
| `stop`, `seed` | Same name | Direct passthrough |
| `truncate` | `truncate` | Direct passthrough |
| `dimensions` | `dimensions` | Direct passthrough (embeddings) |

## Fields that go at the top level

| Ollama field | LM Studio parameter | Notes |
|--------------|---------------------|-------|
| `think` / `reasoning_effort` | `reasoning` | `true`→`"on"`, `false`→`"off"`, `"none"`→`"off"`; levels `low\|medium\|high\|on\|off` pass through; `reasoning_effort` is an alias used only when `think` is absent |
| `logprobs`, `top_logprobs` | Same name | Direct passthrough |
| `suffix` | `suffix` | Forwarded on non-vision generate requests only |
| `raw` | _none_ | Disables system-prompt injection in generate requests |
| `keep_alive` | `ttl` | Seconds (int) or duration string (`"5m"`); `0` unloads the model immediately |
| `integrations` | `integrations` | **Native path only** (`--use-native-chat`). Array of MCP tool specs forwarded verbatim. See [MCP Integrations](MCP-Integrations). |
