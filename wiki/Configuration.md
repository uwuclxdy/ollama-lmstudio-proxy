# ⚙️ Configuration

Requires LM Studio **0.4.0+**. Model listing, loading, unloading, downloads: all of it rides the `/api/v1` REST API that release introduced.

Every setting is a CLI flag. Where a flag reads an environment variable too, the table names it.

## CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--listen` | `0.0.0.0:11434` | Server bind address |
| `--lmstudio-url` | `http://localhost:1234` | LM Studio URL |
| `--log-level` | `info` | `off`, `error`, `warn`, `info`, `debug`, `trace` (`RUST_LOG` env) |
| `--load-timeout-seconds` | `15` | Model loading wait timeout in seconds (after trigger); must be greater than 0 |
| `--stream-timeout-seconds` | `60` | Maximum time in seconds between streaming response chunks (must be greater than 0); a `keep_alive:0` request also holds the model until its response body ends (all chunks delivered) or the client disconnects, then unloads |
| `--model-resolution-cache-ttl-seconds` | `300` | Cache TTL for model resolution |
| `--max-buffer-size` | `262144` | Initial buffer size for SSE message assembly (bytes) |
| `--enable-chunk-recovery` | `false` | Enable partial chunk recovery for streams |
| `--lmstudio-token` | _none_ | Bearer token for LM Studio auth (`LMSTUDIO_TOKEN` env); sent on backend requests. On an open proxy (no `--api-key`), a caller-supplied `Authorization` overrides it on the `/v1/*` and `/api/v1/*` passthrough (the `/api/*` handlers forward no inbound headers); with `--api-key` set the gate strips the inbound credential, so this token is the one that reaches the backend |
| `--api-key` | _none_ | Inbound credential gate (`OLLAMA_API_KEY` env). Unset means the proxy is open. When set, every request needs `Authorization: Bearer <key>` or `x-api-key: <key>`; the matched credential (both headers) is removed before forwarding, so the proxy key never rides upstream; `GET /api/version` and CORS preflight stay open |
| `--use-native-chat` | `false` | Route `/api/chat` through native `/api/v1/chat` for richer reasoning events, MCP tools, and accurate stats |
| `--native-chat-streaming` | `false` | Same routing for streaming `/api/chat` only; non-streaming stays on the v0 path |
| `--auto-evict` | `false` | Unload every other model's instances before loading a requested one (mirrors Ollama's single-model default). Single-tenant setups only: one client's load evicts another's |
| `--flash-attention` | `false` | Experimental: enable flash attention when loading models via `/api/v1/models/load` |
| `--offload-kv-cache` | `false` | Experimental: offload KV cache to GPU when loading models via `/api/v1/models/load` |
| `--eval-batch-size` | _none_ | Experimental: set eval batch size when loading models via `/api/v1/models/load` |
| `--default-context-length` | _none_ | Server-wide `num_ctx` fallback applied when a request omits it (`OLLAMA_CONTEXT_LENGTH` env); a per-request `num_ctx` still wins |
| `--ollama-version` | `0.30.0` | Version string reported by `GET /api/version` (`OLLAMA_VERSION` env) |
| `--allow-private-fetch` | `false` | Allow `/api/web_fetch` to reach loopback/private/link-local addresses; when off, SSRF guard rejects those targets with 400 |
| `--search-url` | _none_ | Search provider endpoint for `/api/web_search`; unset returns 501 (`SEARCH_URL` env) |
| `--search-api-key` | _none_ | Bearer token sent to the search provider (`SEARCH_API_KEY` env) |

## Native chat mode

`--use-native-chat` routes every `/api/chat` request through LM Studio's `/api/v1/chat`. `--native-chat-streaming` routes the streaming ones only; the rest stay on the v0 path.

The native path adds per-event reasoning deltas, real streaming token stats, [MCP integrations](MCP-Integrations). Its schema has no slot for `tools`, `tool_choice`, `format`, `response_format`, `logprobs`, `top_logprobs` or `images`: a request carrying any of them gets a `warning` field naming what the proxy dropped.

`keep_alive` sets no backend TTL on this path: LM Studio 0.4.28's `/api/v1` surface rejects a `ttl` field (measured 2026-11-16), so a finite value only drives `/api/ps`'s deadline. `keep_alive: 0` still unloads the model once the response body ends.

## Update check

One background call to the GitHub releases API on startup, throttled to once per 24h by a cache file. A newer tag logs at `warn`, a failure at `debug`. No flag disables it.

## Experimental flags

`--flash-attention`, `--offload-kv-cache`, `--eval-batch-size` tune `/api/v1/models/load`. Leave them off by default.

`--allow-private-fetch` disables the SSRF guard on `/api/web_fetch`. Use it only to fetch from localhost or a local network (e.g. testing).
