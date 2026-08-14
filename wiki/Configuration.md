# ⚙️ Configuration

Requires LM Studio **0.4.0+**. Model listing, loading, unloading, downloads: all of it rides the `/api/v1` REST API that release introduced.

Every setting is a CLI flag. Where a flag reads an environment variable too, the table names it.

## CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--listen` | `0.0.0.0:11434` | Server bind address |
| `--lmstudio-url` | `http://localhost:1234` | LM Studio URL |
| `--log-level` | `info` | `off`, `error`, `warn`, `info`, `debug`, `trace` (`RUST_LOG` env) |
| `--load-timeout-seconds` | `15` | Model loading wait timeout in seconds (after trigger) |
| `--model-resolution-cache-ttl-seconds` | `300` | Cache TTL for model resolution |
| `--max-buffer-size` | `262144` | Initial buffer size for SSE message assembly (bytes) |
| `--enable-chunk-recovery` | `false` | Enable partial chunk recovery for streams |
| `--lmstudio-token` | _none_ | Bearer token for LM Studio auth (`LMSTUDIO_TOKEN` env); sent on backend requests, overridden by a caller-supplied `Authorization` |
| `--api-key` | _none_ | Inbound credential gate (`OLLAMA_API_KEY` env). Unset means the proxy is open. When set, every request needs `Authorization: Bearer <key>` or `x-api-key: <key>`; `GET /api/version` and CORS preflight stay open |
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

The native path adds per-event reasoning deltas, real streaming token stats, [MCP integrations](MCP-Integrations). Its schema has no slot for `tools`, `tool_choice`, `format`: a request carrying any of them gets a `warning` field naming what the proxy dropped.

## Update check

One background call to the GitHub releases API on startup, throttled to once per 24h by a cache file. A newer tag logs at `warn`, a failure at `debug`. No flag disables it.

## Experimental flags

`--flash-attention`, `--offload-kv-cache`, `--eval-batch-size` tune `/api/v1/models/load`. Leave them off by default.

`--allow-private-fetch` disables the SSRF guard on `/api/web_fetch`. Use it only to fetch from localhost or a local network (e.g. testing).
