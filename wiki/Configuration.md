# ⚙️ Configuration

Requires LM Studio **0.3.6+**.

All settings are passed as CLI flags. `--log-level` also reads the `RUST_LOG`
environment variable. Other flags that read env vars are noted in the table below.

## CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--listen` | `0.0.0.0:11434` | Server bind address |
| `--lmstudio-url` | `http://localhost:1234` | LM Studio URL |
| `--log-level` | `info` | `off`, `error`, `warn`, `info`, `debug`, `trace`; also reads `RUST_LOG` |
| `--load-timeout-seconds` | `15` | Model loading wait timeout in seconds (after trigger) |
| `--model-resolution-cache-ttl-seconds` | `300` | Cache TTL for model resolution |
| `--max-buffer-size` | `262144` | Initial buffer size for SSE message assembly (bytes) |
| `--enable-chunk-recovery` | `false` | Enable partial chunk recovery for streams |
| `--lmstudio-token` | _none_ | Bearer token for LM Studio auth (`LMSTUDIO_TOKEN` env); sent on backend requests, overridden by a caller-supplied `Authorization` |
| `--use-native-chat` | `false` | Experimental: route `/api/chat` through native `/api/v1/chat` for richer reasoning events and accurate stats |
| `--flash-attention` | `false` | Experimental: enable flash attention when loading models via `/api/v1/models/load` |
| `--offload-kv-cache` | `false` | Experimental: offload KV cache to GPU when loading models via `/api/v1/models/load` |
| `--eval-batch-size` | _none_ | Experimental: set eval batch size when loading models via `/api/v1/models/load` |
| `--default-context-length` | _none_ | Server-wide `num_ctx` fallback applied when a request omits it (`OLLAMA_CONTEXT_LENGTH` env); a per-request `num_ctx` still wins |
| `--ollama-version` | `0.30.0` | Version string reported by `GET /api/version` (`OLLAMA_VERSION` env) |
| `--allow-private-fetch` | `false` | Allow `/api/web_fetch` to reach loopback/private/link-local addresses; when off, SSRF guard rejects those targets with 400 |
| `--search-url` | _none_ | Search provider endpoint for `/api/web_search`; unset returns 501 (`SEARCH_URL` env) |
| `--search-api-key` | _none_ | Bearer token sent to the search provider (`SEARCH_API_KEY` env) |

## Experimental flags

`--use-native-chat`, `--flash-attention`, `--offload-kv-cache`, and
`--eval-batch-size` are experimental. The first routes chat through LM Studio's
native endpoint (see [MCP Integrations](MCP-Integrations)); the other three tune
`/api/v1/models/load`. Leave them off unless you know you need them.

`--allow-private-fetch` disables the SSRF guard on `/api/web_fetch`; only use it
when you need to fetch from localhost or a local network (e.g. testing).
