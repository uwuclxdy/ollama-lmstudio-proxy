# 🔌 API compatibility

`/v1/*` and `/api/v1/*` are forwarded directly to LM Studio. Every other Ollama
endpoint is translated to its native equivalent.

## Endpoint support

| Endpoint | Behaviour |
|----------|-----------|
| `GET /` | Returns "Ollama is running" |
| `GET /api/tags` | Translates to `/api/v1/models`; includes proxy-managed aliases |
| `GET /api/ps` | Translates to `/api/v1/models`; shows loaded models plus aliases |
| `POST /api/show` | Fetches real LM Studio metadata; merges alias info when present |
| `POST /api/chat` | Translates to `/v1/chat/completions` (or native `/api/v1/chat` with `--use-native-chat`) |
| `POST /api/generate` | Translates to `/v1/completions`; vision requests use chat endpoint |
| `POST /api/embed` | Translates to `/v1/embeddings`; also handles `/api/embeddings` |
| `GET /api/version` | Returns proxy version in Ollama format |
| `GET /health` | Validates LM Studio reachability |
| `POST /api/create` | Creates proxy-managed virtual aliases (no custom blobs) |
| `POST /api/pull` | Translates to `/api/v1/models/download`; streams download progress |
| `POST /api/push` | Returns 501 (LM Studio has no model registry) |
| `POST /api/web_search` | Returns 501 (cloud-only Ollama feature, no LM Studio backend) |
| `POST /api/web_fetch` | Returns 501 (cloud-only Ollama feature, no LM Studio backend) |
| `DELETE /api/delete` | Removes proxy-managed aliases only |
| `POST /api/copy` | Duplicates aliases or references LM Studio models |
| `HEAD/POST /api/blobs/:digest` | Stores and validates blobs for alias manifests |

## Verbatim passthrough

`ANY /v1/*` and `ANY /api/v1/*` are forwarded directly to LM Studio without
modification. This includes `POST /v1/messages` (Anthropic-compat) and
`POST /v1/responses` (OpenAI Responses), which LM Studio serves natively. The
proxy only remaps the `model` field from the Ollama-style name to the resolved
LM Studio id before forwarding.

Anthropic clients such as Claude Code work against `/v1/messages` with no extra
setup. See the
[Claude Code section](https://github.com/uwuclxdy/ollama-lmstudio-proxy#-claude-code-clients)
in the README.

## Virtual model aliases

- `/api/create` and `/api/copy` manage aliases stored under
  `$XDG_CACHE_HOME/ollama-lmstudio-proxy/virtual_models.json` (fallback:
  `$HOME/.cache/ollama-lmstudio-proxy/`, then system temp). Metadata such as
  `system`, `template`, `parameters`, `license`, `adapters`, and `messages` is
  merged into subsequent requests.
- `/api/delete` removes only proxy-managed aliases. `/api/show` returns LM Studio
  metadata plus alias info when present.
- `/api/pull` streams LM Studio catalog downloads (or blocks when
  `"stream": false`); optional `quantization` and `source` fields are forwarded.
