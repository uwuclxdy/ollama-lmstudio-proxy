# 🔌 API compatibility

`/v1/*` and `/api/v1/*` are forwarded directly to LM Studio. Every other Ollama
endpoint is translated to its native equivalent.

## Endpoint support

| Endpoint | Behaviour |
|----------|-----------|
| `GET /` | Returns "Ollama is running" |
| `GET /api/tags` | Translates to `/api/v1/models`; includes proxy-managed aliases |
| `GET /api/ps` | Translates to `/api/v1/models`; shows loaded models plus aliases; `size_vram` mirrors the loaded model `size` (LM Studio reports no GPU/CPU split); `details.parent_model` is `""`; `expires_at` is a best-effort placeholder |
| `POST /api/show` | Fetches real LM Studio metadata; capabilities (`vision`/`tools`/`thinking`) come from the backend `capabilities` object, with an id-keyword fallback only when the backend reports none; `description`/`display_name` surfaced; verbose `model_info` adds loaded tuning (`flash_attention`/`eval_batch_size`/`parallel`) while the model is loaded; merges alias info when present |
| `POST /api/chat` | Translates to `/api/v0/chat/completions` for real token stats (or native `/api/v1/chat` with `--use-native-chat`) |
| `POST /api/generate` | Translates to `/api/v0/completions`; vision requests use the v0 chat endpoint |
| `POST /api/embed` | Translates to `/v1/embeddings`; also handles `/api/embeddings`. Auto-loads (JIT) an unloaded embedding model on demand instead of returning "no models loaded"; honors `num_ctx`; `truncate` defaults to `true` |
| `GET /api/version` | Returns configurable version string (`--ollama-version`, default `0.30.0`) in Ollama format |
| `GET /health` | Validates LM Studio reachability |
| `POST /api/create` | Creates proxy-managed virtual aliases (no custom blobs) |
| `POST /api/pull` | Translates to `/api/v1/models/download`; streams download progress; `insecure` is accepted and ignored (no TLS-skip surface to emulate); failed downloads surface LM Studio's `error_message` |
| `POST /api/push` | Returns 501 (LM Studio has no model registry) |
| `POST /api/web_search` | Generic JSON passthrough to a configurable provider (`--search-url`); returns 501 when unconfigured. Request: `{query, max_results?}`; provider response returned verbatim |
| `POST /api/web_fetch` | Fetches URL, renders HTML to markdown. Request: `{url}`; response: `{title, content, links}`. SSRF guard on by default (disable with `--allow-private-fetch`). No LM Studio dependency |
| `DELETE /api/delete` | Removes proxy-managed aliases only |
| `POST /api/copy` | Duplicates aliases or references LM Studio models; returns an empty `200` body and upserts (overwrites an existing destination) |
| `HEAD/POST /api/blobs/:digest` | Stores and validates blobs for alias manifests |

## Error codes

Upstream LM Studio `429` (rate limited) and `502` (bad gateway) pass through
unchanged; other upstream-unreachable failures map to `503`. Proxy-side validation
errors return `400`, and a model missing from LM Studio returns `404`.

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
