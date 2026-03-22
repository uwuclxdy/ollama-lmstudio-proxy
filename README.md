# Ollama ↔ LM Studio Proxy

Proxy server that bridges **Ollama API** and **LM Studio**

Useful if you want to **connect models from LM Studio to applications that support only Ollama API** (such as Copilot in
VS Code).

> ⚠️ This project was in the majority vibe coded. I only want to let you know, despite me regularly using it. **Feel
free to report any bugs or auditing code before use!**

![preview.png](preview.png)

## Highlights

- Ollama API endpoints translated to LM Studio native (`/api/v1/*`) and OpenAI-compatible (`/v1/*`) backends
- Automatic model name mapping to Ollama format (with virtual aliases stored on disk)
- **Streaming**: Optimized SSE processing with chunk recovery and cancellation
- Thinking/reasoning capability detection for models (e.g. DeepSeek-R1, QwQ)
- Auto-retry, model preload hints, and catalog-backed downloads via `/api/pull`

## Configuration

### CLI Options

| Flag                                    | Default                 | Description                                              |
|-----------------------------------------|-------------------------|----------------------------------------------------------|
| `--listen`                              | `0.0.0.0:11434`         | Server bind address                                      |
| `--lmstudio-url`                        | `http://localhost:1234` | LM Studio URL                                            |
| `--log-level`                           | `info`                  | Set log level (`off`, `error`, `warn`, `info`, `debug`, `trace`) |
| `--load-timeout-seconds`                | `15`                    | Model loading wait timeout in seconds (after trigger)    |
| `--model-resolution-cache-ttl-seconds`  | `300`                   | Cache TTL for model resolution                           |
| `--max-buffer-size`                     | `262144`                | Initial buffer size for SSE message assembly (bytes)     |
| `--enable-chunk-recovery`               | `false`                 | Enable partial chunk recovery for streams                |
| `--update`                              | —                       | Check for updates and replace the executable if a newer version is available |

## LM Studio API Compatibility

Requires LM Studio **0.3.6+**. Legacy mode is removed; `/v1/*` requests are forwarded directly, while Ollama endpoints
translate to LM Studio native APIs.

### Endpoint Support

| Endpoint                        | Behaviour                                                          |
|---------------------------------|--------------------------------------------------------------------|
| `GET /`                         | Returns "Ollama is running"                                        |
| `GET /api/tags`                 | Translates to `/api/v1/models`; includes proxy-managed aliases     |
| `GET /api/ps`                   | Translates to `/api/v1/models`; shows loaded models plus aliases   |
| `POST /api/show`                | Fetches real LM Studio metadata; merges alias info when present    |
| `POST /api/chat`                | Translates to `/v1/chat/completions`                               |
| `POST /api/generate`            | Translates to `/v1/completions`; vision requests use chat endpoint |
| `POST /api/embed`               | Translates to `/v1/embeddings`; also handles `/api/embeddings`     |
| `GET /api/version`              | Returns proxy version in Ollama format                             |
| `GET /health`                   | Validates LM Studio reachability                                   |
| `POST /api/create`              | Creates proxy-managed virtual aliases (no custom blobs)            |
| `POST /api/pull`                | Translates to `/api/v1/models/download`; streams download progress |
| `POST /api/push`                | No-op; validates that the target model exists                      |
| `DELETE /api/delete`            | Removes proxy-managed aliases only                                 |
| `POST /api/copy`                | Duplicates aliases or references LM Studio models                  |
| `HEAD/POST /api/blobs/:digest`  | Stores and validates blobs for alias manifests                     |

`ANY /v1/*` and `ANY /api/v1/*` are forwarded directly to LM Studio without modification.

### Virtual model aliases

- `/api/create` and `/api/copy` manage aliases stored under
  `$XDG_CACHE_HOME/ollama-lmstudio-proxy/virtual_models.json` (fallback: `$HOME/.cache/ollama-lmstudio-proxy/`, then system temp). Metadata such as `system`,
  `template`, `parameters`, `license`, `adapters`, and `messages` is merged into subsequent requests.
- `/api/delete` removes only proxy-managed aliases. `/api/show` returns LM Studio metadata plus alias info when present.
- `/api/pull` streams LM Studio catalog downloads (or blocks when `"stream": false`); optional `quantization` and
  `source` fields are forwarded.

## Installation options

### 3. Cargo (Recommended)

```bash
cargo install ollama-lmstudio-proxy
```

### 2. Pre-built Binary

1. Download the latest release from the [Releases](https://github.com/uwuclxdy/ollama-lmstudio-proxy/releases) page.
2. Run the binary in terminal.

### 3. Source

```bash
# Clone the repository
git clone https://github.com/uwuclxdy/ollama-lmstudio-proxy.git
cd ollama-lmstudio-proxy

# Build release version
cargo build --release

# Run
./target/release/ollama-lmstudio-proxy
```

## Quick Start

### Basic Usage

```bash
# Start with default settings (native API)
ollama-lmstudio-proxy

# Custom configuration
ollama-lmstudio-proxy \
  --listen 0.0.0.0:11434 \
  --lmstudio-url http://localhost:1234 \
  --load-timeout-seconds 30
```

**Make sure the Ollama server is not running when using the proxy!**

## Request Shapes & Examples

The proxy understands two payload styles:

- **Ollama-style**: model parameters belong in `options`. Use this when calling `/api/*` endpoints.
- **OpenAI-style passthrough**: send OpenAI-compatible JSON to `/v1/*` and the proxy forwards it untouched.

When you target Ollama endpoints, put fields such as `temperature`, `num_predict`, `max_tokens`, `logit_bias`, and
structured `format` values inside the `options` object (or top-level `format`). The proxy translates them to LM
Studio's native parameters. Set `"format": "json"` for quick JSON responses or provide a JSON Schema object (also
accepted inside `options.format`) to use LM Studio's structured output enforcement.

### Option mappings

These parameters go inside the `options` object:

| Ollama option              | LM Studio parameter                  | Notes                                                      |
|----------------------------|--------------------------------------|------------------------------------------------------------|
| `temperature`, `top_p`     | Same name                            | Direct passthrough                                         |
| `top_k`                    | `top_k`                              | Direct passthrough                                         |
| `min_p`                    | `min_p`                              | Direct passthrough                                         |
| `presence_penalty`         | `presence_penalty`                   | Direct passthrough                                         |
| `frequency_penalty`        | `frequency_penalty`                  | Direct passthrough                                         |
| `repeat_penalty`           | `repeat_penalty`/`frequency_penalty` | Mapped depending on what is already set                    |
| `max_tokens`/`num_predict` | `max_tokens`                         | Picks whichever you set; `max_tokens` takes priority       |
| `num_ctx`                  | `context_length`                     | Context window size                                        |
| `logit_bias`               | `logit_bias`                         | Accepts JSON object or map notation                        |
| `system` (in `options`)    | `system`                             | Injected as LM Studio system prompt                        |
| `stop`, `seed`             | Same name                            | Direct passthrough                                         |
| `truncate`                 | `truncate`                           | Direct passthrough                                         |
| `dimensions`               | `dimensions`                         | Direct passthrough (embeddings)                            |

These parameters go at the **top level** of the request body (not inside `options`):

| Ollama field               | LM Studio parameter                  | Notes                                                                        |
|----------------------------|--------------------------------------|------------------------------------------------------------------------------|
| `think`                    | `reasoning`                          | `true`→`"on"`, `false`→`"off"`, string passed through; omit to leave unset  |
| `logprobs`, `top_logprobs` | Same name                            | Direct passthrough                                                           |
| `suffix`                   | `suffix`                             | Forwarded on non-vision generate requests only                               |
| `raw`                      | —                                    | Disables system-prompt injection in generate requests                        |
| `keep_alive`               | `ttl`                                | Seconds (int) or duration string (`"5m"`); `0` unloads the model immediately |

If you do not need structured output, you can still pass overrides such as `stop`, `seed`, or additional
OpenAI-compatible payloads and the proxy will pass them through untouched.
