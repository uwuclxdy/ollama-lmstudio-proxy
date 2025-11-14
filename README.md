# Ollama ↔ LM Studio Proxy

Proxy server that bridges **Ollama API** and **LM Studio**

Useful if you want to **connect models from LM Studio to applications that support only Ollama API** (such as Copilot in VS Code).

> ⚠️ This project was in the majority vibe coded. I only want to let you know, despite me regularly using it. **Feel free to report any bugs or auditing code before use!**

## Highlights

- Supports both Native LM Studio REST API (`/api/v0/`) or legacy OpenAI endpoints (`/v1/`)
- Automatic model name mapping to Ollama format
- **Streaming**: Optimized SSE processing with chunk recovery and cancellation
- Auto-retry and auto-load models in LM Studio

## Configuration

### CLI Options

| Flag                                   | Default                 | Description                    |
|----------------------------------------|-------------------------|--------------------------------|
| `--listen`                             | `0.0.0.0:11434`         | Server bind address            |
| `--lmstudio_url`                       | `http://localhost:1234` | LM Studio URL          |
| `--legacy`                             | `false`                 | Use legacy OpenAI API mode (in LM Studio)     |
| `--no_log`                             | `false`                 | Disable logging         |
| `--load_timeout_seconds`               | `15`                    | Model loading timeout          |
| `--model_resolution_cache_ttl_seconds` | `300`                   | Cache TTL for model resolution |
| `--max_buffer_size`                    | `262144`                | Buffer size for streaming (bytes)        |
| `--enable_chunk_recovery`              | `false`                 | Enable stream chunk recovery   |

## LM Studio API Comparison

| Feature                   | Native API    | Legacy API  |
|---------------------------|----------------|--------------|
| **LM Studio Version**     | 0.3.6+         | 0.2.0+       |
| **Model Loading State**   | ✅ Real-time    | ⚠️ Estimated  |
| **Context Length Limits** | ✅ Accurate     | ⚠️ Generic    |
| **Performance Metrics**   | ✅ Native stats | ⚠️ Calculated |
| **Model Metadata**        | ✅ Rich details | ⚠️ Basic info |
| **Publisher Info**        | ✅ Available    | ❌ Unknown    |

### Endpoint Support

| Ollama Endpoint      | Legacy API              | Native API                  | Notes                              |
|----------------------|--------------------------|------------------------------|------------------------------------|
| `GET /api/tags`      | ✅ `/v1/models`           | ✅ `/api/v0/models`           |                                    |
| `GET /api/ps`        | ✅ `/v1/models`           | ✅ `/api/v0/models`           | Shows loaded models only           |
| `POST /api/show`     | ⚠️ *Fabricated info*           | ⚠️ *Fabricated info*               | Generated from model name          |
| `POST /api/chat`     | ✅ `/v1/chat/completions` | ✅ `/api/v0/chat/completions` |                                    |
| `POST /api/generate` | ✅ `/v1/completions`      | ✅ `/api/v0/completions`      | Vision support via chat endpoint   |
| `POST /api/embed`    | ✅ `/v1/embeddings`       | ✅ `/api/v0/embeddings`       | Also supports `/api/embeddings`    |
| `GET /api/version`   | ✅ *Proxy response*       | ✅ *Proxy response*           |                                    |
| `GET /health`        | ✅ *Health check*         | ✅ *Health check*             |                                    |
| `POST /v1/*`         | ✅ *Direct passthrough*   | ✅ *Converts to /api/v0/*     |                                    |
| `POST /api/create`   | ❌                        | ❌                            | Use LM Studio for model management |
| `POST /api/pull`     | ❌                        | ❌                            |                                    |
| `POST /api/push`     | ❌                        | ❌                            |                                    |
| `POST /api/delete`   | ❌                        | ❌                            |                                    |
| `POST /api/copy`     | ❌                        | ❌                            |                                    |

## Installation

### From Pre-built Binaries

1. Download latest release from the [Releases](https://github.com/uwuclxdy/ollama-lmstudio-proxy/releases) page.
2. Extract and run the binary in terminal.

### From Source

```bash
# Clone the repository
git clone https://github.com/uwuclxdy/ollama-lmstudio-proxy.git
cd ollama-lmstudio-proxy

# Build release version
cargo build --release

# Run
./target/release/ollama-lmstudio-proxy
```

### Using Cargo

```bash
cargo install --git https://github.com/uwuclxdy/ollama-lmstudio-proxy.git
```

## Quick Start

### Basic Usage

```bash
# Start with default settings (native API)
ollama-lmstudio-proxy

# Use legacy API for older LM Studio versions
ollama-lmstudio-proxy --legacy

# Custom configuration
ollama-lmstudio-proxy \
  --listen 0.0.0.0:11434 \
  --lmstudio_url http://localhost:1234 \
  --load_timeout_seconds 30
```

### Test the Connection

```bash
# Check health status
curl http://localhost:11434/health

# List available models
curl http://localhost:11434/api/tags

# Send a chat request
curl http://localhost:11434/api/chat -d '{
  "model": "llama2",
  "messages": [{"role": "user", "content": "Hello!"}]
}'
```

## Request Shapes & Examples

The proxy understands two payload styles:

- **Ollama-style**: advanced parameters belong in `options`. Use this when calling `/api/*` endpoints.
- **OpenAI-style passthrough**: send OpenAI-compatible JSON to `/v1/*` and the proxy forwards it untouched.

When you target Ollama endpoints, keep fields such as `temperature`, `num_predict`, `max_tokens`, `logit_bias`, and structured `format` values inside the `options` object (or top-level `format`), and the proxy will translate them to LM Studio's native parameters.

### Structured output via `/api/generate`

```bash
curl http://localhost:11434/api/generate -H "Content-Type: application/json" -d '{
  "model": "llama3.1:8b",
  "prompt": "Return a status payload for ACME widgets.",
  "stream": false,
  "format": {
    "type": "object",
    "properties": {file:color-palette-template.md 
      "status": {"type": "string"},
      "checked_at": {"type": "string", "format": "date-time"}
    },
    "required": ["status", "checked_at"]
  }
}'
```

### Structured output via `/api/chat`

```bash
curl http://localhost:11434/api/chat -H "Content-Type: application/json" -d '{
  "model": "llama3.1:8b",
  "messages": [
    {"role": "system", "content": "Answer with machine readable JSON"},
    {"role": "user", "content": "List two tasks with priorities"}
  ],
  "stream": false,
  "format": {
    "type": "object",
    "properties": {
      "tasks": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "title": {"type": "string"},
            "priority": {"type": "integer"}
          },
          "required": ["title", "priority"]
        }
      }
    },
    "required": ["tasks"]
  }
}'
```

### `logit_bias` (discourage/force tokens)

```bash
curl http://localhost:11434/api/chat -H "Content-Type: application/json" -d '{
  "model": "llama3.1:8b",
  "messages": [{"role": "user", "content": "Reply yes or no"}],
  "options": {
    "logit_bias": {
      "464": -10,   // discourage "No"
      "302": 15     // strongly prefer "Yes"
    }
  },
  "stream": false
}'
```

### `max_tokens` vs `num_predict`

Both knobs map to LM Studio's `max_tokens`. Pick the style that matches your client.

```bash
# Ollama default knob
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "prompt": "Write a haiku",
  "options": {"num_predict": 64}
}'

# OpenAI-style knob (same effect)
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "prompt": "Write a haiku",
  "options": {"max_tokens": 64}
}'
```
