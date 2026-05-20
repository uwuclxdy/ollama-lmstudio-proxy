# Unreleased — 2026-05-20

## Breaking changes
- `POST /api/push` now returns HTTP 501 instead of a synthesised success response — LM Studio has no model registry, so the previous no-op masked the unsupported operation.

## Features
- Graceful shutdown: ctrl-c and SIGTERM now drain in-flight requests instead of killing connections mid-stream.
- Each request emits an access log line with method, path, status, and duration.
- `--log-level` reads from the `RUST_LOG` environment variable when the flag is not passed.
- `--update` detects whether the binary was installed via cargo or downloaded directly and prints the right next step.
- Release binaries published for macOS x64 and macOS ARM64 in addition to Linux x64 and Windows x64.
- Continuous integration runs fmt, clippy, and the full test suite on every push and pull request.

## Bug fixes
- `/api/embed` rejects an empty `input` string up front instead of forwarding an invalid request to LM Studio.
- `/api/create` returns HTTP 400 for blob-based and quantize requests, which the proxy does not support.
- `/api/pull` emits a bare `{"status":"success"}` on the terminal chunk to match Ollama's wire format.
- Passthrough requests to `/v1/*` and `/api/v1/*` now forward LM Studio responses verbatim, including headers.
- Streaming chunk recovery parses each chunk as a whole before falling back to object or array heuristics.
- Model name resolution strips only the `:latest` tag, matching Ollama's behaviour.
- Retry logic skips the model-load wait when LM Studio returns a 4xx backend error.
- `/api/blobs/:digest` correctly hex-encodes sha256 digests after the `sha2 0.11` API change.

## Internal
- Source tree reorganised into `api`, `lmstudio`, and `proxy` modules; single-use submodules inlined.
- Passthrough query encoding moved from a hand-rolled routine to `url::form_urlencoded`.
- Test coverage substantially expanded: wiremock-driven integration tests for every Ollama endpoint, plus unit coverage for streaming, model resolution, storage, and HTTP helpers.
- Package metadata extended with `rust-version`, `categories`, and `readme` for crates.io.
- Dependencies refreshed across the tree.
