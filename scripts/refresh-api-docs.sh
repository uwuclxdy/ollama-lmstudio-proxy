#!/usr/bin/env bash
#
# Refresh the upstream-mirrored API docs under "api_docs/".
#
#   - "api_docs/ollama/<path>"     ← https://docs.ollama.com/<path>
#   - "api_docs/lmstudio/<path>"   ← lmstudio-ai/docs@main:<path>
#
# Both mirrors are gated by explicit allowlists below. Anything not on the
# list is intentionally omitted (out of scope for a translation proxy:
# integrations, platform installers, CLI/Modelfile/FAQ explainers, Ollama
# cloud auth, LM Studio app UI / SDK pages, Anthropic-compat endpoints,
# native REST endpoints the proxy never calls, etc.).
#
# Local-only files under "api_docs/" (lmstudio_ollama_openai.md,
# lmstudio_vs_ollama.md, and anything outside the mirrored trees) are
# never touched.
#
# Re-run safely; the script wipes and rebuilds only the mirrored trees.
set -euo pipefail

OLLAMA_BASE="https://docs.ollama.com"
LMS_REPO="https://github.com/lmstudio-ai/docs.git"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCS_DIR="$REPO_ROOT/api_docs"
OLLAMA_DIR="$DOCS_DIR/ollama"
LMS_DIR="$DOCS_DIR/lmstudio"

# Explicit allowlist of ollama doc paths the proxy mirrors. Each entry is
# the path portion of an https://docs.ollama.com/<path> URL. Anything not
# listed is dropped (see top-of-file comment for the rationale).
OLLAMA_KEEP_PATHS=(
    api/chat.md
    api/copy.md
    api/create.md
    api/delete.md
    api/embed.md
    api/errors.md
    api/generate.md
    api/introduction.md
    api/ps.md
    api/pull.md
    api/streaming.md
    api/tags.md
    api/usage.md
    api-reference/get-version.md
    api-reference/show-model-details.md
    capabilities/embeddings.md
    capabilities/streaming.md
    capabilities/structured-outputs.md
    capabilities/thinking.md
    capabilities/tool-calling.md
    capabilities/vision.md
)

# Explicit allowlist of LM Studio doc paths under "1_developer/" that the
# proxy actually depends on (REST endpoints it calls, OpenAI-compat shapes
# it produces). All other pages — app UI, SDK guides, native chat REST,
# Anthropic-compat, changelog — are dropped.
LMS_KEEP_PATHS=(
    1_developer/2_rest/download.md
    1_developer/2_rest/download-status.md
    1_developer/2_rest/list.md
    1_developer/2_rest/unload.md
    1_developer/3_openai-compat/chat-completions.md
    1_developer/3_openai-compat/embeddings.md
    1_developer/3_openai-compat/models.md
    1_developer/3_openai-compat/structured-output.md
)

if [[ ! -d "$DOCS_DIR" ]]; then
    echo "error: '$DOCS_DIR' does not exist; run from repo containing api_docs/" >&2
    exit 1
fi

for cmd in curl git; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "error: '$cmd' is required but not on PATH" >&2
        exit 1
    fi
done

tmpdir="$(mktemp -d -t refresh-api-docs.XXXXXX)"
trap 'rm -rf "$tmpdir"' EXIT

# --- Ollama ----------------------------------------------------------------

# Drop legacy single-file dump; replaced by the per-page tree under ollama/.
rm -f "$DOCS_DIR/ollama.md"
rm -rf "$OLLAMA_DIR"
mkdir -p "$OLLAMA_DIR"

echo "→ fetching ${#OLLAMA_KEEP_PATHS[@]} ollama pages"
for path in "${OLLAMA_KEEP_PATHS[@]}"; do
    target="$OLLAMA_DIR/$path"
    mkdir -p "$(dirname "$target")"
    curl -fsSL --retry 3 --retry-delay 2 "$OLLAMA_BASE/$path" -o "$target"
done

# --- LM Studio -------------------------------------------------------------

echo "→ cloning lmstudio-ai/docs (shallow)"
git clone --depth=1 --quiet "$LMS_REPO" "$tmpdir/lms-docs"

rm -rf "$LMS_DIR"
mkdir -p "$LMS_DIR"

echo "→ copying ${#LMS_KEEP_PATHS[@]} lmstudio pages"
for path in "${LMS_KEEP_PATHS[@]}"; do
    source="$tmpdir/lms-docs/$path"
    target="$LMS_DIR/$path"
    if [[ ! -f "$source" ]]; then
        echo "warning: upstream file '$path' not found; skipping" >&2
        continue
    fi
    mkdir -p "$(dirname "$target")"
    cp "$source" "$target"
done

echo "✓ done"
