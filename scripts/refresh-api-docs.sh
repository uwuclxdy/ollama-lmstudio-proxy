#!/usr/bin/env bash
#
# Refresh the upstream-mirrored API docs under "api-docs/".
#
#   - api-docs/ollama/<path>           ← https://docs.ollama.com/<path>
#         paths listed in OLLAMA_ACTIVE
#   - api-docs/future/ollama/<path>    ← https://docs.ollama.com/<path>
#         paths reachable from llms.txt but neither active nor denied
#   - api-docs/lmstudio/<path>         ← lmstudio-ai/docs@main:<path>
#         paths listed in LMS_ACTIVE
#   - api-docs/future/lmstudio/<path>  ← lmstudio-ai/docs@main:<path>
#         paths in the upstream tree but neither active nor denied
#
# Layout invariant: future/<source>/<path> mirrors the upstream path of
# the same doc, so promotion is a one-line list edit — move the entry from
# its implicit future bucket into the matching *_ACTIVE list and re-run.
#
# DENY lists capture pages that are pure noise — install guides, third-
# party integration walkthroughs, marketing landings, lorem-ipsum
# placeholders, historical changelogs. They are neither active nor future.
# Anything genuinely new added upstream auto-flows into future/ so it
# surfaces for review on the next refresh.
#
# Local-only files under api-docs/ (e.g. lmstudio_ollama_openai.md,
# lmstudio_vs_ollama.md) are never touched.
#
# Re-run safely: only the mirrored trees are wiped and rebuilt.

set -euo pipefail

OLLAMA_BASE="https://docs.ollama.com"
OLLAMA_LLMS_TXT="$OLLAMA_BASE/llms.txt"
LMS_REPO="https://github.com/lmstudio-ai/docs.git"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCS_DIR="$REPO_ROOT/api-docs"
OLLAMA_DIR="$DOCS_DIR/ollama"
LMS_DIR="$DOCS_DIR/lmstudio"
FUTURE_OLLAMA_DIR="$DOCS_DIR/future/ollama"
FUTURE_LMS_DIR="$DOCS_DIR/future/lmstudio"

# Ollama: faithfully implemented by the proxy today. Keep alphabetised.
OLLAMA_ACTIVE=(
    api-reference/get-version.md
    api-reference/show-model-details.md
    api/chat.md
    api/copy.md
    api/create.md
    api/delete.md
    api/embed.md
    api/errors.md
    api/generate.md
    api/ps.md
    api/pull.md
    api/streaming.md
    api/tags.md
    api/usage.md
    capabilities/embeddings.md
    capabilities/structured-outputs.md
    capabilities/thinking.md
    capabilities/tool-calling.md
    capabilities/vision.md
)

# Ollama: irrelevant to a translation proxy. Never fetched.
OLLAMA_DENY=(
    api/introduction.md
    cloud.md
    docker.md
    gpu.md
    import.md
    index.md
    linux.md
    macos.md
    quickstart.md
    troubleshooting.md
    windows.md
)

# Ollama: subtree denylist (path prefixes). Never fetched.
OLLAMA_DENY_PREFIXES=(
    integrations/
)

# LM Studio: actually called as upstream by the proxy. Keep alphabetised.
LMS_ACTIVE=(
    1_developer/2_rest/download-status.md
    1_developer/2_rest/download.md
    1_developer/2_rest/list.md
    1_developer/2_rest/unload.md
    1_developer/3_openai-compat/chat-completions.md
    1_developer/3_openai-compat/models.md
    1_developer/3_openai-compat/structured-output.md
)

# LM Studio: pure noise. Never copied.
LMS_DENY=(
    1_developer/_embeddings.md
    1_developer/api-changelog.md
)

contains() {
    local needle="$1"; shift
    local item
    for item in "$@"; do
        [[ "$item" == "$needle" ]] && return 0
    done
    return 1
}

has_prefix() {
    local path="$1"; shift
    local prefix
    for prefix in "$@"; do
        [[ "$path" == "$prefix"* ]] && return 0
    done
    return 1
}

if [[ ! -d "$DOCS_DIR" ]]; then
    echo "error: '$DOCS_DIR' does not exist; run from repo containing api-docs/" >&2
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

# Drop legacy single-file dump from older script versions.
rm -f "$DOCS_DIR/ollama.md"
rm -rf "$OLLAMA_DIR" "$FUTURE_OLLAMA_DIR"
mkdir -p "$OLLAMA_DIR" "$FUTURE_OLLAMA_DIR"

echo "→ fetching ollama llms.txt"
curl -fsSL --retry 3 --retry-delay 2 "$OLLAMA_LLMS_TXT" -o "$tmpdir/llms.txt"

mapfile -t ollama_paths < <(
    grep -oE "\\(${OLLAMA_BASE}/[^)]+\\)" "$tmpdir/llms.txt" \
        | sed -E "s#^\\(${OLLAMA_BASE}/##; s#\\)\$##" \
        | sort -u
)

if [[ ${#ollama_paths[@]} -eq 0 ]]; then
    echo "error: llms.txt yielded no paths" >&2
    exit 1
fi

ollama_active=0
ollama_future=0
ollama_future_paths=()

for path in "${ollama_paths[@]}"; do
    if contains "$path" "${OLLAMA_DENY[@]}"; then
        continue
    fi
    if has_prefix "$path" "${OLLAMA_DENY_PREFIXES[@]}"; then
        continue
    fi
    if contains "$path" "${OLLAMA_ACTIVE[@]}"; then
        target="$OLLAMA_DIR/$path"
    else
        target="$FUTURE_OLLAMA_DIR/$path"
        ollama_future_paths+=("$path")
    fi
    mkdir -p "$(dirname "$target")"
    curl -fsSL --retry 3 --retry-delay 2 "$OLLAMA_BASE/$path" -o "$target"
    if [[ "$target" == "$OLLAMA_DIR/"* ]]; then
        ollama_active=$((ollama_active + 1))
    else
        ollama_future=$((ollama_future + 1))
    fi
done

echo "  ollama: $ollama_active active, $ollama_future future"

# --- LM Studio -------------------------------------------------------------

echo "→ cloning lmstudio-ai/docs (shallow)"
git clone --depth=1 --quiet "$LMS_REPO" "$tmpdir/lms-docs"

rm -rf "$LMS_DIR" "$FUTURE_LMS_DIR"
mkdir -p "$LMS_DIR" "$FUTURE_LMS_DIR"

mapfile -t lms_paths < <(
    cd "$tmpdir/lms-docs" \
        && find 1_developer -type f \( -name '*.md' -o -name '*.mdx' \) \
        | sort
)

lms_active=0
lms_future=0

for path in "${lms_paths[@]}"; do
    if contains "$path" "${LMS_DENY[@]}"; then
        continue
    fi
    src="$tmpdir/lms-docs/$path"
    if contains "$path" "${LMS_ACTIVE[@]}"; then
        target="$LMS_DIR/$path"
        lms_active=$((lms_active + 1))
    else
        target="$FUTURE_LMS_DIR/$path"
        lms_future=$((lms_future + 1))
    fi
    mkdir -p "$(dirname "$target")"
    cp "$src" "$target"
done

echo "  lmstudio: $lms_active active, $lms_future future"

if (( ollama_future > 0 || lms_future > 0 )); then
    echo "ℹ future/ contains $((ollama_future + lms_future)) doc(s) not yet implemented"
fi

echo "✓ done"
