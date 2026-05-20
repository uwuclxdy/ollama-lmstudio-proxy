#!/usr/bin/env bash
#
# Refresh the upstream-mirrored API docs under "api_docs/".
#
#   - "api_docs/ollama/<tree>/"        ← https://docs.ollama.com/<tree>/
#         tree ∈ {api, api-reference, capabilities} + a few top-level pages
#   - "api_docs/lmstudio/<tree>/"      ← lmstudio-ai/docs@main:<tree>/
#         tree ∈ {1_developer}
#
# The ollama whitelist drops integrations/* and platform installers (linux,
# macos, windows, docker, gpu, import, quickstart, troubleshooting, cloud)
# which are irrelevant to a proxy.
#
# Local-only files under "api_docs/" (lmstudio_ollama_openai.md,
# lmstudio_vs_ollama.md, and anything outside the mirrored trees) are
# never touched.
#
# Re-run safely; the script wipes and rebuilds only the mirrored trees.
set -euo pipefail

OLLAMA_BASE="https://docs.ollama.com"
OLLAMA_LLMS_TXT="$OLLAMA_BASE/llms.txt"
LMS_REPO="https://github.com/lmstudio-ai/docs.git"
LMS_TREES=(1_developer)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCS_DIR="$REPO_ROOT/api_docs"
OLLAMA_DIR="$DOCS_DIR/ollama"
LMS_DIR="$DOCS_DIR/lmstudio"

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

echo "→ fetching ollama llms.txt"
curl -fsSL --retry 3 --retry-delay 2 "$OLLAMA_LLMS_TXT" -o "$tmpdir/llms.txt"

# Whitelist matches the path portion of each docs.ollama.com URL in llms.txt.
keep_re='^(api/|api-reference/|capabilities/|modelfile\.md$|cli\.md$|context-length\.md$|faq\.md$|index\.md$)'

mapfile -t paths < <(
    grep -oE "\\(${OLLAMA_BASE}/[^)]+\\)" "$tmpdir/llms.txt" \
        | sed -E "s#^\\(${OLLAMA_BASE}/##; s#\\)\$##" \
        | grep -E "$keep_re" \
        | sort -u
)

if [[ ${#paths[@]} -eq 0 ]]; then
    echo "error: llms.txt yielded no matching paths" >&2
    exit 1
fi

echo "→ fetching ${#paths[@]} ollama pages"
for path in "${paths[@]}"; do
    target="$OLLAMA_DIR/$path"
    mkdir -p "$(dirname "$target")"
    curl -fsSL --retry 3 --retry-delay 2 "$OLLAMA_BASE/$path" -o "$target"
done

# --- LM Studio -------------------------------------------------------------

echo "→ cloning lmstudio-ai/docs (shallow)"
git clone --depth=1 --quiet "$LMS_REPO" "$tmpdir/lms-docs"

mkdir -p "$LMS_DIR"
for tree in "${LMS_TREES[@]}"; do
    target="$LMS_DIR/$tree"
    source="$tmpdir/lms-docs/$tree"
    if [[ ! -d "$source" ]]; then
        echo "warning: upstream tree '$tree' not found; skipping" >&2
        continue
    fi
    echo "→ syncing $tree"
    rm -rf "$target"
    cp -R "$source" "$target"
    # Drop upstream nav cruft (Mintlify meta.json, lockfiles, etc.). Keep
    # markdown only — that is what the proxy references.
    find "$target" -type f ! \( -name '*.md' -o -name '*.mdx' \) -delete
    find "$target" -type d -empty -delete
done

echo "✓ done"
