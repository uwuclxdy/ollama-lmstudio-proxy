#!/usr/bin/env bash
#
# Refresh the upstream-mirrored API docs under "api docs/".
#
#   - "api docs/ollama.md"           ← ollama/ollama@main:docs/api.md
#   - "api docs/lm studio/<tree>/"   ← lmstudio-ai/docs@main:<tree>/
#         tree ∈ {1_developer, 1_python, 2_typescript, 3_cli}
#
# Local-only files under "api docs/" (lmstudio_ollama_openai.md,
# lmstudio_vs_ollama.md, and anything outside the mirrored trees) are
# never touched.
#
# Re-run safely; the script wipes and rebuilds only the mirrored trees.
set -euo pipefail

OLLAMA_URL="https://raw.githubusercontent.com/ollama/ollama/main/docs/api.md"
LMS_REPO="https://github.com/lmstudio-ai/docs.git"
LMS_TREES=(1_developer 1_python 2_typescript 3_cli)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCS_DIR="$REPO_ROOT/api docs"
LMS_DIR="$DOCS_DIR/lm studio"

if [[ ! -d "$DOCS_DIR" ]]; then
    echo "error: '$DOCS_DIR' does not exist; run from repo containing the api docs/ directory" >&2
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

echo "→ fetching ollama api.md"
curl -fsSL --retry 3 --retry-delay 2 "$OLLAMA_URL" -o "$DOCS_DIR/ollama.md"

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
