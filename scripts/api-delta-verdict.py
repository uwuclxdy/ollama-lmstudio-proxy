#!/usr/bin/env python3
"""Semantic-delta verdict for the api-docs refresh.

Compares every changed file under api-docs/ against its pre-refresh blob and
classifies the delta, so the refresh PR body separates real API changes from
reformat noise:

- yaml: structural compare. Added/removed keys, enum/required list items and
  changed scalars are substantive, except scalars under pure-prose keys
  (description/summary/title/example) — those and key reordering are chrome.
- md / mdx / html snapshots: markup stripped (renderer component markers,
  blockquote prefixes, bold markers, build tokens), an html document reduced
  to its rendered <main> body (full text when it has none), then a text diff.
  Fragments made only of site-chrome strings clear as informational; anything
  else reports "text changed" for review — prose is never auto-cleared.

Prints a markdown section for the refresh PR body; prints nothing when no
api-docs/ file changed, so the body stays untouched on no-op refreshes.

Modes:
  default      worktree against HEAD (what the refresh step sees)
  --old-ref R  worktree against blob R:path (replaying a past refresh)
"""

from __future__ import annotations

import difflib
import html
import re
import subprocess
import sys

try:
    import yaml
except ImportError:  # yaml files then report "unparseable — review manually"
    yaml = None

FUTURE_PREFIX = "api-docs/future/"
MAX_DETAILS_PER_FILE = 12
MAX_CHARS = 140

# Scalars under these keys are prose or sample data; their rewording is chrome.
CHROME_KEYS = {"description", "summary", "title", "example", "examples"}
# Site furniture Mintlify injects around docs content; text fragments made only
# of these are chrome, not API change.
SITE_CHROME = {"back to top", "previous", "next", "on this page", "skip to main content"}
# Renderer components the markdown mirrors carry as literal text.
COMPONENT_MARKER = re.compile(r"\[(?:tab|card|accordion)\]")
# Blockquote prefix the renderer wraps each component body line in.
QUOTE_PREFIX = re.compile(r"(?m)^[ \t]*>[ \t]?")
# Bold label the renderer turns into a component marker.
BOLD = re.compile(r"(?<![\w*])\*\*(?=\S)(.+?)(?<=\S)\*\*(?![\w*])")
# An html snapshot: the mirror serves html pages for some paths.
HTML_DOC = re.compile(r"\s*(?:<!doctype html\b|<html\b)", re.I)
# Rendered page body; the rest of an html snapshot is site chrome. Greedy to
# the last close, so a stray "</main>" in an embedded payload or a second main
# block cannot truncate the compared body.
MAIN = re.compile(r"<main(?=[\s>]).*</main\s*>", re.S | re.I)
CHROME_TOKEN = re.compile(
    r"dpl_[A-Za-z0-9]+"                      # deploy ids
    r"|\b[0-9a-f]{16,}\b"                    # build hashes
    r"|index-[A-Za-z0-9_]{6,}\.(?:js|css)"   # asset bundle names
    r'|clientVersion":"[^"]*"'
)
TAGS = re.compile(r"<script\b.*?</script>|<style\b.*?</style>|<!--.*?-->|<[^>]+>", re.S | re.I)
WHITESPACE = re.compile(r"\s+")


def git(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], capture_output=True)


def blob(ref: str, path: str) -> str | None:
    r = git("show", f"{ref}:{path}")
    return r.stdout.decode("utf-8", "replace") if r.returncode == 0 else None


def worktree(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return None


def visible_text(text: str) -> str:
    text = QUOTE_PREFIX.sub("", text)
    text = BOLD.sub(r"\1", text)
    return WHITESPACE.sub(" ", html.unescape(TAGS.sub(" ", text))).strip()


def text_fragments(old: str, new: str) -> list[str]:
    to = CHROME_TOKEN.sub(" ", COMPONENT_MARKER.sub(" ", visible_text(old)))
    tn = CHROME_TOKEN.sub(" ", COMPONENT_MARKER.sub(" ", visible_text(new)))
    sm = difflib.SequenceMatcher(None, to, tn, autojunk=False)
    frags = []
    for op, a1, a2, b1, b2 in sm.get_opcodes():
        if op == "equal":
            continue
        piece = ("-" + to[a1:a2] if op in ("delete", "replace") else "") + (
            "+" + tn[b1:b2] if op in ("insert", "replace") else ""
        )
        if piece.replace("-", "").replace("+", "").strip():
            frags.append(piece.strip())
    return frags


def summary(value: object) -> str:
    if isinstance(value, dict):
        keys = ", ".join(sorted(value, key=str)[:8])
        more = " …" if len(value) > 8 else ""
        return "{ " + keys + more + " }"
    if isinstance(value, list):
        return f"[{len(value)} items]"
    text = repr(value)
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + "…"


def walk(old: object, new: object, path: str, out: list[tuple[str, str, str, str]]) -> None:
    if type(old) is not type(new):
        out.append(("type", path, summary(old), summary(new)))
        return
    if isinstance(old, dict):
        for key in sorted(old.keys() | new.keys(), key=str):
            child = f"{path}.{key}"
            if key not in new:
                out.append(("removed key", child, summary(old[key]), ""))
            elif key not in old:
                out.append(("added key", child, "", summary(new[key])))
            elif isinstance(old[key], (dict, list)):
                walk(old[key], new[key], child, out)
            elif old[key] != new[key] and key not in CHROME_KEYS:
                out.append(("changed", child, summary(old[key]), summary(new[key])))
    elif isinstance(old, list):
        olds = [repr(item) for item in old]
        news = [repr(item) for item in new]
        if sorted(olds) == sorted(news):
            return  # same items, reordered: reformat noise
        for text in olds:
            if text not in news:
                out.append(("removed item", path, text[:MAX_CHARS], ""))
        for text in news:
            if text not in olds:
                out.append(("added item", path, "", text[:MAX_CHARS]))


def rendered_main(text: str) -> str:
    if not HTML_DOC.match(text):
        return text
    body = MAIN.search(text)
    return body.group(0) if body else text


def compare_file(path: str, old_text: str, new_text: str) -> tuple[str, list[str]]:
    """Verdict + detail lines for one file's old/new contents."""
    if path.endswith((".yaml", ".yml")):
        if yaml is None:
            return "yaml unparseable (no pyyaml) — review manually", []
        try:
            old_tree = yaml.safe_load(old_text)
            new_tree = yaml.safe_load(new_text)
        except yaml.YAMLError as exc:
            return f"yaml unparseable — review manually ({str(exc)[:MAX_CHARS]})", []
        out: list[tuple[str, str, str, str]] = []
        walk(old_tree, new_tree, path, out)
        if not out:
            return "structural no-change (formatting only)", []
        return f"substantive ({len(out)} delta(s))", [_render(*entry) for entry in out]

    if not path.lower().endswith((".md", ".mdx")):
        old_text, new_text = rendered_main(old_text), rendered_main(new_text)
    frags = text_fragments(old_text, new_text)
    if not frags:
        return "no text change", []
    remaining = [frag for frag in frags if frag.lstrip("+-").strip().casefold() not in SITE_CHROME]
    if not remaining:
        return "site-chrome only", [frag[:MAX_CHARS] for frag in frags]
    shown = remaining[:MAX_DETAILS_PER_FILE]
    more = f" … {len(remaining) - len(shown)} more" if len(remaining) > len(shown) else ""
    return f"text changed ({len(remaining)} fragment(s)) — review{more}", [
        frag[:MAX_CHARS] for frag in shown
    ]


def _render(kind: str, path: str, old: str, new: str) -> str:
    where = path.removeprefix("api-docs/")
    arrow = ""
    if old and new:
        arrow = f" `{old}` → `{new}`"
    elif old or new:
        arrow = f" `{old or new}`"
    return f"{kind} `{where}`{arrow}"


def main() -> int:
    argv = sys.argv[1:]
    old_ref = argv[argv.index("--old-ref") + 1] if "--old-ref" in argv else "HEAD"

    status = git("diff", "--name-status", old_ref, "--", "api-docs/")
    if status.returncode != 0:
        print(f"git diff failed: {status.stderr.decode(errors='replace')[:200]}", file=sys.stderr)
        return 1
    rows = [line.split("\t") for line in status.stdout.decode().splitlines() if line.strip()]
    if not rows:
        return 0

    table_rows: list[str] = []
    details: list[str] = []
    substantive_active = substantive_future = review_files = 0
    for row in rows:
        state, path = row[0][:1], row[-1]
        if state == "A":
            verdict, file_details = "new file — review as substantive", []
        elif state == "D":
            verdict, file_details = "file removed", []
        else:
            verdict, file_details = compare_file(path, blob(old_ref, path) or "", worktree(path) or "")
        scope = "future" if path.startswith(FUTURE_PREFIX) else "active"
        is_substantive = (
            "substantive" in verdict
            or verdict.startswith("new file")
            or verdict == "file removed"
            or verdict.startswith("yaml unparseable")
        )
        if is_substantive:
            substantive_active += scope == "active"
            substantive_future += scope == "future"
        elif verdict.startswith("text changed"):
            review_files += 1
        table_rows.append(f"| `{path}` | {scope} | {verdict} |")
        for line in file_details[:MAX_DETAILS_PER_FILE]:
            details.append(f"  - {line}")
        if len(file_details) > MAX_DETAILS_PER_FILE:
            details.append(f"  - … {len(file_details) - MAX_DETAILS_PER_FILE} more")

    if substantive_active:
        top = f"**verdict: SUBSTANTIVE — {substantive_active} active-file delta(s); patch the matching handler before merging.**"
    elif substantive_future:
        top = f"**verdict: future-only — {substantive_future} future-mirror delta(s); nothing the proxy implements moved.**"
    elif review_files:
        top = f"**verdict: review — text changed in {review_files} file(s) with no auto-classifiable delta.**"
    else:
        top = "**verdict: chrome-only — reformats and build tokens only, nothing to implement.**"

    print("### semantic delta")
    print()
    print(top)
    print()
    print("| doc | scope | verdict |")
    print("|---|---|---|")
    for line in table_rows:
        print(line)
    if details:
        print()
        for line in details:
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
