"""Regression corpus for api-delta-verdict.py.

Each case pins one verdict the classifier must return: a chrome-only rewrite of
the api-docs mirror clears, any prose, identifier or constraint change reviews.
"""

import importlib.util
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
TOOL = HERE / "api-delta-verdict.py"

_SPEC = importlib.util.spec_from_file_location("api_delta_verdict", TOOL)
verdict = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(verdict)

CLEAR = ("no text change", "site-chrome only", "structural no-change (formatting only)")
REVIEW = ("text changed", "substantive", "yaml unparseable")

MD = "api-docs/ollama/capabilities/x.md"
HTML = "api-docs/future/ollama/capabilities/x"

TEXT_CASES = [
    (
        "tab marker replaces a bold label",
        MD,
        "**cURL**\n\n```shell\ntrue\n```\n",
        "> [tab] cURL\n>\n> ```shell\n> true\n> ```\n",
        True,
    ),
    (
        "accordion marker replaces a bold label",
        MD,
        "> **Legacy plans**\n>\n> text\n",
        "> [accordion] Legacy plans\n>\n> text\n",
        True,
    ),
    (
        "card marker replaces a bold label",
        MD,
        "**Model library**\n",
        "> [card] Model library\n",
        True,
    ),
    (
        "bold pair removed without a marker",
        MD,
        "**Request body**\n",
        "Request body\n",
        True,
    ),
    ("near-miss marker [tabs] reviews", MD, "cURL\n", "[tabs] cURL\n", False),
    ("uppercase marker [TAB] reviews", MD, "cURL\n", "[TAB] cURL\n", False),
    ("spaced marker [ card ] reviews", MD, "cURL\n", "[ card ] cURL\n", False),
    (
        "prose change inside a tab reviews",
        MD,
        "> [tab] cURL\n>\n> stream false\n",
        "> [tab] cURL\n>\n> stream true\n",
        False,
    ),
    (
        "code identifier change inside a tab reviews",
        MD,
        '> [tab] cURL\n>\n> "think": true\n',
        '> [tab] cURL\n>\n> "think": false\n',
        False,
    ),
    (
        "line wrapped in a blockquote is chrome",
        MD,
        "curl -s http://localhost:11434/api/chat\n",
        "> curl -s http://localhost:11434/api/chat\n",
        True,
    ),
    (
        "nested quote prose change reviews",
        MD,
        "> > original quote\n",
        "> > changed quote\n",
        False,
    ),
    ("nested quote depth change reviews", MD, "> > foo bar\n", "> foo bar\n", False),
    (
        "shell append operator change reviews",
        MD,
        "> [tab] shell\n>\n> > out.txt\n",
        "> [tab] shell\n>\n> >> out.txt\n",
        False,
    ),
    ("multiplication text change reviews", MD, "2*3*4\n", "2*34\n", False),
    ("python kwargs change reviews", MD, "f(**kwargs)\n", "f(**kw)\n", False),
    ("digit change reviews", MD, "status 200\n", "status 201\n", False),
    (
        "url change reviews",
        MD,
        "https://example.com/a\n",
        "https://example.com/b\n",
        False,
    ),
    (
        "manifest flag churn with the same body is chrome",
        HTML,
        "<!doctype html><main><p>body</p></main>"
        '<template data-mint="manifest">{"toggles":[{"name":"a","enabled":true}]}</template>',
        "<!doctype html><main><p>body</p></main>"
        '<template data-mint="manifest">{"toggles":[{"name":"b","enabled":false}]}</template>',
        True,
    ),
    (
        "manifest source drift with an unchanged body is chrome",
        HTML,
        "<!doctype html><main><p>body</p></main>"
        '<template data-mint="manifest">{"compiledSource":"alpha"}</template>',
        "<!doctype html><main><p>body</p></main>"
        '<template data-mint="manifest">{"compiledSource":"beta"}</template>',
        True,
    ),
    (
        "rendered main content change reviews",
        HTML,
        "<!doctype html><main><p>body text</p></main>",
        "<!doctype html><main><p>changed text</p></main>",
        False,
    ),
    (
        "html without a main diffs the full text",
        HTML,
        '<!doctype html><template id="page">alpha</template>\n',
        '<!doctype html><template id="page">beta</template>\n',
        False,
    ),
    (
        "html custom element main-content is not main",
        HTML,
        "<!doctype html><main-content>alpha</main-content><main><p>body</p></main>",
        "<!doctype html><main-content>beta</main-content><main><p>body</p></main>",
        True,
    ),
    (
        "html starting at <html> gets main extraction",
        HTML,
        "<html><main><p>body</p></main>"
        '<template data-mint="manifest">{"toggles":[{"name":"a"}]}</template>',
        "<html><main><p>body</p></main>"
        '<template data-mint="manifest">{"toggles":[{"name":"b"}]}</template>',
        True,
    ),
    (
        "a script payload holding </main> cannot truncate the body",
        HTML,
        '<!doctype html><main><script>var x="</main>"</script><p>alpha</p></main>',
        '<!doctype html><main><script>var x="</main>"</script><p>beta</p></main>',
        False,
    ),
    (
        "a second main block is compared",
        HTML,
        "<!doctype html><main><p>one</p></main><main><p>alpha</p></main>",
        "<!doctype html><main><p>one</p></main><main><p>beta</p></main>",
        False,
    ),
    (
        "component swap keeps the body is chrome",
        MD,
        "> [tab] cURL\n>\n> true\n",
        "> [card] cURL\n>\n> true\n",
        True,
    ),
    (
        "quote wrap only is chrome",
        MD,
        "curl -s http://localhost:11434/api/tags\n",
        "> curl -s http://localhost:11434/api/tags\n",
        True,
    ),
    (
        "md page with a main code sample diffs the full text",
        MD,
        "example:\n\n```html\n<main>alpha</main>\n```\n\nprose one\n",
        "example:\n\n```html\n<main>alpha</main>\n```\n\nprose two\n",
        False,
    ),
    (
        "md page starting with a doctype keeps the full text",
        MD,
        "<!doctype html><main>alpha</main>\nprose one\n",
        "<!doctype html><main>alpha</main>\nprose two\n",
        False,
    ),
    ("site-chrome word removal is chrome", MD, "previous next\n", "previous\n", True),
    (
        "deploy id churn is chrome",
        MD,
        'href="/x.js?dpl=dpl_EnxDqhjaNc9a8MVa833TGULPd9uJ"\n',
        'href="/x.js?dpl=dpl_G9nrg4qeJSZtah1Ziwbc2n6WkLch"\n',
        True,
    ),
    (
        "build hash churn is chrome",
        MD,
        "hash 0123456789abcdef\n",
        "hash fedcba9876543210\n",
        True,
    ),
    (
        "asset bundle name churn is chrome",
        MD,
        'src="/_mintlify/assets/index-BppZ3guj.js"\n',
        'src="/_mintlify/assets/index-AbCdEf12.js"\n',
        True,
    ),
    (
        "client version churn is chrome",
        MD,
        '{"clientVersion":"1.2.3"}\n',
        '{"clientVersion":"1.2.4"}\n',
        True,
    ),
]

YAML_CASES = [
    (
        "yaml formatting-only is clear",
        "required: [model]\n",
        "required:\n  - model\n",
        True,
    ),
    (
        "yaml required list change reviews",
        "required: [model]\n",
        "required: [model, stream]\n",
        False,
    ),
    ("yaml added key reviews", "model: x\n", "model: x\nstream: true\n", False),
    ("yaml removed key reviews", "model: x\nstream: true\n", "model: x\n", False),
    ("yaml changed scalar reviews", "model: x\n", "model: y\n", False),
    ("yaml type change reviews", "model: x\n", "model: 1\n", False),
    ("yaml list item added reviews", "stops: [a]\n", "stops: [a, b]\n", False),
    (
        "yaml chrome-key rewording is clear",
        'description: "a b"\n',
        'description: "a b c"\n',
        True,
    ),
    ("yaml key reorder is clear", "a: 1\nb: 2\n", "b: 2\na: 1\n", True),
    ("yaml unparseable reviews", "a: [\n", "a: {\n", False),
]


class CompareFileTest(unittest.TestCase):
    def check(self, name, path, before, after, clear):
        got, _details = verdict.compare_file(path, before, after)
        with self.subTest(case=name):
            if clear:
                self.assertIn(got, CLEAR)
            else:
                self.assertTrue(got.startswith(REVIEW), got)

    def test_text_cases(self):
        for name, path, before, after, clear in TEXT_CASES:
            self.check(name, path, before, after, clear)

    @unittest.skipUnless(verdict.yaml is not None, "pyyaml not installed")
    def test_yaml_cases(self):
        for name, before, after, clear in YAML_CASES:
            self.check(name, "api-docs/ollama/openapi.yaml", before, after, clear)


@unittest.skipUnless(shutil.which("git"), "git not installed")
class EntryPointTest(unittest.TestCase):
    def setUp(self):
        self.repo = tempfile.TemporaryDirectory()
        self.addCleanup(self.repo.cleanup)
        root = pathlib.Path(self.repo.name)
        (root / "api-docs").mkdir()
        self.doc = root / "api-docs/x.md"
        self.doc.write_text("status 200\n")
        self.git(root, "init", "-q")
        self.git(root, "add", "api-docs/x.md")
        self.git(
            root,
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-q",
            "--no-verify",
            "-m",
            "test: init fixture",
        )

    def git(self, root, *args):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)

    def run_tool(self):
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        return subprocess.run(
            [sys.executable, str(TOOL), "--old-ref", "HEAD"],
            cwd=self.repo.name,
            capture_output=True,
            text=True,
            env=env,
        )

    def test_silent_when_nothing_changed(self):
        done = self.run_tool()
        self.assertEqual(done.returncode, 0)
        self.assertEqual(done.stdout, "")

    def test_renderer_churn_prints_chrome_only(self):
        self.doc.write_text("**status 200**\n")
        done = self.run_tool()
        self.assertEqual(done.returncode, 0)
        self.assertIn("verdict: chrome-only", done.stdout)
        self.assertIn("| `api-docs/x.md` | active |", done.stdout)

    def test_prose_change_prints_review(self):
        self.doc.write_text("status 201\n")
        done = self.run_tool()
        self.assertEqual(done.returncode, 0)
        self.assertIn("verdict: review", done.stdout)

    def test_new_file_prints_substantive(self):
        (pathlib.Path(self.repo.name) / "api-docs/new.md").write_text("stream true\n")
        self.git(self.repo.name, "add", "api-docs/new.md")
        done = self.run_tool()
        self.assertEqual(done.returncode, 0)
        self.assertIn("verdict: SUBSTANTIVE", done.stdout)
        self.assertIn("new file — review as substantive", done.stdout)

    def test_removed_file_prints_substantive(self):
        self.doc.unlink()
        done = self.run_tool()
        self.assertEqual(done.returncode, 0)
        self.assertIn("verdict: SUBSTANTIVE", done.stdout)
        self.assertIn("file removed", done.stdout)

    def test_future_scope_is_its_own_verdict(self):
        future = pathlib.Path(self.repo.name) / "api-docs/future/y.md"
        future.parent.mkdir()
        future.write_text("example page\n")
        self.git(self.repo.name, "add", "api-docs/future/y.md")
        done = self.run_tool()
        self.assertEqual(done.returncode, 0)
        self.assertIn("verdict: future-only", done.stdout)
        self.assertIn("| `api-docs/future/y.md` | future |", done.stdout)


if __name__ == "__main__":
    unittest.main()
