import ast
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import re
import textwrap
import unittest

from onnx_light.__main__ import _build_parser
from onnx_light.ext_test_case import ExtTestCase


class TestCommandLineDocs(ExtTestCase):
    """Tests API command-line coverage and executable help snippets."""

    ROOT = Path(__file__).resolve().parents[2]
    DOCS = ROOT / "docs/api/command_lines"

    def test_command_pages_are_linked(self):
        api_index = (self.ROOT / "docs/api/index.rst").read_text(encoding="utf-8")
        self.assertIn("    command_lines/index\n", api_index)
        index = (self.DOCS / "index.rst").read_text(encoding="utf-8")
        commands = _build_parser()._subparsers._group_actions[0].choices
        modules = {
            path.stem
            for path in (self.ROOT / "onnx_light").rglob("*.py")
            if 'if __name__ == "__main__":' in path.read_text(encoding="utf-8")
        }
        modules.discard("__main__")
        for name in {"onnx_light", *commands, *modules}:
            with self.subTest(command=name):
                self.assertIn(f"    {name}\n", index)
                page = self.DOCS / f"{name}.rst"
                self.assertTrue(page.is_file(), str(page))
                self.assertIn("Command-line reference", page.read_text(encoding="utf-8"))

    def test_command_summary_table(self):
        index = (self.DOCS / "index.rst").read_text(encoding="utf-8")
        self.assertIn(".. list-table::", index)
        table = index.split(".. list-table::", 1)[1].split(".. toctree::", 1)[0]
        self.assertIn(":header-rows: 1", table)
        rows = re.findall(r"    \* - :doc:`([^`]+) <([^>]+)>`\n      - (\S[^\n]*)", table)
        pages = {path.stem for path in self.DOCS.glob("*.rst")} - {"index"}
        self.assertEqual({name for _, name, _ in rows}, pages)
        self.assertEqual(len(rows), len(pages))
        for invocation, name, summary in rows:
            with self.subTest(command=name):
                page = (self.DOCS / f"{name}.rst").read_text(encoding="utf-8")
                self.assertIn(invocation, page)
                self.assertTrue(summary.endswith("."), summary)

    def test_help_snippets(self):
        parser = _build_parser()
        commands = parser._subparsers._group_actions[0].choices
        for name, command_parser in {"onnx_light": parser, **commands}.items():
            with self.subTest(command=name):
                text = (self.DOCS / f"{name}.rst").read_text(encoding="utf-8")
                snippets = re.findall(r"\.\. runpython::\n((?:\n|[ \t]+[^\n]*\n?)*)", text)
                parsers = [command_parser]
                if name != "onnx_light" and command_parser._subparsers is not None:
                    parsers.extend(command_parser._subparsers._group_actions[0].choices.values())
                self.assertEqual(len(snippets), len(parsers))
                for snippet, help_parser in zip(snippets, parsers):
                    code = textwrap.dedent(snippet).strip()
                    output = StringIO()
                    with redirect_stdout(output):
                        exec(compile(code, str(self.DOCS / f"{name}.rst"), "exec"), {})
                    self.assertEqual(output.getvalue(), help_parser.format_help() + "\n")
                invocation = "python -m onnx_light"
                if name != "onnx_light":
                    invocation += f" {name}"
                self.assertIn(invocation, text)

    def test_kernel_reports_options(self):
        module = self.ROOT / "onnx_light/tools/kernel_reports.py"
        tree = ast.parse(module.read_text(encoding="utf-8"))
        main = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        text = (self.DOCS / "kernel_reports.rst").read_text(encoding="utf-8")
        self.assertIn("python -m onnx_light.tools.kernel_reports", text)
        for node in ast.walk(main):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
            ):
                continue
            with self.subTest(argument=ast.literal_eval(node.args[0])):
                for argument in node.args:
                    self.assertIn(f"``{ast.literal_eval(argument)}", text)
                for keyword in node.keywords:
                    if keyword.arg == "default":
                        self.assertIn(f"``{ast.literal_eval(keyword.value)}``", text)
                    elif keyword.arg == "choices":
                        for choice in ast.literal_eval(keyword.value):
                            self.assertIn(f"``{choice}``", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
