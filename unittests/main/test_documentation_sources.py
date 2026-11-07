import importlib.util
import unittest
from io import StringIO
from pathlib import Path


class TestDocumentationSources(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[2]

    def test_api_heading_underlines(self):
        for source, title, adornment in [
            (
                "docs/api/python/onnx_core/graph_builder.rst",
                "Builder-local operator registries",
                "-",
            ),
            ("docs/api/python/tools/kernel_reports.rst", "onnx_light.tools.kernel_reports", "="),
        ]:
            with self.subTest(source=source):
                lines = (self.ROOT / source).read_text(encoding="utf-8").splitlines()
                underline = lines[lines.index(title) + 1]
                self.assertEqual(set(underline), {adornment})
                self.assertGreaterEqual(len(underline), len(title))

    @unittest.skipUnless(importlib.util.find_spec("docutils"), "docutils is not installed")
    def test_pattern_progress_table(self):
        from docutils import nodes
        from docutils.core import publish_doctree

        path = self.ROOT / "docs/next_steps/2026/2026-08_graph_builder_optimization.rst"
        lines = path.read_text(encoding="utf-8").splitlines()
        start = lines.index(".. list-table::")
        end = next(
            index
            for index in range(start + 1, len(lines))
            if lines[index] and not lines[index].startswith(" ")
        )
        warnings = StringIO()
        document = publish_doctree(
            "\n".join(lines[start:end]),
            settings_overrides={"warning_stream": warnings, "halt_level": 2},
        )
        self.assertEqual(warnings.getvalue(), "")
        tables = list(document.findall(nodes.table))
        self.assertEqual(len(tables), 1)
        for row in tables[0].findall(nodes.row):
            self.assertEqual(len(row.children), 3)

    def test_proto_shim_does_not_duplicate_imported_members(self):
        text = (self.ROOT / "docs/api/python/onnx/onnx_pb.rst").read_text(encoding="utf-8")
        self.assertIn(".. automodule:: onnx_light.onnx.onnx_pb", text)
        self.assertNotIn(":imported-members:", text)
        self.assertIn(":doc:`../../protos/index`", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
