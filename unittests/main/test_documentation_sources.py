import importlib.util
import re
import unittest
from io import StringIO
from pathlib import Path


class TestDocumentationSources(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[2]

    def test_homepage_build_metadata(self):
        text = (self.ROOT / "docs/index.rst").read_text(encoding="utf-8")
        self.assertIn("Version: |release|", text)
        self.assertIn("Last generated: |today|", text)
        config = (self.ROOT / "docs/conf.py").read_text(encoding="utf-8")
        self.assertIn("release = onnx_light.__version__", config)
        self.assertIn('today_fmt = "%Y-%m-%d"', config)

    def test_internal_links_use_relative_paths(self):
        docs = self.ROOT / "docs"
        for source in sorted(docs.rglob("*")):
            if source.suffix not in {".rst", ".md"}:
                continue
            with self.subTest(source=source.relative_to(self.ROOT)):
                text = source.read_text(encoding="utf-8")
                self.assertNotRegex(
                    text, r"https://github\.com/xadupre/onnx-light/(?:blob|tree)/main/docs/"
                )
                self.assertNotIn(":epkg:`C++ onnx-light examples`", text)

    def test_relative_documentation_targets(self):
        for name in (
            "quick_tour.rst",
            "getting_started.rst",
            "design/link/cplusplus_linking.rst",
            "design/proto/differences.rst",
            "md/pattern_optimization_comparison.md",
            "misc/technical/thread_pools.rst",
        ):
            source = self.ROOT / "docs" / name
            text = source.read_text(encoding="utf-8")
            targets = re.findall(r":doc:`[^`<>]*<([^<>]+)>`", text)
            targets.extend(re.findall(r"\]\(([^)]+\.rst)\)", text))
            self.assertTrue(targets, source)
            for target in targets:
                with self.subTest(source=name, target=target):
                    self.assertFalse(Path(target).is_absolute())
                    path = source.parent / target
                    if path.suffix != ".rst":
                        path = path.with_suffix(".rst")
                    self.assertTrue(path.is_file(), path)

    def test_thread_pool_references_use_local_api(self):
        source = self.ROOT / "docs/misc/technical/thread_pools.rst"
        text = source.read_text(encoding="utf-8")
        for name in ("cpu_executor", "parallel_for"):
            self.assertIn(f"<../../api/cpp/onnx_core/runtime/{name}>`", text)
        self.assertNotIn("https://github.com/xadupre/onnx-light/", text)

    def test_source_links_preserve_github_browsing(self):
        docs = self.ROOT / "docs"
        sources = [docs / "quick_tour.rst"]
        for name in (
            "aidge_onnx_light",
            "build_save_load_onnx_proto",
            "check_onnx_light_model",
            "export_nnef",
            "load_onnx_light_time",
            "load_onnx_time",
            "print_proto_debug",
            "register_custom_kernel",
            "run_backend_test_ort",
        ):
            source = docs / "examples_cc" / f"{name}_example.rst"
            text = source.read_text(encoding="utf-8")
            self.assertIn(
                f"<https://github.com/xadupre/onnx-light/tree/main/examples/{name}>`_", text
            )
            sources.append(source)
        for source in sources:
            text = source.read_text(encoding="utf-8")
            self.assertNotIn(":download:", text)
            targets = re.findall(
                r"<https://github\.com/xadupre/onnx-light/(?:blob|tree)/main/([^<>]+)>`_", text
            )
            self.assertTrue(targets, source)
            for target in targets:
                with self.subTest(source=source.relative_to(docs), target=target):
                    self.assertFalse(Path(target).is_absolute())
                    path = (self.ROOT / target).resolve()
                    self.assertTrue(path.is_relative_to(self.ROOT))
                    self.assertTrue(path.exists(), path)

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
