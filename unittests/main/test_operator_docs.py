"""Tests for the operator documentation generator (onnx_light.doc)."""

import os
import tempfile
import unittest
from pathlib import Path

from onnx_light.ext_test_case import ExtTestCase
import onnx_light.doc as doc_module


class TestGenOperators(ExtTestCase):
    """Tests that generate_operators_doc generates valid RST operator documentation."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def test_generate_creates_files(self):
        """Checks that generate_operators_doc() produces at least an index and one domain file."""
        try:
            import onnx.defs  # noqa: F401
        except ImportError:
            self.skipTest("onnx package not installed")

        doc_module.generate_operators_doc(self.tmp_dir)
        files = os.listdir(self.tmp_dir)
        self.assertIn("index.rst", files, "index.rst must be generated")
        self.assertIn("ai_onnx.rst", files, "ai_onnx.rst must be generated")

    def test_index_lists_all_domains(self):
        """Checks that the index lists all generated domain stems."""
        try:
            import onnx.defs  # noqa: F401
        except ImportError:
            self.skipTest("onnx package not installed")

        doc_module.generate_operators_doc(self.tmp_dir)
        index = Path(self.tmp_dir, "index.rst").read_text(encoding="utf-8")
        self.assertIn("ai_onnx", index)
        self.assertIn("ai_onnx_ml", index)

    def test_domain_page_contains_operators(self):
        """Checks that the ai_onnx domain page contains well-known operator names."""
        try:
            import onnx.defs  # noqa: F401
        except ImportError:
            self.skipTest("onnx package not installed")

        doc_module.generate_operators_doc(self.tmp_dir)
        content = Path(self.tmp_dir, "ai_onnx.rst").read_text(encoding="utf-8")
        for name in ("Abs", "Add", "Conv", "Relu"):
            self.assertIn(name, content, f"Expected operator {name!r} in ai_onnx.rst")

    def test_domain_page_contains_anchors(self):
        """Checks that operator anchor labels are present in the domain page."""
        try:
            import onnx.defs  # noqa: F401
        except ImportError:
            self.skipTest("onnx package not installed")

        doc_module.generate_operators_doc(self.tmp_dir)
        content = Path(self.tmp_dir, "ai_onnx.rst").read_text(encoding="utf-8")
        self.assertIn(".. _op_ai_onnx_Abs:", content)
        self.assertIn(".. _op_ai_onnx_Add:", content)

    def test_operator_section_contains_inputs_outputs(self):
        """Checks that operator sections include Inputs and Outputs headings."""
        try:
            import onnx.defs  # noqa: F401
        except ImportError:
            self.skipTest("onnx package not installed")

        doc_module.generate_operators_doc(self.tmp_dir)
        content = Path(self.tmp_dir, "ai_onnx.rst").read_text(encoding="utf-8")
        self.assertIn("**Inputs**", content)
        self.assertIn("**Outputs**", content)

    def test_domain_file_stem(self):
        """Checks domain-to-filename mapping."""
        self.assertEqual(doc_module._domain_file_stem(""), "ai_onnx")
        self.assertEqual(doc_module._domain_file_stem("ai.onnx.ml"), "ai_onnx_ml")
        self.assertEqual(
            doc_module._domain_file_stem("ai.onnx.preview.training"), "ai_onnx_preview_training"
        )

    def test_main_docs_index_references_operators(self):
        """Checks that the main docs/index.rst references operators/index."""
        index_path = Path(__file__).resolve().parents[2] / "docs" / "index.rst"
        content = index_path.read_text(encoding="utf-8")
        self.assertIn("operators/index", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
