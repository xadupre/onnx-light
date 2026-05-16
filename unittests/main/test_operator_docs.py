"""Tests for the operator documentation generator (onnx_light.doc)."""

import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from onnx_light.ext_test_case import ExtTestCase
import onnx_light.doc as doc_module


class TestGenOperators(ExtTestCase):
    """Tests that generate_operators_doc generates valid RST operator documentation."""

    def _init(self, clean=False):
        from onnx_light.onnx.defs import register_onnx_operator_set_schema

        folder = self.get_dump_folder("test_gen_operators", clean=clean)
        register_onnx_operator_set_schema()
        doc_module.generate_operators_doc(folder)
        self.tmp_dir = folder

    def test_a_start(self):
        self._init(True)

    def test_generate_creates_files(self):
        self._init()
        files = os.listdir(self.tmp_dir)
        self.assertIn("index.rst", files, "index.rst must be generated")
        self.assertIn("ai_onnx.rst", files, "ai_onnx.rst must be generated")

    def test_index_lists_all_domains(self):
        self._init()
        index = Path(self.tmp_dir, "index.rst").read_text(encoding="utf-8")
        self.assertIn("ai_onnx", index)
        self.assertIn("ai_onnx_ml", index)

    def test_domain_page_contains_operators(self):
        self._init()
        content = Path(self.tmp_dir, "ai_onnx.rst").read_text(encoding="utf-8")
        for name in ("Abs", "Add", "Conv", "Relu"):
            self.assertIn(name, content, f"Expected operator {name!r} in ai_onnx.rst")

    def test_domain_page_contains_anchors(self):
        self._init()
        content = Path(self.tmp_dir, "ai_onnx", "Abs.rst").read_text(encoding="utf-8")
        self.assertIn(".. _op_ai_onnx_Abs:", content)
        content = Path(self.tmp_dir, "ai_onnx", "Add.rst").read_text(encoding="utf-8")
        self.assertIn(".. _op_ai_onnx_Add:", content)

    def test_operator_section_contains_inputs_outputs(self):
        self._init()
        # Check that operator pages include domain and version metadata
        content = Path(self.tmp_dir, "ai_onnx", "Add.rst").read_text(encoding="utf-8")
        self.assertIn("**Domain**", content)
        self.assertIn("**Since version**", content)

    def test_individual_operator_pages_created(self):
        self._init()
        op_dir = Path(self.tmp_dir, "ai_onnx")
        self.assertTrue(op_dir.is_dir(), "ai_onnx/ subdirectory must exist")
        for name in ("Abs", "Add", "Conv", "Relu"):
            op_file = op_dir / f"{name}.rst"
            self.assertTrue(op_file.exists(), f"Individual page {name}.rst must exist")

    def test_domain_file_stem(self):
        self._init()
        self.assertEqual(doc_module._domain_file_stem(""), "ai_onnx")
        self.assertEqual(doc_module._domain_file_stem("ai.onnx.ml"), "ai_onnx_ml")
        self.assertEqual(
            doc_module._domain_file_stem("ai.onnx.preview.training"), "ai_onnx_preview_training"
        )

    def test_main_docs_index_references_operators(self):
        self._init()
        index_path = Path(__file__).resolve().parents[2] / "docs" / "index.rst"
        content = index_path.read_text(encoding="utf-8")
        self.assertIn("operators/index", content)

    def test_generate_prefers_onnx_defs_when_available(self):
        class _FakeFormalParameter:
            def __init__(self, name, type_str, description):
                self.name = name
                self.type_str = type_str
                self.description = description
                self.option = "Single"

        class _FakeAttribute:
            def __init__(self, attr_type, description):
                self.type = attr_type
                self.description = description

        class _FakeConstraint:
            def __init__(self, type_param_str, description, allowed_type_strs):
                self.type_param_str = type_param_str
                self.description = description
                self.allowed_type_strs = allowed_type_strs

        class _FakeSchema:
            def __init__(self):
                self.name = "FakeOp"
                self.domain = ""
                self.since_version = 1
                self.deprecated = False
                self.doc = "Fake operator documentation."
                self.inputs = [_FakeFormalParameter("X", "tensor(float)", "Input tensor.")]
                self.outputs = [_FakeFormalParameter("Y", "tensor(float)", "Output tensor.")]
                self.attributes = {"alpha": _FakeAttribute(1, "Attribute alpha.")}
                self.type_constraints = [
                    _FakeConstraint(
                        "T", "Constrain tensor type.", ["tensor(float)", "tensor(double)"]
                    )
                ]

        class _FakeDefs:
            @staticmethod
            def get_all_schemas():
                return [_FakeSchema()]

            @staticmethod
            def get_all_schemas_with_history():
                return [_FakeSchema()]

        folder = self.get_dump_folder("test_gen_operators_with_fake_onnx_defs", clean=True)
        fake_onnx = types.ModuleType("onnx")
        fake_onnx.defs = _FakeDefs
        with patch.dict(sys.modules, {"onnx": fake_onnx}, clear=False):
            doc_module.generate_operators_doc(folder)

        content = Path(folder, "ai_onnx", "FakeOp.rst").read_text(encoding="utf-8")
        self.assertIn("Fake operator documentation.", content)
        self.assertIn("**Inputs**", content)
        self.assertIn("**Outputs**", content)
        self.assertIn("**Attributes**", content)
        self.assertIn("**Type Constraints**", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
