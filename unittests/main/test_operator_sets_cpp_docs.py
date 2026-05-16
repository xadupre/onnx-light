import unittest
from pathlib import Path


class TestOperatorSetsCppDocs(unittest.TestCase):
    def test_operator_set_headers_have_file_level_doxygen_documentation(self):
        repo = Path(__file__).resolve().parents[2]
        defs_dir = repo / "onnx_light" / "onnx" / "defs"

        expectations = {
            "operator_sets.h": (
                "@brief Declares ai.onnx operator-set schema forward declarations and registrars."
            ),
            "operator_sets_ml.h": (
                "@brief Declares ai.onnx.ml operator-set schema declarations and registrars."
            ),
            "operator_sets_preview.h": (
                "@brief Declares ai.onnx.preview operator-set schema declarations and registrars."
            ),
            "operator_sets_training.h": (
                "@brief Declares ai.onnx.training operator-set schema registration helpers."
            ),
        }

        for filename, brief_line in expectations.items():
            content = (defs_dir / filename).read_text(encoding="utf-8")
            self.assertIn(f"@file {filename}", content)
            self.assertIn(brief_line, content)

    def test_operator_set_rst_pages_have_intro_content(self):
        repo = Path(__file__).resolve().parents[2]
        docs_dir = repo / "docs" / "api" / "cpp" / "onnx" / "defs"
        expectations = {
            "operator_sets.rst": (
                "Core ai.onnx operator-set declarations and registration helpers",
                ":cpp:func:`onnx::RegisterOnnxOperatorSetSchema`",
            ),
            "operator_sets_ml.rst": (
                "ai.onnx.ml operator-set declarations and registration entry points",
                ":cpp:func:`onnx::RegisterOnnxMLOperatorSetSchema`",
            ),
            "operator_sets_preview.rst": (
                "ai.onnx.preview operator declarations and registration helpers",
                ":cpp:func:`onnx::RegisterOnnxPreviewOperatorSetSchema`",
            ),
            "operator_sets_training.rst": (
                "ai.onnx.training operator-set declarations and registration helpers",
                ":cpp:func:`onnx::RegisterOnnxTrainingOperatorSetSchema`",
            ),
        }

        for filename, snippets in expectations.items():
            content = (docs_dir / filename).read_text(encoding="utf-8")
            for snippet in snippets:
                self.assertIn(snippet, content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
