import unittest
from pathlib import Path


class TestOperatorSetsImport(unittest.TestCase):
    def test_operator_sets_files_imported(self):
        """Verifies that operator_sets header files are vendored."""
        defs = Path(__file__).resolve().parents[2] / "onnx_light" / "onnx" / "defs"

        expected = {
            "operator_sets.h",
            "operator_sets_ml.h",
            "operator_sets_preview.h",
            "operator_sets_training.h",
        }
        present = {path.name for path in defs.glob("*") if path.is_file()}
        self.assertTrue(expected.issubset(present), msg=f"missing={sorted(expected - present)}")

    def test_operator_sets_use_light_namespace(self):
        """Verifies that operator_sets headers use ONNX_LIGHT_NAMESPACE."""
        defs = Path(__file__).resolve().parents[2] / "onnx_light" / "onnx" / "defs"

        for name in (
            "operator_sets.h",
            "operator_sets_ml.h",
            "operator_sets_preview.h",
            "operator_sets_training.h",
        ):
            content = (defs / name).read_text(encoding="utf-8")
            self.assertIn(
                "ONNX_LIGHT_NAMESPACE",
                content,
                msg=f"{name} does not reference ONNX_LIGHT_NAMESPACE",
            )

    def test_schema_h_exports_register_opset_schema(self):
        """Verifies that schema.h exports RegisterOpSetSchema and GetOpSchema."""
        schema_h = (
            Path(__file__).resolve().parents[2] / "onnx_light" / "onnx" / "defs" / "schema.h"
        )
        content = schema_h.read_text(encoding="utf-8")
        self.assertIn("RegisterOpSetSchema", content)
        self.assertIn("GetOpSchema", content)
        self.assertIn("ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME", content)
        self.assertIn("ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
