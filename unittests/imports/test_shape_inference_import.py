import unittest
from pathlib import Path


class TestShapeInferenceImport(unittest.TestCase):
    def test_defs_shape_inference_files_imported(self):
        """Checks that defs shape_inference files are vendored."""
        defs = Path(__file__).resolve().parents[2] / "onnx_light" / "onnx" / "defs"

        expected = {"shape_inference.cc", "shape_inference.h"}
        present = {path.name for path in defs.glob("*") if path.is_file()}
        self.assertTrue(expected.issubset(present), msg=f"missing={sorted(expected - present)}")

    def test_defs_shape_inference_files_use_light_namespace(self):
        """Checks that defs shape_inference files use ONNX_LIGHT_NAMESPACE."""
        defs = Path(__file__).resolve().parents[2] / "onnx_light" / "onnx" / "defs"

        for name in ("shape_inference.cc", "shape_inference.h"):
            content = (defs / name).read_text(encoding="utf-8")
            self.assertIn("ONNX_LIGHT_NAMESPACE", content)

    def test_shape_inference_dir_files_imported(self):
        """Checks that shape_inference directory files are vendored."""
        root = Path(__file__).resolve().parents[2]
        si = root / "onnx_light" / "onnx" / "shape_inference"

        expected = {"attribute_binder.h", "implementation.cc", "implementation.h"}
        present = {path.name for path in si.glob("*") if path.is_file()}
        self.assertTrue(expected.issubset(present), msg=f"missing={sorted(expected - present)}")

    def test_shape_inference_dir_files_use_light_namespace(self):
        """Checks that shape_inference directory files use ONNX_LIGHT_NAMESPACE."""
        root = Path(__file__).resolve().parents[2]
        si = root / "onnx_light" / "onnx" / "shape_inference"

        for name in ("attribute_binder.h", "implementation.cc", "implementation.h"):
            content = (si / name).read_text(encoding="utf-8")
            self.assertIn("ONNX_LIGHT_NAMESPACE", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
