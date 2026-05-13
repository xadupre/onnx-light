import unittest

from onnx_light.ext_test_case import ExtTestCase
from pathlib import Path


class TestTensorImport(ExtTestCase):
    def test_tensor_files_imported(self):
        """Verifies that tensor defs files are vendored."""
        root = Path(__file__).resolve().parents[2]
        tensor = root / "onnx_light" / "onnx" / "defs" / "tensor"

        expected = {"defs.cc", "old.cc", "utils.cc", "utils.h"}
        present = {path.name for path in tensor.glob("*") if path.is_file()}
        self.assertTrue(expected.issubset(present), msg=f"missing={sorted(expected - present)}")

    def test_tensor_uses_light_namespace(self):
        """Verifies that tensor files use ONNX_LIGHT_NAMESPACE."""
        root = Path(__file__).resolve().parents[2]
        tensor = root / "onnx_light" / "onnx" / "defs" / "tensor"

        for name in ("defs.cc", "old.cc", "utils.cc", "utils.h"):
            content = (tensor / name).read_text(encoding="utf-8")
            self.assertIn("ONNX_LIGHT_NAMESPACE", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
