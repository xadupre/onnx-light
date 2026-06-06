import unittest

from onnx_light.ext_test_case import ExtTestCase
from pathlib import Path


class TestVersionConverterImport(ExtTestCase):
    def test_version_converter_files_imported(self):
        """Verifies that version_converter files are vendored."""
        root = Path(__file__).resolve().parents[2]
        vc = root / "onnx_light" / "onnx_lib" / "version_converter"

        expected = {"BaseConverter.h", "convert.cc", "convert.h", "helper.cc", "helper.h"}
        present = {path.name for path in vc.glob("*") if path.is_file()}
        self.assertTrue(expected.issubset(present), msg=f"missing={sorted(expected - present)}")

    def test_version_converter_files_use_light_namespace(self):
        """Verifies that version_converter files use ONNX_LIGHT_NAMESPACE."""
        root = Path(__file__).resolve().parents[2]
        vc = root / "onnx_light" / "onnx_lib" / "version_converter"

        for name in ("BaseConverter.h", "convert.cc", "convert.h", "helper.cc", "helper.h"):
            content = (vc / name).read_text(encoding="utf-8")
            self.assertIn("ONNX_LIGHT_NAMESPACE", content)

    def test_version_converter_adapters_imported(self):
        """Verifies that version_converter adapter files are vendored."""
        root = Path(__file__).resolve().parents[2]
        adapters = root / "onnx_light" / "onnx_lib" / "version_converter" / "adapters"

        expected = {
            "adapter.h",
            "compatible.h",
            "transformers.h",
            "no_previous_version.h",
            "broadcast_backward_compatibility.h",
            "broadcast_forward_compatibility.h",
        }
        present = {path.name for path in adapters.glob("*.h") if path.is_file()}
        self.assertTrue(expected.issubset(present), msg=f"missing={sorted(expected - present)}")

    def test_version_converter_adapters_use_light_namespace(self):
        """Verifies that version_converter adapter files use ONNX_LIGHT_NAMESPACE."""
        root = Path(__file__).resolve().parents[2]
        adapters = root / "onnx_light" / "onnx_lib" / "version_converter" / "adapters"

        for path in sorted(adapters.glob("*.h")):
            content = path.read_text(encoding="utf-8")
            self.assertIn(
                "ONNX_LIGHT_NAMESPACE",
                content,
                msg=f"{path.name} does not use ONNX_LIGHT_NAMESPACE",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
