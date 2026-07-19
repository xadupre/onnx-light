import re
import unittest
from pathlib import Path


class TestOnnxOptimDependency(unittest.TestCase):
    """Verifies that lib_onnx_optim depends on lib_onnx_core, not lib_onnx_op."""

    def setUp(self):
        self.root = Path(__file__).resolve().parents[2]

    def test_optim_tensor_includes_onnx_core_not_onnx_op(self):
        header = self.root / "onnx_light" / "onnx_optim" / "optim_tensor.h"
        content = header.read_text(encoding="utf-8")
        self.assertIn(
            "onnx_core/tensor_type.h",
            content,
            "optim_tensor.h must include onnx_core/tensor_type.h",
        )
        self.assertNotIn(
            "onnx_op/light_op_schema.h",
            content,
            "optim_tensor.h must not include onnx_op/light_op_schema.h",
        )

    def test_optim_tensor_type_alias_uses_onnx_core(self):
        header = self.root / "onnx_light" / "onnx_optim" / "optim_tensor.h"
        content = header.read_text(encoding="utf-8")
        self.assertIn(
            "onnx_core::TensorType",
            content,
            "TensorType alias in optim_tensor.h must reference onnx_core::TensorType",
        )
        self.assertNotIn(
            "onnx_op::TensorType",
            content,
            "TensorType alias in optim_tensor.h must not reference onnx_op::TensorType",
        )

    def test_cmake_optim_links_onnx_core(self):
        cmake = self.root / "CMakeLists.txt"
        content = cmake.read_text(encoding="utf-8")
        self.assertRegex(
            content,
            r"target_link_libraries\(lib_onnx_optim\s+PUBLIC\s+lib_onnx_core\)",
            "lib_onnx_optim must link against lib_onnx_core",
        )

    def test_cmake_optim_does_not_link_onnx_op(self):
        cmake = self.root / "CMakeLists.txt"
        content = cmake.read_text(encoding="utf-8")
        self.assertNotRegex(
            content,
            r"target_link_libraries\(lib_onnx_optim\s+PUBLIC\s+lib_onnx_op\)",
            "lib_onnx_optim must not directly link against lib_onnx_op",
        )

    def test_tensor_type_header_in_onnx_core(self):
        header = self.root / "onnx_light" / "onnx_core" / "tensor_type.h"
        self.assertTrue(
            header.exists(), "onnx_core/tensor_type.h must exist"
        )
        content = header.read_text(encoding="utf-8")
        self.assertIn(
            "enum class TensorType",
            content,
            "tensor_type.h must define the TensorType enum",
        )
        self.assertIn(
            "const char *ToTypeString",
            content,
            "tensor_type.h must declare ToTypeString",
        )

    def test_light_op_schema_re_exports_tensor_type(self):
        header = self.root / "onnx_light" / "onnx_op" / "light_op_schema.h"
        content = header.read_text(encoding="utf-8")
        self.assertIn(
            "onnx_core/tensor_type.h",
            content,
            "light_op_schema.h must include onnx_core/tensor_type.h",
        )
        self.assertIn(
            "using TensorType = onnx_core::TensorType",
            content,
            "light_op_schema.h must re-export TensorType from onnx_core",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
