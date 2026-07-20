import unittest
from pathlib import Path


class TestOnnxOptimDependency(unittest.TestCase):
    """Verifies that lib_onnx_optim depends on lib_onnx_core, not lib_onnx_op."""

    def setUp(self):
        self.root = Path(__file__).resolve().parents[2]

    def test_sym_tensor_includes_onnx_core_not_onnx_op(self):
        header = self.root / "onnx_light" / "onnx_core" / "symbolic" / "sym_tensor.h"
        content = header.read_text(encoding="utf-8")
        self.assertIn(
            "onnx_proto/type_helper.h",
            content,
            "sym_tensor.h must include onnx_proto/type_helper.h",
        )
        self.assertNotIn(
            "onnx_op/light_op_schema.h",
            content,
            "sym_tensor.h must not include onnx_op/light_op_schema.h",
        )

    def test_sym_tensor_type_alias_uses_onnx_core(self):
        header = self.root / "onnx_light" / "onnx_core" / "symbolic" / "sym_tensor.h"
        content = header.read_text(encoding="utf-8")
        self.assertIn(
            "onnx_proto::TensorType",
            content,
            "TensorType alias in sym_tensor.h must reference onnx_proto::TensorType",
        )
        self.assertNotIn(
            "onnx_op::TensorType",
            content,
            "TensorType alias in sym_tensor.h must not reference onnx_op::TensorType",
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

    def test_tensor_type_header_in_onnx_proto(self):
        header = self.root / "onnx_light" / "onnx_proto" / "type_helper.h"
        self.assertTrue(header.exists(), "onnx_proto/type_helper.h must exist")
        content = header.read_text(encoding="utf-8")
        self.assertIn(
            "enum class TensorType", content, "type_helper.h must define the TensorType enum"
        )
        self.assertIn(
            "const char *ToTypeString", content, "type_helper.h must declare ToTypeString"
        )

    def test_onnx_op_light_op_schema_wrapper_removed(self):
        header = self.root / "onnx_light" / "onnx_op" / "light_op_schema.h"
        self.assertFalse(
            header.exists(),
            "onnx_op/light_op_schema.h must not exist; consumers include "
            "onnx_core/light_op_schema/light_op_schema.h directly",
        )

    def test_operator_sets_includes_onnx_core_light_op_schema(self):
        header = self.root / "onnx_light" / "onnx_op" / "operator_sets.h"
        content = header.read_text(encoding="utf-8")
        self.assertIn(
            "onnx_core/light_op_schema/light_op_schema.h",
            content,
            "operator_sets.h must include onnx_core/light_op_schema/light_op_schema.h",
        )
        self.assertNotIn(
            '"onnx_op/light_op_schema.h"',
            content,
            "operator_sets.h must not include the removed onnx_op/light_op_schema.h wrapper",
        )

    def test_onnx_core_light_op_schema_re_exports_tensor_type(self):
        header = (
            self.root / "onnx_light" / "onnx_core" / "light_op_schema" / "light_op_schema.h"
        )
        self.assertTrue(header.exists(), "onnx_core/light_op_schema/light_op_schema.h must exist")
        content = header.read_text(encoding="utf-8")
        self.assertIn(
            "onnx_proto/type_helper.h",
            content,
            "light_op_schema.h must include onnx_proto/type_helper.h",
        )
        self.assertIn(
            "using TensorType = onnx_proto::TensorType",
            content,
            "light_op_schema.h must re-export TensorType from onnx_proto",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
