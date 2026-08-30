import unittest

from onnx_light.ext_test_case import ExtTestCase
from pathlib import Path


class TestOnnxOptimDependency(ExtTestCase):
    """Verifies that lib_onnx_shape depends on lib_onnx_core, not lib_onnx_op."""

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
            r"target_link_libraries\(lib_onnx_shape\s+PUBLIC\s+lib_onnx_core\)",
            "lib_onnx_shape must link against lib_onnx_core",
        )

    def test_cmake_optim_does_not_link_onnx_op(self):
        cmake = self.root / "CMakeLists.txt"
        content = cmake.read_text(encoding="utf-8")
        self.assertNotRegex(
            content,
            r"target_link_libraries\(lib_onnx_shape\s+PUBLIC\s+lib_onnx_op\)",
            "lib_onnx_shape must not directly link against lib_onnx_op",
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
        header = self.root / "onnx_light" / "onnx_core" / "light_op_schema" / "light_op_schema.h"
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

    def test_shapes_engine_lives_in_onnx_core(self):
        shapes_dir = self.root / "onnx_light" / "onnx_core" / "shapes"
        for name in (
            "shapes_context.h",
            "shape_check.h",
            "shape_broadcast.h",
            "shape_inference.h",
            "dispatch_table.h",
            "dispatch_table.cc",
        ):
            self.assertTrue(
                (shapes_dir / name).exists(),
                f"onnx_core/shapes/{name} must exist: the generic shape-inference engine "
                "(ShapesContext, the dispatch table, ...) lives in onnx_core, while the "
                "per-operator ComputeShape* functions stay in onnx_shapes",
            )
        for name in (
            "shapes_context.h",
            "shape_check.h",
            "shape_broadcast.h",
            "shape_inference.h",
        ):
            self.assertFalse(
                (
                    self.root / "onnx_light" / "onnx_extensions" / "onnx_shapes" / "shapes" / name
                ).exists(),
                f"onnx_extensions/shapes/shapes/{name} must not exist; consumers must "
                f"include onnx_core/shapes/{name} directly",
            )

    def test_onnx_core_dispatch_table_does_not_include_onnx_shapes(self):
        for name in (
            "dispatch_table.h",
            "dispatch_table.cc",
            "shape_inference.h",
            "shape_inference.cc",
        ):
            header = self.root / "onnx_light" / "onnx_core" / "shapes" / name
            content = header.read_text(encoding="utf-8")
            self.assertNotIn(
                "onnx_extensions/shapes/",
                content,
                f"onnx_core/shapes/{name} must not include any onnx_shapes header",
            )

    def test_onnx_core_dispatch_table_exposes_registration_function(self):
        header = self.root / "onnx_light" / "onnx_core" / "shapes" / "dispatch_table.h"
        content = header.read_text(encoding="utf-8")
        self.assertIn(
            "RegisterComputeShapeFn",
            content,
            "onnx_core/shapes/dispatch_table.h must declare RegisterComputeShapeFn so "
            "onnx_shapes can populate the dispatch table without onnx_core depending on it",
        )

    def test_onnx_shapes_registers_its_builtin_shape_functions(self):
        header = self.root / "onnx_light" / "onnx_extensions" / "shapes" / "dispatch_table.h"
        content = header.read_text(encoding="utf-8")
        self.assertIn(
            "RegisterShapeFunctions",
            content,
            "onnx_extensions/shapes/dispatch_table.h must declare RegisterShapeFunctions, "
            "which registers every built-in onnx_shapes shape function with "
            "core::shapes::RegisterComputeShapeFn",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
