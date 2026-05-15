# source: https://github.com/onnx/onnx/blob/main/onnx/test/function_inference_test.py
#
# This test adapts the ONNX function-inference test to onnx_light.
# onnx_light parses functions via parser.parse_function and uses the native
# C++ shape-inference binding to infer output types over the function body.
from __future__ import annotations

import unittest
from typing import TYPE_CHECKING

from onnx_light.ext_test_case import ExtTestCase
import onnx_light.onnx as onnxl
import onnx_light.onnx.defs as defs
import onnx_light.onnx.helper as oh
import onnx_light.onnx.parser as parser
import onnx_light.onnx.shape_inference as shape_inference

if TYPE_CHECKING:
    from collections.abc import Sequence

float_type_ = oh.make_tensor_type_proto(1, None)
uint8_type_ = oh.make_tensor_type_proto(2, None)
int8_type_ = oh.make_tensor_type_proto(3, None)
int32_type_ = oh.make_tensor_type_proto(6, None)
float16_type_ = oh.make_tensor_type_proto(10, None)
no_type_ = onnxl.TypeProto()


def _infer_function_output_types(
    function: onnxl.FunctionProto,
    input_types: Sequence[onnxl.TypeProto],
    attributes: Sequence[onnxl.AttributeProto],
) -> list[onnxl.TypeProto]:
    """Infers output types via the native onnx_light C++ shape-inference binding.

    Returns:
        A list of :class:`TypeProto` objects, one per function output.

    Raises:
        shape_inference.InferenceError: If type or shape inference fails.
    """
    return shape_inference.infer_function_output_types(function, input_types, attributes)


class TestFunctionInference(ExtTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Registers test op schemas needed for function-level shape inference."""
        defs.register_onnx_operator_set_schema()

    def _compare_value_infos(
        self, vi_type: onnxl.TypeProto, inferred_vi_type: onnxl.TypeProto
    ) -> None:
        """Compares two TypeProto objects for compatible type and shape."""
        if vi_type.has_tensor_type():
            assert inferred_vi_type.has_tensor_type(), f"No type for {inferred_vi_type!r}"
            self.assertEqual(
                vi_type.tensor_type.elem_type, inferred_vi_type.tensor_type.elem_type
            )
            has_shape = vi_type.tensor_type.has_shape()
            self.assertEqual(has_shape, inferred_vi_type.tensor_type.has_shape())
        elif vi_type == onnxl.TypeProto():
            # Empty TypeProto (no type set) – treated as a match with anything.
            pass

    def _check(
        self,
        function_text: str,
        input_types: Sequence[onnxl.TypeProto],
        attributes: Sequence[onnxl.AttributeProto],
        expected_output_types: Sequence[onnxl.TypeProto],
    ) -> None:
        """Parses a function, runs inference, and checks the output types."""
        function = parser.parse_function(function_text)
        print(function)
        result = _infer_function_output_types(function, input_types, attributes)
        self.assertEqual(len(expected_output_types), len(result))
        for expected, actual in zip(expected_output_types, result, strict=True):
            self._compare_value_infos(expected, actual)

    def _check_fails(
        self,
        function_text: str,
        input_types: Sequence[onnxl.TypeProto],
        attributes: Sequence[onnxl.AttributeProto],
    ) -> None:
        """Asserts that inference raises InferenceError."""
        function = parser.parse_function(function_text)
        with self.assertRaises(shape_inference.InferenceError):
            _infer_function_output_types(function, input_types, attributes)

    def test_fi_basic(self) -> None:
        code = """
            <opset_import: [ "" : 18 ], domain: "local">
            f (y, z) => (w) {
                x = Add(y, z)
                w = Mul(x, y)
            }
        """
        self._check(code, [float_type_, float_type_], [], [float_type_])
        self._check(code, [int32_type_, int32_type_], [], [int32_type_])
        self._check_fails(code, [float_type_, int32_type_], [])

    def test_fi_attribute(self) -> None:
        code = """
            <opset_import: [ "" : 18 ], domain: "local">
            CastTo <dtype> (x) => (y) {
                y = Cast <to : int = @dtype> (x)
            }
        """
        dtype_6 = oh.make_attribute("dtype", 6)
        self._check(code, [float_type_], [dtype_6], [int32_type_])

        dtype_10 = oh.make_attribute("dtype", 10)
        self._check(code, [float_type_], [dtype_10], [float16_type_])

    def test_fi_optional_input(self) -> None:
        code = """
            <opset_import: [ "" : 18 ], domain: "local">
            DoReduce (x, axes) => (y) {
                y = ReduceMax (x, axes)
            }
        """
        # We can omit the type for a missing trailing optional parameter.
        self._check(code, [float_type_], [], [float_type_])
        # Or, we can pass in a default-value of TypeProto() for a missing optional parameter.
        self._check(code, [float_type_, no_type_], [], [float_type_])

        code = """
            <opset_import: [ "" : 18 ], domain: "local">
            Quantize (x, scale, zero_point) => (y) {
                y = QuantizeLinear (x, scale, zero_point)
            }
        """
        # If the optional third parameter is specified, it determines the output type.
        self._check(code, [float_type_, float_type_, int8_type_], [], [int8_type_])
        self._check(code, [float_type_, float_type_, uint8_type_], [], [uint8_type_])
        # If the optional third parameter is omitted, the output type is uint8 (default).
        self._check(code, [float_type_, float_type_, no_type_], [], [uint8_type_])

        code = """
            <opset_import: [ "" : 18 ], domain: "local">
            DoClip (x, min, max) => (y) {
                y = Clip (x, min, max)
            }
        """
        # A test-case with a non-trailing missing optional parameter.
        self._check(code, [float_type_, no_type_, float_type_], [], [float_type_])

        # A failing test-case with a non-trailing missing optional parameter.
        self._check_fails(code, [float_type_, no_type_, int8_type_], [])


if __name__ == "__main__":
    unittest.main()
