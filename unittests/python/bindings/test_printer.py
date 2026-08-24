import unittest

import numpy as np

from onnx_light.ext_test_case import ExtTestCase
import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
from onnx_light.onnx import numpy_helper, parser
from onnx_light.onnx.printer import to_text
from onnx_light.onnx_lib.printer import to_text as lib_to_text


class TestPrinter(ExtTestCase):
    @staticmethod
    def _print_initializer(initializer):
        graph = oh.make_graph([], "graph", [], [], [initializer])
        return to_text(graph)

    def test_to_text_model_proto(self):
        """Tests that to_text converts a ModelProto to text."""
        inp = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [2, 3])
        out = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [2, 3])
        node = oh.make_node("Relu", ["X"], ["Y"])
        graph = oh.make_graph([node], "test_graph", [inp], [out])
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])

        text = to_text(model)
        self.assertIsInstance(text, str)
        self.assertIn("test_graph", text)
        self.assertIn("Relu", text)

    def test_to_text_graph_proto(self):
        """Tests that to_text converts a GraphProto to text."""
        inp = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [2, 3])
        out = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [2, 3])
        node = oh.make_node("Sigmoid", ["X"], ["Y"])
        graph = oh.make_graph([node], "test_graph", [inp], [out])

        text = to_text(graph)
        self.assertIsInstance(text, str)
        self.assertIn("test_graph", text)
        self.assertIn("Sigmoid", text)

    def test_to_text_function_proto(self):
        """Tests that to_text converts a FunctionProto to text."""
        func = onnxl.FunctionProto()
        func.name = "TestFunc"
        func.domain = ""
        func.input.extend(["X"])
        func.output.extend(["Y"])
        node = onnxl.NodeProto()
        node.op_type = "Abs"
        node.input.extend(["X"])
        node.output.extend(["Y"])
        func.node.extend([node])

        text = to_text(func)
        self.assertIsInstance(text, str)
        self.assertIn("TestFunc", text)
        self.assertIn("Abs", text)

    def test_float16_initializer_roundtrip(self):
        values = np.array([1.0, -2.0, 0.5], dtype=np.float16)
        text = self._print_initializer(numpy_helper.from_array(values, name="weights"))

        self.assertIn("{15360,49152,14336}", text)
        parsed = parser.parse_graph(text)
        np.testing.assert_array_equal(numpy_helper.to_array(parsed.initializer[0]), values)

    def test_raw_initializers_roundtrip(self):
        cases = [
            (np.int8, [1, -2, 127, -128], "{1,-2,127,-128}"),
            (np.uint8, [0, 1, 255], "{0,1,255}"),
            (np.bool_, [True, False, True], "{1,0,1}"),
            (np.int16, [1, -2, 32767, -32768], "{1,-2,32767,-32768}"),
            (np.uint16, [1, 2, 65535], "{1,2,65535}"),
            (np.int32, [1, -2, 2147483647], "{1,-2,2147483647}"),
            (np.uint32, [1, 2, 4294967295], "{1,2,4294967295}"),
            (np.int64, [1, -2, 9223372036854775807], "{1,-2,9223372036854775807}"),
            (np.uint64, [1, 2, 18446744073709551615], "{1,2,18446744073709551615}"),
            (np.float32, [1.5, -2.5, 0.0], "{1.5,-2.5,0}"),
            (np.float64, [1.5, -2.5, 0.0], "{1.5,-2.5,0}"),
        ]
        for dtype, values, expected in cases:
            with self.subTest(dtype=dtype):
                array = np.array(values, dtype=dtype)
                text = self._print_initializer(numpy_helper.from_array(array, name="weights"))
                self.assertIn(expected, text)
                parsed = parser.parse_graph(text)
                np.testing.assert_array_equal(numpy_helper.to_array(parsed.initializer[0]), array)

    def test_raw_float_bit_patterns(self):
        cases = [
            (onnxl.TensorProto.BFLOAT16, b"\x80\x3f\x00\xc0\x00\x3f", [16256, 49152, 16128]),
            *[
                (data_type, b"\x38\xc0\x30", [56, 192, 48])
                for data_type in (
                    onnxl.TensorProto.FLOAT8E4M3FN,
                    onnxl.TensorProto.FLOAT8E4M3FNUZ,
                    onnxl.TensorProto.FLOAT8E5M2,
                    onnxl.TensorProto.FLOAT8E5M2FNUZ,
                    onnxl.TensorProto.FLOAT8E8M0,
                )
            ],
        ]
        for data_type, raw_data, expected in cases:
            with self.subTest(data_type=data_type):
                initializer = oh.make_tensor("weights", data_type, [3], raw_data, raw=True)
                text = self._print_initializer(initializer)
                self.assertIn("{" + ",".join(map(str, expected)) + "}", text)
                self.assertEqual(
                    list(parser.parse_graph(text).initializer[0].int32_data), expected
                )

    def test_raw_data_size_mismatch_raises(self):
        initializer = onnxl.TensorProto()
        initializer.name = "weights"
        initializer.data_type = onnxl.TensorProto.FLOAT16
        initializer.dims.extend([2])
        initializer.raw_data = b"\x00\x3c"

        with self.assertRaisesRegex(RuntimeError, "Data size mismatch"):
            self._print_initializer(initializer)

    def test_subbyte_initializer_prints_placeholder(self):
        initializer = oh.make_tensor(
            "weights", onnxl.TensorProto.INT4, [4], b"\x21\x43", raw=True
        )
        self.assertIn("...", self._print_initializer(initializer))

    def test_to_text_unsupported_type_raises(self):
        """Tests that to_text raises TypeError for unsupported types."""
        tensor = onnxl.TensorProto()
        tensor.name = "test"

        with self.assertRaises(TypeError) as ctx:
            to_text(tensor)
        self.assertIn("TensorProto", str(ctx.exception))
        self.assertIn("ModelProto, FunctionProto, or GraphProto", str(ctx.exception))

    def test_lib_to_text_model_proto(self):
        """Tests that onnx_lib.printer.to_text also works."""
        inp = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [2, 3])
        out = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [2, 3])
        node = oh.make_node("Add", ["X", "X"], ["Y"])
        graph = oh.make_graph([node], "lib_test", [inp], [out])
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])

        text = lib_to_text(model)
        self.assertIsInstance(text, str)
        self.assertIn("lib_test", text)
        self.assertIn("Add", text)

    def test_onnx_printer_import_alias(self):
        """Tests that the onnx.printer module is importable."""
        from onnx_light.onnx import printer

        inp = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [2, 3])
        out = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [2, 3])
        node = oh.make_node("Relu", ["X"], ["Y"])
        graph = oh.make_graph([node], "alias_test", [inp], [out])
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])

        text = printer.to_text(model)
        self.assertIsInstance(text, str)
        self.assertIn("alias_test", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
