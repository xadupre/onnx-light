import unittest

from onnx_light.ext_test_case import ExtTestCase
import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
from onnx_light.onnx.printer import to_text
from onnx_light.onnx_lib.printer import to_text as lib_to_text


class TestPrinter(ExtTestCase):
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
