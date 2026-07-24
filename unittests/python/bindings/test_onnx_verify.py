# Tests for the schema-free structural validation helpers
# (see onnx_light/onnx_proto/verify.py and onnx_verify.h).
import unittest

from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx_lib import (
    AttributeProto,
    FunctionProto,
    GraphProto,
    ModelProto,
    NodeProto,
    TensorProto,
)
from onnx_light.onnx_proto import verify


def _make_valid_model() -> ModelProto:
    """Builds a minimal, structurally valid one-node identity model."""
    model = ModelProto()
    model.add_opset("", 18)
    graph = model.add_graph()
    graph.name = "g"
    graph.add_input("x", TensorProto.FLOAT, [])
    graph.add_output("y", TensorProto.FLOAT, [])
    graph.add_node("Identity", ["x"], ["y"])
    return model


class TestVerify(ExtTestCase):
    def test_verify_model_valid(self):
        """Checks that a well-formed model passes VerifyModel."""
        verify.verify_model(_make_valid_model())

    def test_verify_model_missing_graph(self):
        """Checks that a model without a graph is rejected."""
        with self.assertRaises(ValueError):
            verify.verify_model(ModelProto())

    def test_verify_model_missing_opset(self):
        """Checks that a model without any opset import is rejected."""
        model = _make_valid_model()
        del model.opset_import[:]
        with self.assertRaises(ValueError):
            verify.verify_model(model)

    def test_verify_graph_not_topologically_sorted(self):
        """Checks that a node consuming an undefined name is rejected."""
        graph = GraphProto()
        graph.name = "g"
        graph.add_input("x", TensorProto.FLOAT, [])
        graph.add_node("Identity", ["not_defined"], ["y"])
        with self.assertRaises(ValueError):
            verify.verify_graph(graph)

    def test_verify_graph_unproduced_output(self):
        """Checks that a declared output which is never produced is rejected."""
        graph = GraphProto()
        graph.name = "g"
        graph.add_input("x", TensorProto.FLOAT, [])
        graph.add_output("never_produced", TensorProto.FLOAT, [])
        with self.assertRaises(ValueError):
            verify.verify_graph(graph)

    def test_verify_graph_subgraph_closure_over_outer_scope(self):
        """Checks that a control-flow body may reference an outer-scope name."""
        body = GraphProto()
        body.name = "body"
        body.add_node("Identity", ["outer_x"], ["y"])
        body.add_output("y", None, None)
        verify.verify_graph(body, is_main_graph=False, outer_scope={"outer_x"})
        with self.assertRaises(ValueError):
            verify.verify_graph(body, is_main_graph=False)

    def test_verify_node_empty_op_type(self):
        """Checks that a node without op_type is rejected."""
        node = NodeProto()
        node.input.append("x")
        node.output.append("y")
        with self.assertRaises(ValueError):
            verify.verify_node(node)

    def test_verify_node_no_input_no_output(self):
        """Checks that a node with no input and no output is rejected."""
        node = NodeProto()
        node.op_type = "Identity"
        with self.assertRaises(ValueError):
            verify.verify_node(node)

    def test_verify_tensor_valid(self):
        """Checks that a well-formed tensor passes VerifyTensor."""
        tensor = TensorProto()
        tensor.name = "w"
        tensor.data_type = TensorProto.FLOAT
        tensor.dims.append(2)
        tensor.raw_data = b"\x00" * 8
        verify.verify_tensor(tensor)

    def test_verify_tensor_undefined_data_type(self):
        """Checks that a tensor without a data_type is rejected."""
        tensor = TensorProto()
        tensor.name = "w"
        with self.assertRaises(ValueError):
            verify.verify_tensor(tensor)

    def test_verify_attribute_ref_attr_name_outside_function(self):
        """Checks that ref_attr_name is only allowed inside a function body."""
        attr = AttributeProto()
        attr.name = "alpha"
        attr.ref_attr_name = "alpha"
        with self.assertRaises(ValueError):
            verify.verify_attribute(attr, in_function_body=False)
        verify.verify_attribute(attr, in_function_body=True)

    def test_verify_function_unproduced_output(self):
        """Checks that a function output which is never produced is rejected."""
        function = FunctionProto()
        function.name = "f"
        function.input.append("a")
        function.output.append("never_produced")
        with self.assertRaises(ValueError):
            verify.verify_function(function)


if __name__ == "__main__":
    unittest.main()
