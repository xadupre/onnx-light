import unittest

import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx_optim.shape_inference import infer_shapes_model


class TestOnnxOptimShapeInferenceModel(ExtTestCase):
    """End-to-end Python tests for ``onnx_optim`` shape inference at the
    ``ModelProto`` level. They mirror the C++ ``OnnxOptimShapeInference``
    tests for ``Reshape(X, Constant([-1, 2]))``.
    """

    @classmethod
    def setUpClass(cls):
        onnxl.defs.register_onnx_operator_set_schema()

    @staticmethod
    def _make_reshape_with_constant_model(input_shape, target):
        """Builds ``Y = Reshape(X, Constant(value_ints=target))`` where
        each entry of ``input_shape`` is either an ``int`` (concrete
        dim_value) or a ``str`` (symbolic dim_param)."""
        constant = oh.make_node(
            "Constant",
            inputs=[],
            outputs=["S"],
            value_ints=list(target),
        )
        reshape = oh.make_node("Reshape", inputs=["X", "S"], outputs=["Y"])
        x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, list(input_shape))
        y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, None)
        graph = oh.make_graph([constant, reshape], "g", [x], [y])
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
        model.ir_version = 8
        return model

    def test_reshape_static_shape(self):
        # X = {3, 4}, target = [-1, 2] ⇒ Y = {6, 2}.
        model = self._make_reshape_with_constant_model([3, 4], [-1, 2])
        infer_shapes_model(model)

        # Y output type is float with shape {6, 2}.
        self.assertEqual(len(model.graph.output), 1)
        y = model.graph.output[0]
        self.assertEqual(y.name, "Y")
        self.assertEqual(y.type.tensor_type.elem_type, onnxl.TensorProto.FLOAT)
        dims = list(y.type.tensor_type.shape.dim)
        self.assertEqual(len(dims), 2)
        self.assertEqual(dims[0].dim_value, 6)
        self.assertEqual(dims[1].dim_value, 2)

        # The intermediate S tensor ends up in graph.value_info.
        names = [vi.name for vi in model.graph.value_info]
        self.assertIn("S", names)
        # The graph input X is not duplicated in value_info.
        self.assertNotIn("X", names)

    def test_reshape_dynamic_shape(self):
        # X = {N, 4}, target = [-1, 2] ⇒ Y[0] symbolic, Y[1] = 2.
        model = self._make_reshape_with_constant_model(["N", 4], [-1, 2])
        infer_shapes_model(model)

        y = model.graph.output[0]
        self.assertEqual(y.type.tensor_type.elem_type, onnxl.TensorProto.FLOAT)
        dims = list(y.type.tensor_type.shape.dim)
        self.assertEqual(len(dims), 2)
        # First dim is symbolic (dim_param), not a concrete dim_value.
        self.assertFalse(dims[0].has_dim_value())
        self.assertTrue(dims[0].has_dim_param())
        self.assertEqual(dims[1].dim_value, 2)

    def test_rejects_model_without_graph(self):
        model = onnxl.ModelProto()
        with self.assertRaises(ValueError):
            infer_shapes_model(model)


if __name__ == "__main__":
    unittest.main()
