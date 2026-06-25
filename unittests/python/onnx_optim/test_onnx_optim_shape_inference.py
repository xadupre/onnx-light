import unittest

import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx import defs
from onnx_light.onnx_optim.shape_inference import infer_shapes_model


class TestOnnxOptimShapeInferenceModel(ExtTestCase):
    """End-to-end Python tests for ``onnx_optim`` shape inference at the
    ``ModelProto`` level. They mirror the C++ ``OnnxOptimShapeInference``
    tests for ``Reshape(X, Constant([-1, 2]))``.
    """

    @classmethod
    def setUpClass(cls):
        defs.register_onnx_operator_set_schema()

    @staticmethod
    def _make_reshape_with_constant_model(input_shape, target):
        """Builds ``Y = Reshape(X, Constant(value_ints=target))`` where
        each entry of ``input_shape`` is either an ``int`` (concrete
        dim_value) or a ``str`` (symbolic dim_param)."""
        constant = oh.make_node("Constant", inputs=[], outputs=["S"], value_ints=list(target))
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

    def test_prefill_prefers_output_anchor(self):
        model = self._make_reshape_with_constant_model(["N", 4], [-1, 2])
        model.graph.output[0].CopyFrom(
            oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, ["ANCHOR", 2])
        )
        infer_shapes_model(model, prefill_with_value_info_output=True)

        y = model.graph.output[0]
        dims = list(y.type.tensor_type.shape.dim)
        self.assertEqual(len(dims), 2)
        self.assertTrue(dims[0].has_dim_param())
        self.assertEqual(dims[0].dim_param, "ANCHOR")
        self.assertEqual(dims[1].dim_value, 2)

    def test_rejects_model_without_graph(self):
        model = onnxl.ModelProto()
        with self.assertRaises(ValueError):
            infer_shapes_model(model)

    def test_reshape_from_gather_shape_unsqueeze_concat(self):
        """Tests that Reshape(x, Concat(Unsqueeze(Gather(Shape(y), 1), 0), ...))
        infers the output shape from symbolic dims without adding undefined names."""
        # Graph:
        #   x: float[N, D1, D2]
        #   y: float[M, D1]   <- dim 1 is "D1"
        #   z: float[K, D2]   <- dim 1 is "D2"
        #   idx1 = Constant(1)
        #   idx2 = Constant(1)
        #   axes0 = Constant([0])
        #   shape_y = Shape(y)                    # int64[2] values=[M, D1]
        #   shape_z = Shape(z)                    # int64[2] values=[K, D2]
        #   d1 = Gather(shape_y, idx1, axis=0)   # scalar with value D1
        #   d2 = Gather(shape_z, idx2, axis=0)   # scalar with value D2
        #   u1 = Unsqueeze(d1, axes0)             # int64[1] values=[D1]
        #   u2 = Unsqueeze(d2, axes0)             # int64[1] values=[D2]
        #   new_shape = Concat(u1, u2, axis=0)   # int64[2] values=[D1, D2]
        #   out = Reshape(x[*, D1, D2], new_shape) # float[*, D1, D2]
        nodes = [
            oh.make_node(
                "Constant",
                [],
                ["idx"],
                value=oh.make_tensor("", onnxl.TensorProto.INT64, [], [1]),
            ),
            oh.make_node(
                "Constant",
                [],
                ["axes0"],
                value=oh.make_tensor("", onnxl.TensorProto.INT64, [1], [0]),
            ),
            oh.make_node("Shape", ["y"], ["shape_y"]),
            oh.make_node("Shape", ["z"], ["shape_z"]),
            oh.make_node("Gather", ["shape_y", "idx"], ["d1"]),
            oh.make_node("Gather", ["shape_z", "idx"], ["d2"]),
            oh.make_node("Unsqueeze", ["d1", "axes0"], ["u1"]),
            oh.make_node("Unsqueeze", ["d2", "axes0"], ["u2"]),
            oh.make_node("Concat", ["u1", "u2"], ["new_shape"], axis=0),
            oh.make_node("Reshape", ["x", "new_shape"], ["out"]),
        ]
        x = oh.make_tensor_value_info("x", onnxl.TensorProto.FLOAT, ["N", "D1", "D2"])
        y = oh.make_tensor_value_info("y", onnxl.TensorProto.FLOAT, ["M", "D1"])
        z = oh.make_tensor_value_info("z", onnxl.TensorProto.FLOAT, ["K", "D2"])
        out = oh.make_tensor_value_info("out", onnxl.TensorProto.FLOAT, None)
        graph = oh.make_graph(nodes, "g", [x, y, z], [out])
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
        model.ir_version = 8

        infer_shapes_model(model)

        dims = list(model.graph.output[0].type.tensor_type.shape.dim)
        self.assertEqual(len(dims), 2)
        # Both output dims must be symbolic (derived from D1 and D2), not
        # undefined placeholder names like "Reshape_dim0".
        for dim in dims:
            self.assertFalse(
                dim.has_dim_value(), "Expected symbolic output dim, got concrete value"
            )
            self.assertTrue(dim.has_dim_param(), "Expected symbolic dim_param, got no name")
        # The dim_param values must match the original symbolic names from y and z.
        self.assertEqual(dims[0].dim_param, "D1")
        self.assertEqual(dims[1].dim_param, "D2")


if __name__ == "__main__":
    unittest.main()
