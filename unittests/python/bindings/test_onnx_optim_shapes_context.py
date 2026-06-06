import unittest

import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx_py._onnxpy import shape_inference as si


class TestShapesContextBindings(ExtTestCase):
    """Python tests for the ``ShapesContext`` and ``ComputeShapeNode``
    bindings exposed by ``onnx_light.onnx_py._onnxpy.shape_inference``."""

    @classmethod
    def setUpClass(cls):
        onnxl.defs.register_onnx_operator_set_schema()

    # ------------------------------------------------------------------
    # OptimDim / OptimShape / OptimTensor value-type bindings.
    # ------------------------------------------------------------------
    def test_optim_dim_int(self):
        d = si.OptimDim(5)
        self.assertTrue(d.is_int())
        self.assertFalse(d.is_expr())
        self.assertEqual(d.as_int(), 5)
        self.assertEqual(d.value(), 5)
        self.assertEqual(str(d), "5")

    def test_optim_dim_expr(self):
        d = si.OptimDim("N")
        self.assertTrue(d.is_expr())
        self.assertFalse(d.is_int())
        self.assertEqual(d.as_expr(), "N")
        self.assertEqual(d.value(), "N")
        self.assertEqual(str(d), "N")

    def test_optim_dim_equality(self):
        self.assertEqual(si.OptimDim(3), si.OptimDim(3))
        self.assertNotEqual(si.OptimDim(3), si.OptimDim(4))
        self.assertNotEqual(si.OptimDim(3), si.OptimDim("3"))
        self.assertEqual(si.OptimDim("N"), si.OptimDim("N"))
        self.assertNotEqual(si.OptimDim("N"), si.OptimDim("M"))

    def test_optim_dim_repr(self):
        self.assertEqual(repr(si.OptimDim(5)), "OptimDim(5)")
        self.assertEqual(repr(si.OptimDim("N")), "OptimDim('N')")

    def test_optim_shape_construction_and_indexing(self):
        s = si.OptimShape([3, "N", 5])
        self.assertEqual(s.rank(), 3)
        self.assertEqual(len(s), 3)
        self.assertFalse(s.empty())
        self.assertFalse(s.is_fully_known())
        self.assertEqual(s[0], 3)
        self.assertEqual(s[1], "N")
        self.assertEqual(s[2], 5)
        self.assertEqual(list(s), [3, "N", 5])
        self.assertEqual(s.dims(), [3, "N", 5])

    def test_optim_shape_fully_known(self):
        s = si.OptimShape([2, 3, 4])
        self.assertTrue(s.is_fully_known())

    def test_optim_shape_equality(self):
        self.assertEqual(si.OptimShape([2, 3]), si.OptimShape([2, 3]))
        self.assertEqual(si.OptimShape([2, "N"]), si.OptimShape([2, "N"]))
        self.assertNotEqual(si.OptimShape([2, 3]), si.OptimShape([2, 4]))
        self.assertNotEqual(si.OptimShape([2, 3]), si.OptimShape([2, 3, 1]))
        self.assertNotEqual(si.OptimShape([2, "N"]), si.OptimShape([2, "M"]))

    def test_optim_shape_repr(self):
        self.assertEqual(repr(si.OptimShape([2, 3])), "OptimShape([2, 3])")
        self.assertEqual(repr(si.OptimShape([2, "N", 5])), "OptimShape([2, 'N', 5])")
        self.assertEqual(repr(si.OptimShape([])), "OptimShape([])")

    def test_optim_tensor_basic(self):
        t = si.OptimTensor(onnxl.TensorProto.FLOAT, [2, 3])
        self.assertEqual(t.dtype, onnxl.TensorProto.FLOAT)
        self.assertEqual(list(t.shape), [2, 3])
        self.assertTrue(t.is_null())
        self.assertFalse(t.has_value_as_shape())

    def test_optim_tensor_symbolic_shape(self):
        t = si.OptimTensor(onnxl.TensorProto.INT64, ["B", 4])
        self.assertEqual(t.dtype, onnxl.TensorProto.INT64)
        self.assertEqual(list(t.shape), ["B", 4])

    def test_optim_tensor_value_as_shape(self):
        t = si.OptimTensor(onnxl.TensorProto.INT64, [2])
        self.assertFalse(t.has_value_as_shape())
        t.set_value_as_shape([3, "K"])
        self.assertTrue(t.has_value_as_shape())
        self.assertEqual(list(t.value_as_shape()), [3, "K"])
        t.clear_value_as_shape()
        self.assertFalse(t.has_value_as_shape())

    def test_optim_tensor_equality(self):
        a = si.OptimTensor(onnxl.TensorProto.FLOAT, [2, 3])
        b = si.OptimTensor(onnxl.TensorProto.FLOAT, [2, 3])
        c = si.OptimTensor(onnxl.TensorProto.FLOAT, [2, 4])
        d = si.OptimTensor(onnxl.TensorProto.INT64, [2, 3])
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, d)

    def test_optim_tensor_repr(self):
        t = si.OptimTensor(onnxl.TensorProto.FLOAT, [2, 3])
        r = repr(t)
        self.assertIn("OptimTensor(", r)
        self.assertIn("Float", r)
        self.assertIn("[2,3]", r)

    # ------------------------------------------------------------------
    # ShapesContext bindings.
    # ------------------------------------------------------------------
    def test_shapes_context_basics(self):
        ctx = si.ShapesContext()
        self.assertTrue(ctx.empty())
        self.assertEqual(ctx.size(), 0)
        self.assertFalse(ctx.has("X"))

        ctx.set("X", si.OptimTensor(onnxl.TensorProto.FLOAT, [2, 3]))
        self.assertTrue(ctx.has("X"))
        self.assertEqual(ctx.size(), 1)
        self.assertFalse(ctx.empty())
        self.assertIn("X", ctx.names())

        x = ctx.get("X")
        self.assertEqual(x.dtype, onnxl.TensorProto.FLOAT)
        self.assertEqual(list(x.shape), [2, 3])

        ctx.clear()
        self.assertTrue(ctx.empty())

    def test_shapes_context_opset_versions(self):
        ctx = si.ShapesContext()
        self.assertFalse(ctx.has_opset_version(""))
        self.assertEqual(ctx.opset_version(""), si.kUnknownOpsetVersion)

        ctx.set_opset_version("", 18)
        # The empty domain is normalised to ``ai.onnx``.
        self.assertTrue(ctx.has_opset_version(""))
        self.assertTrue(ctx.has_opset_version(si.kOnnxDomain))
        self.assertEqual(ctx.opset_version(""), 18)
        self.assertEqual(ctx.opset_version(si.kOnnxDomain), 18)
        self.assertEqual(ctx.opsets()[si.kOnnxDomain], 18)

    # ------------------------------------------------------------------
    # compute_shape_node / compute_shape_graph / compute_shape_model.
    # ------------------------------------------------------------------
    def test_compute_shape_node_relu(self):
        ctx = si.ShapesContext()
        ctx.set("X", si.OptimTensor(onnxl.TensorProto.FLOAT, [2, 3]))
        node = oh.make_node("Relu", inputs=["X"], outputs=["Y"])
        si.compute_shape_node(ctx, node)
        self.assertTrue(ctx.has("Y"))
        y = ctx.get("Y")
        self.assertEqual(y.dtype, onnxl.TensorProto.FLOAT)
        self.assertEqual(list(y.shape), [2, 3])

    def test_compute_shape_node_unsupported_op(self):
        ctx = si.ShapesContext()
        ctx.set("X", si.OptimTensor(onnxl.TensorProto.FLOAT, [2, 3]))
        # ``ComputeShapeNode`` should reject an unknown op.
        node = oh.make_node("ThisOpDoesNotExist", inputs=["X"], outputs=["Y"])
        with self.assertRaises(ValueError):
            si.compute_shape_node(ctx, node)

    def test_check_inputs_available(self):
        ctx = si.ShapesContext()
        node = oh.make_node("Relu", inputs=["X"], outputs=["Y"])
        with self.assertRaises(ValueError):
            si.check_inputs_available(ctx, node)
        ctx.set("X", si.OptimTensor(onnxl.TensorProto.FLOAT, [1]))
        si.check_inputs_available(ctx, node)  # should not raise

    def test_compute_shape_model_end_to_end(self):
        node = oh.make_node("Relu", inputs=["X"], outputs=["Y"])
        x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, ["N", 4])
        y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, None)
        graph = oh.make_graph([node], "g", [x], [y])
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
        model.ir_version = 8

        ctx = si.ShapesContext()
        si.compute_shape_model(ctx, model)
        self.assertTrue(ctx.has("X"))
        self.assertTrue(ctx.has("Y"))
        self.assertEqual(ctx.opset_version(""), 18)

        y_desc = ctx.get("Y")
        self.assertEqual(y_desc.dtype, onnxl.TensorProto.FLOAT)
        self.assertEqual(list(y_desc.shape), ["N", 4])

        # apply_inferred_shapes_to_model writes the inferred shape back.
        si.apply_inferred_shapes_to_model(ctx, model)
        out_dims = list(model.graph.output[0].type.tensor_type.shape.dim)
        self.assertEqual(len(out_dims), 2)
        self.assertTrue(out_dims[0].has_dim_param())
        self.assertEqual(out_dims[0].dim_param, "N")
        self.assertEqual(out_dims[1].dim_value, 4)


if __name__ == "__main__":
    unittest.main()
