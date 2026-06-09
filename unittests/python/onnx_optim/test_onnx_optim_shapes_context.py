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

    def test_shapes_context_repr(self):
        ctx = si.ShapesContext()
        self.assertEqual(repr(ctx), "ShapesContext(tensors=[], sequences=[], opsets={})")

        ctx.set("X", si.OptimTensor(onnxl.TensorProto.FLOAT, [2, 3]))
        ctx.set_opset_version("", 18)
        r = repr(ctx)
        self.assertIn("ShapesContext(", r)
        self.assertIn("tensors=['X']", r)
        self.assertIn("sequences=[]", r)
        self.assertIn("opsets={'ai.onnx': 18}", r)

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

    def test_compute_shape_model_prefill_prefers_anchor(self):
        node = oh.make_node("Relu", inputs=["X"], outputs=["Y"])
        x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, ["N", 4])
        y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, ["ANCHOR", 4])
        graph = oh.make_graph([node], "g", [x], [y])
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
        model.ir_version = 8

        ctx_no_prefill = si.ShapesContext()
        si.compute_shape_model(ctx_no_prefill, model)
        self.assertEqual(list(ctx_no_prefill.get("Y").shape), ["N", 4])

        ctx_prefill = si.ShapesContext()
        si.compute_shape_model(ctx_prefill, model, prefill_with_value_info_output=True)
        self.assertEqual(list(ctx_prefill.get("Y").shape), ["ANCHOR", 4])
        # A constraint linking the inferred symbol to the anchor symbol is
        # recorded so downstream passes can unify them.
        self.assertTrue(ctx_prefill.has_constraint("N", "ANCHOR"))
        self.assertTrue(ctx_prefill.has_constraint("ANCHOR", "N"))
        self.assertEqual(ctx_prefill.constraints_size(), 1)
        self.assertEqual(list(ctx_prefill.constraints()), [("ANCHOR", "N")])

    # ------------------------------------------------------------------
    # Symbolic-dimension equality constraints.
    # ------------------------------------------------------------------
    def test_add_constraint_canonical_and_dedup(self):
        ctx = si.ShapesContext()
        self.assertEqual(ctx.constraints_size(), 0)
        self.assertFalse(ctx.has_constraint("N", "M"))

        self.assertTrue(ctx.add_constraint("N", "M"))
        self.assertEqual(ctx.constraints_size(), 1)
        # Canonical ordering: smaller first.
        self.assertEqual(list(ctx.constraints()), [("M", "N")])

        # Inserting the reversed pair is a no-op.
        self.assertFalse(ctx.add_constraint("M", "N"))
        self.assertEqual(ctx.constraints_size(), 1)

        # Self constraint is dropped but lookup still returns True.
        self.assertFalse(ctx.add_constraint("X", "X"))
        self.assertEqual(ctx.constraints_size(), 1)
        self.assertTrue(ctx.has_constraint("X", "X"))

        # Lookup uses canonical order.
        self.assertTrue(ctx.has_constraint("N", "M"))
        self.assertTrue(ctx.has_constraint("M", "N"))

        ctx.clear()
        self.assertEqual(ctx.constraints_size(), 0)

    def test_compute_shape_model_prefill_raises_on_dim_conflict(self):
        # Reshape with target [-1, 2] applied to X[N,4] gives Y[?, 2]; the
        # anchor declares Y[ANCHOR, 4] which carries an incompatible
        # concrete dim (4 vs 2) → must raise.
        target = oh.make_node(
            "Constant",
            inputs=[],
            outputs=["target"],
            value=oh.make_tensor("target", onnxl.TensorProto.INT64, [2], [-1, 2]),
        )
        reshape = oh.make_node("Reshape", inputs=["X", "target"], outputs=["Y"])
        x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, ["N", 4])
        y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, ["ANCHOR", 4])
        graph = oh.make_graph([target, reshape], "g", [x], [y])
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
        model.ir_version = 8

        ctx = si.ShapesContext()
        with self.assertRaises(ValueError):
            si.compute_shape_model(ctx, model, prefill_with_value_info_output=True)

    def test_compute_shape_model_prefill_raises_on_dtype_conflict(self):
        node = oh.make_node("Relu", inputs=["X"], outputs=["Y"])
        x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, ["N", 4])
        # Anchor declares Y as DOUBLE, incompatible with Relu(FLOAT).
        y = oh.make_tensor_value_info("Y", onnxl.TensorProto.DOUBLE, ["N", 4])
        graph = oh.make_graph([node], "g", [x], [y])
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
        model.ir_version = 8

        ctx = si.ShapesContext()
        with self.assertRaises(ValueError):
            si.compute_shape_model(ctx, model, prefill_with_value_info_output=True)


if __name__ == "__main__":
    unittest.main()
