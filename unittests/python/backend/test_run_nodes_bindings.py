# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Python bindings of the ``RunNode`` / ``RunGraph`` /
``RunFunction`` / ``RunModel`` dispatcher exposed by
:mod:`onnx_light.onnx_py._onnxkernels`.

The dispatcher is exposed as the ``runtime`` submodule of
``_onnxkernels`` (and also surfaced through the ``_onnxpy`` shim).
"""

from __future__ import annotations

import struct
import unittest

from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx_lib import parser
from onnx_light.onnx_proto._onnxpy import TensorProto
from onnx_light.onnx_py._onnxkernels import runtime as rt


def _make_float_tensor(name: str, values: list[float]):
    """Builds an owned :class:`Tensor` of dtype ``FLOAT`` from ``values``."""
    tp = TensorProto()
    tp.name = name
    tp.dims.append(len(values))
    tp.data_type = int(TensorProto.FLOAT)
    tp.raw_data = struct.pack(f"<{len(values)}f", *values)
    return rt.tensor_from_proto(tp)


def _unpack_floats(tensor) -> tuple[float, ...]:
    return struct.unpack(f"<{tensor.element_count()}f", tensor.raw_data())


# Parsed once at module load — every test that needs it copies the model
# (or addresses individual nodes) from this shared instance.
_MODEL_SRC = (
    '<ir_version: 10, opset_import: ["" : 18]>\n'
    "agraph (float[3] x, float[3] z) => (float[3] y) {\n"
    "  t = Abs(x)\n"
    "  y = Add(t, z)\n"
    "}\n"
)


class TestRunNodesBindings(ExtTestCase):
    def test_runtime_submodule_exposes_expected_names(self):
        for name in [
            "OpsetId",
            "KernelContext",
            "RuntimeContext",
            "default_opset",
            "tensor_from_proto",
            "run_node",
            "run_nodes",
            "run_graph",
            "run_function",
            "run_model",
        ]:
            self.assertTrue(hasattr(rt, name), name)

    def test_default_opset_and_kernel_context(self):
        opset = rt.default_opset(18)
        self.assertEqual(opset.domain, "")
        self.assertEqual(opset.version, 18)
        ctx = rt.KernelContext(opset)
        self.assertEqual(ctx.opset.version, 18)

    def test_runtime_context_set_get_has_remove_names(self):
        ctx = rt.RuntimeContext(rt.KernelContext(rt.default_opset(18)))
        x = _make_float_tensor("x", [1.0, -2.0])
        ctx.set("x", x)
        self.assertTrue(ctx.has("x"))
        self.assertEqual(sorted(ctx.names()), ["x"])
        self.assertEqual(_unpack_floats(ctx.get("x")), (1.0, -2.0))
        self.assertTrue(ctx.remove("x"))
        self.assertFalse(ctx.has("x"))
        self.assertFalse(ctx.remove("x"))

    def test_run_model_abs_then_add(self):
        model = parser.parse_model(_MODEL_SRC)
        ctx = rt.RuntimeContext(rt.KernelContext(rt.default_opset(18)))
        ctx.set("x", _make_float_tensor("x", [-1.0, 2.0, -3.5]))
        ctx.set("z", _make_float_tensor("z", [10.0, 20.0, 30.0]))
        rt.run_model(model, ctx)
        self.assertEqual(_unpack_floats(ctx.get("y")), (11.0, 22.0, 33.5))

    def test_run_graph_matches_run_model(self):
        model = parser.parse_model(_MODEL_SRC)
        ctx = rt.RuntimeContext(rt.KernelContext(rt.default_opset(18)))
        ctx.set("x", _make_float_tensor("x", [-4.0, 5.0, -6.0]))
        ctx.set("z", _make_float_tensor("z", [1.0, 1.0, 1.0]))
        rt.run_graph(model.graph, ctx)
        self.assertEqual(_unpack_floats(ctx.get("y")), (5.0, 6.0, 7.0))

    def test_run_node_executes_a_single_node(self):
        model = parser.parse_model(_MODEL_SRC)
        ctx = rt.RuntimeContext(rt.KernelContext(rt.default_opset(18)))
        ctx.set("x", _make_float_tensor("x", [-1.0, -2.0, -3.0]))
        # node[0] is Abs(x) -> t
        rt.run_node(model.graph.node[0], ctx)
        self.assertEqual(_unpack_floats(ctx.get("t")), (1.0, 2.0, 3.0))

    def test_run_nodes_executes_a_list(self):
        model = parser.parse_model(_MODEL_SRC)
        ctx = rt.RuntimeContext(rt.KernelContext(rt.default_opset(18)))
        ctx.set("x", _make_float_tensor("x", [-1.0, 1.0, -1.0]))
        ctx.set("z", _make_float_tensor("z", [100.0, 200.0, 300.0]))
        rt.run_nodes(list(model.graph.node), ctx)
        self.assertEqual(_unpack_floats(ctx.get("y")), (101.0, 201.0, 301.0))

    def test_run_node_unknown_op_raises(self):
        model = parser.parse_model(_MODEL_SRC)
        node = model.graph.node[0]
        # Force the node to reference an unsupported op_type.
        node.op_type = "ThisOpDoesNotExist"
        ctx = rt.RuntimeContext(rt.KernelContext(rt.default_opset(18)))
        ctx.set("x", _make_float_tensor("x", [1.0]))
        with self.assertRaises(Exception):
            rt.run_node(node, ctx)

    def test_run_model_without_graph_raises(self):
        # A model without a graph must be rejected by RunModel.
        from onnx_light.onnx_proto._onnxpy import ModelProto

        empty = ModelProto()
        empty.ir_version = 10
        ctx = rt.RuntimeContext(rt.KernelContext(rt.default_opset(18)))
        with self.assertRaises(Exception):
            rt.run_model(empty, ctx)


if __name__ == "__main__":
    unittest.main(verbosity=2)
