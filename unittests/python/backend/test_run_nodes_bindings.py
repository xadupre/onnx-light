# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Python bindings of the ``RunNode`` / ``RunGraph`` /
``RunFunction`` / ``RunModel`` dispatcher exposed by
:mod:`onnx_light.onnx_py._onnxpykernels`.

The dispatcher is exposed as the ``runtime`` submodule of
``_onnxpykernels`` (and also surfaced through the ``_onnxpy`` shim).
"""

from __future__ import annotations

import struct
import unittest

from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx import TensorProto
from onnx_light.onnx_lib import parser
from onnx_light.onnx_py._onnxpykernels import runtime as rt


def _make_float_tensor(name: str, values: list[float]):
    """Builds an owned :class:`Tensor` of dtype ``FLOAT`` from ``values``."""
    tp = TensorProto()
    tp.name = name
    tp.dims.append(len(values))
    tp.data_type = int(TensorProto.FLOAT)
    tp.raw_data = struct.pack(f"<{len(values)}f", *values)
    return rt.tensor_from_proto(tp)


def _make_int32_tensor(name: str, values: list[int]):
    """Builds an owned :class:`Tensor` of dtype ``INT32`` from ``values``."""
    tp = TensorProto()
    tp.name = name
    tp.dims.append(len(values))
    tp.data_type = int(TensorProto.INT32)
    tp.raw_data = struct.pack(f"<{len(values)}i", *values)
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

    def test_runtime_context_event_log_records_add_replace_remove(self):
        ctx = rt.RuntimeContext(rt.KernelContext(rt.default_opset(18)))
        ctx.events_enabled = True
        self.assertEqual(ctx.events(), [])

        ctx.set("x", _make_float_tensor("x", [1.0, -2.0]))
        ctx.put("x", _make_float_tensor("x", [3.0]))  # replace
        ctx.remove("x")
        ctx.remove("missing")  # no-op, does not log

        events = ctx.events()
        self.assertEqual([e.action for e in events], ["add", "replace", "remove"])
        self.assertEqual([e.name for e in events], ["x", "x", "x"])
        # ``set`` defaults to kind ``"input"``, ``put`` to ``"intermediate"``;
        # ``remove`` records ``"unknown"``.
        self.assertEqual([e.kind for e in events], ["input", "intermediate", "unknown"])
        # timestamps are non-decreasing nanoseconds since the Unix epoch.
        self.assertTrue(all(isinstance(e.timestamp_ns, int) for e in events))
        self.assertTrue(all(e.timestamp_ns > 0 for e in events))
        self.assertLessEqual(events[0].timestamp_ns, events[1].timestamp_ns)
        self.assertLessEqual(events[1].timestamp_ns, events[2].timestamp_ns)

        # Values are exposed as a list of length ``value_count`` (truncated
        # prefix of the fixed-size buffer).
        self.assertEqual(events[0].data_type, int(TensorProto.FLOAT))
        self.assertEqual(events[0].shape, [2])
        self.assertEqual(events[0].value_count, 2)
        self.assertEqual(events[0].values, [1.0, -2.0])
        self.assertEqual(events[1].value_count, 1)
        self.assertEqual(events[1].values, [3.0])
        # Remove events leave the data/shape/value_count fields empty.
        self.assertEqual(events[2].data_type, int(TensorProto.UNDEFINED))
        self.assertEqual(events[2].shape, [])
        self.assertEqual(events[2].value_count, 0)
        self.assertEqual(events[2].values, [])

        # ``as_dict`` exposes the same fields as a plain ``dict``.
        d0 = events[0].as_dict()
        self.assertEqual(d0["action"], "add")
        self.assertEqual(d0["kind"], "input")
        self.assertEqual(d0["name"], "x")
        self.assertEqual(d0["data_type"], int(TensorProto.FLOAT))
        self.assertEqual(d0["shape"], [2])
        self.assertEqual(d0["value_count"], 2)
        self.assertEqual(d0["values"], [1.0, -2.0])
        self.assertEqual(d0["string_values"], [])
        self.assertEqual(d0["timestamp_ns"], events[0].timestamp_ns)

        ctx.clear_events()
        self.assertEqual(ctx.events(), [])

    def test_runtime_context_event_log_truncates_large_tensors(self):
        ctx = rt.RuntimeContext(rt.KernelContext(rt.default_opset(18)))
        ctx.events_enabled = True
        # 9-element tensor: the buffer keeps only the first 8 entries,
        # data_type is set to -1 and shape is emptied to flag the truncation.
        ctx.put("big", _make_int32_tensor("big", list(range(9))))
        ev = ctx.events()[0]
        self.assertEqual(ev.data_type, -1)
        self.assertEqual(ev.shape, [])
        self.assertEqual(ev.value_count, 8)
        self.assertEqual(ev.values, [float(i) for i in range(8)])

    def test_runtime_context_events_capture_run_model_intermediates(self):
        model = parser.parse_model(_MODEL_SRC)
        ctx = rt.RuntimeContext(rt.KernelContext(rt.default_opset(18)))
        ctx.events_enabled = True
        ctx.set("x", _make_float_tensor("x", [-1.0, 2.0, -3.5]))
        ctx.set("z", _make_float_tensor("z", [10.0, 20.0, 30.0]))
        ctx.clear_events()
        rt.run_model(model, ctx)

        events = ctx.events()
        # The graph produces two intermediates (``t`` and ``y``), each
        # paired with a ``run_node`` event summarising the kernel
        # dispatch (domain, op_type, inputs, duration).
        produced = [(e.action, e.name, e.kind) for e in events]
        self.assertEqual(
            produced,
            [
                ("add", "t", "intermediate"),
                ("run_node", "", "unknown"),
                ("add", "y", "intermediate"),
                ("run_node", "", "unknown"),
            ],
        )
        run_node_events = [e for e in events if e.action == "run_node"]
        self.assertEqual([e.op_type for e in run_node_events], ["Abs", "Add"])
        self.assertEqual([e.op_domain for e in run_node_events], ["ai.onnx", "ai.onnx"])
        self.assertEqual([e.inputs for e in run_node_events], [["x"], ["t", "z"]])
        self.assertTrue(all(e.duration_ns >= 0 for e in run_node_events))

        # ``as_dict`` exposes the new run_node fields.
        d = run_node_events[0].as_dict()
        self.assertEqual(d["action"], "run_node")
        self.assertEqual(d["op_type"], "Abs")
        self.assertEqual(d["op_domain"], "ai.onnx")
        self.assertEqual(d["inputs"], ["x"])
        self.assertEqual(d["duration_ns"], run_node_events[0].duration_ns)

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
        with self.assertRaises((ValueError, RuntimeError)):
            rt.run_node(node, ctx)

    def test_run_model_without_graph_raises(self):
        # A model without a graph must be rejected by RunModel.
        from onnx_light.onnx import ModelProto

        empty = ModelProto()
        empty.ir_version = 10
        ctx = rt.RuntimeContext(rt.KernelContext(rt.default_opset(18)))
        with self.assertRaises((ValueError, RuntimeError)):
            rt.run_model(empty, ctx)


if __name__ == "__main__":
    unittest.main(verbosity=2)
