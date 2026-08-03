# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Python bindings of the ``RunNode`` dispatcher and the
``ExecutionPlan`` / ``RuntimeSession`` execution machinery (used to run any
node list, including a whole model's graph paired with
``register_model_functions``) exposed by
:mod:`onnx_light.onnx_py._onnxpykernels`.

The dispatcher is exposed as the ``runtime`` submodule of
``_onnxpykernels`` (and also surfaced through the ``_onnxpy`` shim).
"""

from __future__ import annotations

import gc
import struct
import tempfile
import unittest
import weakref
from pathlib import Path

import numpy as np

from onnx_light.ext_test_case import ExtTestCase, import_or_skip
from onnx_light.onnx import TensorProto
from onnx_light.onnx_lib import parser

# The kernels runtime is only available in the full build; skip this module on a
# reduced build (ONNX_LIGHT_BUILD_KERNELS=OFF).
rt = import_or_skip("onnx_light.onnx_py._onnxpykernels", "runtime")
bt = import_or_skip("onnx_light.onnx_py._onnxpybackend", "backend_test")


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


def _run_model(model, ctx) -> None:
    """Runs ``model``'s graph through ``ctx``, mirroring what the removed
    ``run_model`` binding used to do: registers every model-local function,
    seeds the graph's initializers (names ``ctx`` already carries are left
    as-is), then builds the graph's :class:`ExecutionPlan` and drives it
    through a fresh :class:`RuntimeSession`.
    """
    rt.register_model_functions(model, ctx)
    for init in model.graph.initializer:
        if not ctx.has(init.name):
            ctx.set(init.name, rt.tensor_from_proto(init), "initializer")
    plan = rt.ExecutionPlan(model.graph)
    rt.RuntimeSession(plan).run(ctx)


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
            "RuntimeParameters",
            "RuntimeEvent",
            "RuntimeEventAction",
            "ExecutionPlan",
            "RuntimeSession",
            "default_opset",
            "tensor_from_proto",
            "tensor_to_proto",
            "tensor_to_numpy",
            "run_node",
            "register_model_functions",
        ]:
            self.assertTrue(hasattr(rt, name), name)

    def test_runtime_event_action_enum_values(self):
        self.assertEqual(int(rt.RuntimeEventAction.kAdd), 0)
        self.assertEqual(int(rt.RuntimeEventAction.kReplace), 1)
        self.assertEqual(int(rt.RuntimeEventAction.kRemove), 2)
        self.assertEqual(int(rt.RuntimeEventAction.kRunNode), 3)

    def test_runtime_session_from_model_builds_plan(self):
        # A session can be constructed directly from a ModelProto (no plan
        # supplied): it builds and owns the plan from ``model.graph``, so run()
        # executes the graph. Here a single-node Add graph is run against a
        # context supplying its two external inputs.
        model = parser.parse_model(
            '<ir_version: 10, opset_import: ["" : 18]>\n'
            "agraph (float[2] x, float[2] y) => (float[2] z) {\n"
            "  z = Add(x, y)\n"
            "}\n"
        )
        ctx = rt.RuntimeContext(rt.KernelContext(rt.default_opset(18)))
        ctx.set("x", _make_float_tensor("x", [1.0, -2.0]))
        ctx.set("y", _make_float_tensor("y", [10.0, 20.0]))
        session = rt.RuntimeSession(model)
        session.run(ctx)
        self.assertEqual(sorted(session.required_inputs), ["x", "y"])
        self.assertEqual(_unpack_floats(ctx.get("z")), (11.0, 18.0))

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

    def test_runtime_parameters_semantics(self):
        params = rt.RuntimeParameters()
        self.assertEqual(params.num_threads, 0)
        self.assertGreaterEqual(params.effective_num_threads(), 1)

        sequential = rt.RuntimeParameters(1)
        self.assertEqual(sequential.num_threads, 1)
        self.assertEqual(sequential.effective_num_threads(), 1)
        self.assertFalse(sequential.is_parallel())

        explicit = rt.RuntimeParameters(4)
        self.assertEqual(explicit.effective_num_threads(), 4)
        self.assertTrue(explicit.is_parallel())

        negative = rt.RuntimeParameters(-3)
        self.assertGreaterEqual(negative.effective_num_threads(), 1)

    def test_runtime_context_event_log_records_add_replace_remove(self):
        ctx = rt.RuntimeContext(rt.KernelContext(rt.default_opset(18)), events_enabled=True)
        self.assertEqual(ctx.events(), [])

        ctx.set("x", _make_float_tensor("x", [1.0, -2.0]))
        ctx.put("x", _make_float_tensor("x", [3.0]))  # replace
        ctx.remove("x")
        ctx.remove("missing")  # no-op, does not log

        events = ctx.events()
        self.assertEqual(
            [e.action for e in events],
            [
                rt.RuntimeEventAction.kAdd,
                rt.RuntimeEventAction.kReplace,
                rt.RuntimeEventAction.kRemove,
            ],
        )
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
        # ``set`` defaults to the ``input`` kind: node_index = -1, CPU device.
        self.assertEqual(d0["node_index"], -1)
        self.assertEqual(d0["device"], -1)
        self.assertEqual(events[0].node_index, -1)
        self.assertEqual(events[0].device, -1)

        ctx.clear_events()
        self.assertEqual(ctx.events(), [])

    def test_run_model_delayed_initializer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            weights = Path(temp_dir) / "weights.bin"
            weights.write_bytes(b"\x00" * 8 + struct.pack("<2f", 1.25, -3.5))

            model = parser.parse_model(f"""
                <
                  ir_version: 10,
                  opset_import: ["" : 18, "ai.rt" : 1]
                >
                agraph () => (float[2] Y)
                {{
                    Y = ai.rt.DelayedInitializer<
                        shape = [2],
                        dtype = {int(TensorProto.FLOAT)},
                        load_device = "file",
                        runtime_device = "cpu",
                        filename = "{weights.as_posix()}",
                        offset = 8
                    >()
                }}
                """)
            ctx = rt.RuntimeContext(rt.KernelContext(rt.default_opset(18)))
            _run_model(model, ctx)
            y = ctx.get("Y")
            self.assertEqual(_unpack_floats(y), (1.25, -3.5))

    def test_runtime_context_event_log_truncates_large_tensors(self):
        ctx = rt.RuntimeContext(rt.KernelContext(rt.default_opset(18)), events_enabled=True)
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
        ctx = rt.RuntimeContext(rt.KernelContext(rt.default_opset(18)), events_enabled=True)
        ctx.set("x", _make_float_tensor("x", [-1.0, 2.0, -3.5]))
        ctx.set("z", _make_float_tensor("z", [10.0, 20.0, 30.0]))
        ctx.clear_events()
        _run_model(model, ctx)

        events = ctx.events()
        # The graph produces two intermediates (``t`` and ``y``), each
        # paired with a ``run_node`` event summarising the kernel
        # dispatch (domain, op_type, inputs, duration).
        produced = [(e.action, e.name, e.kind) for e in events]
        self.assertEqual(
            produced,
            [
                (rt.RuntimeEventAction.kAdd, "t", "intermediate"),
                (rt.RuntimeEventAction.kRunNode, "", "unknown"),
                (rt.RuntimeEventAction.kAdd, "y", "intermediate"),
                (rt.RuntimeEventAction.kRunNode, "", "unknown"),
            ],
        )
        run_node_events = [e for e in events if e.action == rt.RuntimeEventAction.kRunNode]
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

        # ``node_index`` tags inputs with ``-1`` and intermediates / run_node
        # events with the index of the producing node; ``device`` is ``-1``
        # (CPU) for the reference runtime.
        node_indices = [(e.action, e.name, e.node_index) for e in events]
        self.assertEqual(
            node_indices,
            [
                (rt.RuntimeEventAction.kAdd, "t", 0),
                (rt.RuntimeEventAction.kRunNode, "", 0),
                (rt.RuntimeEventAction.kAdd, "y", 1),
                (rt.RuntimeEventAction.kRunNode, "", 1),
            ],
        )
        self.assertTrue(all(e.device == -1 for e in events))
        self.assertEqual(d["node_index"], 0)
        self.assertEqual(d["device"], -1)

    def test_runtime_context_events_capture_allocator_memory(self):
        # With a SimpleRawBufferAllocator attached, every recorded event carries
        # the allocator's live (allocated_bytes) and peak (peak_bytes) memory,
        # and RuntimeEvent.summary() renders a one-line recap including them.
        model = parser.parse_model(_MODEL_SRC)
        alloc = rt.SimpleRawBufferAllocator(16)
        ctx = rt.RuntimeContext(
            rt.KernelContext(rt.default_opset(18)), events_enabled=True, allocator=alloc
        )
        ctx.set("x", _make_float_tensor("x", [-1.0, 2.0, -3.5]))
        ctx.set("z", _make_float_tensor("z", [10.0, 20.0, 30.0]))
        ctx.clear_events()
        _run_model(model, ctx)

        events = ctx.events()
        self.assertNotEqual(events, [])
        # The two 3-float inputs already occupy the allocator, so every event
        # reports a strictly positive live footprint whose peak never drops
        # below the live value.
        for e in events:
            self.assertGreater(e.allocated_bytes, 0)
            self.assertGreaterEqual(e.peak_bytes, e.allocated_bytes)
            text = e.summary()
            self.assertIn("mem=", text)
            self.assertIn("peak=", text)
        # ``as_dict`` mirrors the new fields.
        d = events[0].as_dict()
        self.assertEqual(d["allocated_bytes"], events[0].allocated_bytes)
        self.assertEqual(d["peak_bytes"], events[0].peak_bytes)
        # The allocator itself exposes the live / peak counters.
        self.assertGreater(alloc.peak_allocated_size, 0)
        self.assertGreaterEqual(alloc.capacity, 16)

    def test_runtime_context_without_allocator_reports_zero_memory(self):
        # Without an allocator the memory fields default to 0 (no tracking).
        ctx = rt.RuntimeContext(rt.KernelContext(rt.default_opset(18)), events_enabled=True)
        ctx.set("x", _make_float_tensor("x", [1.0, -2.0]))
        ev = ctx.events()[0]
        self.assertEqual(ev.allocated_bytes, 0)
        self.assertEqual(ev.peak_bytes, 0)

    def test_run_model_abs_then_add(self):
        model = parser.parse_model(_MODEL_SRC)
        ctx = rt.RuntimeContext(rt.KernelContext(rt.default_opset(18)))
        ctx.set("x", _make_float_tensor("x", [-1.0, 2.0, -3.5]))
        ctx.set("z", _make_float_tensor("z", [10.0, 20.0, 30.0]))
        _run_model(model, ctx)
        self.assertEqual(_unpack_floats(ctx.get("y")), (11.0, 22.0, 33.5))

    def test_run_graph_matches_run_model(self):
        model = parser.parse_model(_MODEL_SRC)
        ctx = rt.RuntimeContext(rt.KernelContext(rt.default_opset(18)))
        ctx.set("x", _make_float_tensor("x", [-4.0, 5.0, -6.0]))
        ctx.set("z", _make_float_tensor("z", [1.0, 1.0, 1.0]))
        plan = rt.ExecutionPlan(model.graph)
        rt.RuntimeSession(plan).run(ctx)
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
        plan = rt.ExecutionPlan(model.graph)
        rt.RuntimeSession(plan).run(ctx)
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
        # A model without a graph must be rejected by register_model_functions.
        from onnx_light.onnx import ModelProto

        empty = ModelProto()
        empty.ir_version = 10
        ctx = rt.RuntimeContext(rt.KernelContext(rt.default_opset(18)))
        with self.assertRaises((ValueError, RuntimeError)):
            _run_model(empty, ctx)

    def test_register_custom_kernel(self):
        # Register a Python custom kernel for a node in an unknown domain
        # and verify RunNode dispatches to it.
        model_src = (
            '<ir_version: 10, opset_import: ["" : 18, "my.domain" : 1]>\n'
            "agraph (float[3] x) => (float[3] y) {\n"
            "  y = my.domain.Triple(x)\n"
            "}\n"
        )
        model = parser.parse_model(model_src)
        ctx = rt.RuntimeContext(rt.KernelContext(rt.default_opset(18)))
        ctx.set("x", _make_float_tensor("x", [1.0, 2.0, 3.0]))

        called = []

        def triple(node, c):
            called.append(str(node.op_type))
            x = c.get(str(node.input[0]))
            # Materialize a 3*x tensor via a TensorProto round-trip.
            tp = TensorProto()
            tp.name = str(node.output[0])
            tp.dims.append(3)
            tp.data_type = int(TensorProto.FLOAT)
            vals = struct.unpack("<3f", bytes(x.raw_data()))
            tp.raw_data = struct.pack("<3f", *(v * 3.0 for v in vals))
            c.put(tp.name, rt.tensor_from_proto(tp), "output")

        ctx.register_custom_kernel("my.domain", "Triple", triple)
        _run_model(model, ctx)
        self.assertEqual(called, ["Triple"])
        self.assertEqual(_unpack_floats(ctx.get("y")), (3.0, 6.0, 9.0))

    def test_register_custom_kernel_overrides_builtin(self):
        # Custom kernels registered on the runtime context override the
        # built-in dispatch table.
        model = parser.parse_model(_MODEL_SRC)
        ctx = rt.RuntimeContext(rt.KernelContext(rt.default_opset(18)))
        ctx.set("x", _make_float_tensor("x", [-1.0, -2.0, -3.0]))
        ctx.set("z", _make_float_tensor("z", [0.0, 0.0, 0.0]))

        def fake_abs(node, c):
            # Replace Abs with negation to prove the override applies.
            x = c.get(str(node.input[0]))
            vals = struct.unpack("<3f", bytes(x.raw_data()))
            tp = TensorProto()
            tp.name = str(node.output[0])
            tp.dims.append(3)
            tp.data_type = int(TensorProto.FLOAT)
            tp.raw_data = struct.pack("<3f", *(-v for v in vals))
            c.put(tp.name, rt.tensor_from_proto(tp), "output")

        ctx.register_custom_kernel("", "Abs", fake_abs)
        plan = rt.ExecutionPlan(model.graph)
        rt.RuntimeSession(plan).run(ctx)
        # Abs replaced by negation: -(-1) = 1, -(-2) = 2, -(-3) = 3, +0.
        self.assertEqual(_unpack_floats(ctx.get("y")), (1.0, 2.0, 3.0))

    def test_unregister_custom_kernel_releases_python_function(self):
        ctx = rt.RuntimeContext(rt.KernelContext(rt.default_opset(18)))

        def custom(_node, _ctx):
            return None

        custom_ref = weakref.ref(custom)
        ctx.register_custom_kernel("my.domain", "Tmp", custom)
        self.assertTrue(ctx.unregister_custom_kernel("my.domain", "Tmp"))
        self.assertFalse(ctx.unregister_custom_kernel("my.domain", "Tmp"))

        del custom
        gc.collect()
        self.assertIsNone(custom_ref())


class TestExecutionPlanBindings(ExtTestCase):
    _PLAN_SRC = (
        '<ir_version: 10, opset_import: ["" : 18]>\n'
        "agraph (float[3] x) => (float[3] y) {\n"
        "  t0 = Add(x, x)\n"
        "  t1 = Mul(t0, t0)\n"
        "  y = Sub(t1, t0)\n"
        "}\n"
    )

    def test_execution_plan_from_graph(self):
        model = parser.parse_model(self._PLAN_SRC)
        plan = rt.ExecutionPlan(model.graph)
        self.assertEqual(plan.num_nodes, 3)
        self.assertEqual(sorted(plan.keep()), ["x", "y"])
        # An unannotated graph still yields an execute step per node.
        execute_indices = [a.node_index for a in plan.actions() if a.kind_name == "ExecuteNode"]
        self.assertEqual(execute_indices, [0, 1, 2])

    def test_default_execution_plan_is_empty(self):
        plan = rt.ExecutionPlan()
        self.assertEqual(plan.num_nodes, 0)
        self.assertEqual(plan.keep(), set())
        self.assertEqual(plan.actions(), [])

    def test_get_execution_plan_is_cached(self):
        model = parser.parse_model(self._PLAN_SRC)
        ctx = rt.RuntimeContext(rt.KernelContext(rt.default_opset(18)))
        plan = ctx.get_execution_plan(model.graph)
        self.assertEqual(plan.num_nodes, 3)
        # Same graph object -> same cached plan contents.
        plan_again = ctx.get_execution_plan(model.graph)
        self.assertEqual(plan_again.num_nodes, plan.num_nodes)
        ctx.clear_execution_plans()

    def test_release_after_frees_intermediates(self):
        model = parser.parse_model(self._PLAN_SRC)
        ctx = rt.RuntimeContext(rt.KernelContext(rt.default_opset(18)))
        plan = ctx.get_execution_plan(model.graph)
        for name in ("t0", "t1"):
            ctx.set(name, _make_float_tensor(name, [1.0, 2.0, 3.0]))
        last_node = model.graph.node[2]
        plan.release_after(last_node, ctx)
        self.assertFalse(ctx.has("t0"))
        self.assertFalse(ctx.has("t1"))

    def test_execute_action_construction(self):
        action = rt.ExecuteAction(rt.ExecuteActionKind.kLockInput, "x")
        self.assertEqual(action.kind, rt.ExecuteActionKind.kLockInput)
        self.assertEqual(action.kind_name, "LockInput")
        self.assertEqual(action.name, "x")
        self.assertEqual(action.target, "")
        self.assertEqual(action.node_index, 0)
        self.assertEqual(action.size, 0)
        self.assertIn("LockInput", repr(action))

    def test_execution_plan_actions_shape_tag(self):
        model = parser.parse_model(self._PLAN_SRC)
        node = model.graph.node
        # "t1" is released at the final node and value-tagged as a shape; "t0" is
        # a regular data result also released there. The input "x" reaches its
        # last use at the first node.
        node[0].add_metadata("onnx_light.not_used_after", "x")
        node[2].add_metadata("onnx_light.release_after", "t1;t0")
        node[2].add_metadata("onnx_light.release_after_shape_tag", "t1")
        plan = rt.ExecutionPlan(model.graph)
        actions = plan.actions()
        # A shape is created for "t1" (no data buffer) and destroyed on release.
        self.assertIn(("CreateShape", "t1"), [(a.kind_name, a.name) for a in actions])
        self.assertIn(("DeleteShape", "t1"), [(a.kind_name, a.name) for a in actions])
        # "t0" is a result: it is allocated and freed as a buffer.
        self.assertIn(("AllocateBuffer", "t0"), [(a.kind_name, a.name) for a in actions])
        self.assertIn(("DeleteBuffer", "t0"), [(a.kind_name, a.name) for a in actions])
        # The shape value is never allocated / freed as a buffer.
        self.assertNotIn(("AllocateBuffer", "t1"), [(a.kind_name, a.name) for a in actions])
        self.assertNotIn(("DeleteBuffer", "t1"), [(a.kind_name, a.name) for a in actions])

    def test_execution_plan_actions_peak_memory(self):
        model = parser.parse_model(self._PLAN_SRC)
        node = model.graph.node
        # The first node needs a 512-byte scratch/temporary buffer to handle its
        # memory peak; the others need none.
        node[0].add_metadata("onnx_light.peak_memory", "512")
        node[2].add_metadata("onnx_light.release_after", "t1;t0")
        plan = rt.ExecutionPlan(model.graph)
        actions = plan.actions()
        # A temporary buffer is allocated before node 0 runs and deleted after,
        # sized from the peak-memory metadata.
        alloc = [a for a in actions if a.kind_name == "AllocateTemporaryBuffer"]
        delete = [a for a in actions if a.kind_name == "DeleteTemporaryBuffer"]
        self.assertEqual(len(alloc), 1)
        self.assertEqual(len(delete), 1)
        self.assertEqual(alloc[0].node_index, 0)
        self.assertEqual(alloc[0].size, 512)
        self.assertEqual(delete[0].node_index, 0)
        self.assertEqual(delete[0].size, 512)
        kinds = [a.kind_name for a in actions]
        alloc_index = kinds.index("AllocateTemporaryBuffer")
        delete_index = kinds.index("DeleteTemporaryBuffer")
        execute0_index = next(
            i for i, a in enumerate(actions) if a.kind_name == "ExecuteNode" and a.node_index == 0
        )
        # The scratch buffer wraps the node execution.
        self.assertLess(alloc_index, execute0_index)
        self.assertLess(execute0_index, delete_index)

    def test_default_execution_plan_has_no_actions(self):
        plan = rt.ExecutionPlan()
        self.assertEqual(plan.actions(), [])

    def test_execution_plan_actions_from_graph(self):
        model = parser.parse_model(self._PLAN_SRC)
        # BuildActions is metadata-driven, so annotate the nodes with the
        # in-place / lifetime metadata the in-place reuse pass would write.
        node = model.graph.node
        # Add(x, x) reuses input 0 ("x") in place for its output "t0".
        node[0].add_metadata("onnx_light.inplace_reuse", "0:0:equal")
        node[0].add_metadata("onnx_light.not_used_after", "x")
        # "t0" and "t1" reach their last use at the final Sub node.
        node[2].add_metadata("onnx_light.release_after", "t1;t0")
        plan = rt.ExecutionPlan(model.graph)
        kinds = [(a.kind_name, a.name) for a in plan.actions()]
        # The single input is locked first and, since it is only read by the
        # first node, unlocked right after that node runs.
        self.assertEqual(kinds[0], ("LockInput", "x"))
        self.assertIn(("UnlockInput", "x"), kinds)
        first_execute = kinds.index(("ExecuteNode", ""))
        self.assertGreater(kinds.index(("UnlockInput", "x")), first_execute)
        # Exactly one ExecuteNode action per node, in order.
        execute_indices = [a.node_index for a in plan.actions() if a.kind_name == "ExecuteNode"]
        self.assertEqual(execute_indices, [0, 1, 2])
        # The intermediates are freed; the output ``y`` is kept.
        deleted = {a.name for a in plan.actions() if a.kind_name == "DeleteBuffer"}
        self.assertEqual(deleted, {"t0", "t1"})
        self.assertNotIn("y", deleted)
        # "t0" is produced in place from "x": its allocation carries the reuse
        # information and does not reference an allocator.
        t0_alloc = [
            a for a in plan.actions() if a.kind_name == "AllocateBuffer" and a.name == "t0"
        ]
        self.assertEqual(len(t0_alloc), 1)
        self.assertTrue(t0_alloc[0].is_inplace)
        self.assertEqual(t0_alloc[0].target, "x")
        self.assertEqual(t0_alloc[0].inplace_output_index, 0)
        self.assertEqual(t0_alloc[0].inplace_input_index, 0)
        # "t1" has no reuse opportunity: it is allocated fresh.
        t1_alloc = [
            a for a in plan.actions() if a.kind_name == "AllocateBuffer" and a.name == "t1"
        ]
        self.assertEqual(len(t1_alloc), 1)
        self.assertFalse(t1_alloc[0].is_inplace)


class TestTensorToProto(ExtTestCase):
    def test_tensor_to_proto_numeric_roundtrip(self):
        t = _make_float_tensor("x", [1.0, 2.0, 3.0])
        tp = rt.tensor_to_proto(t)
        self.assertEqual(tp.name, "x")
        self.assertEqual(int(tp.data_type), int(TensorProto.FLOAT))
        self.assertEqual(list(tp.dims), [3])
        self.assertEqual(struct.unpack("<3f", bytes(tp.raw_data)), (1.0, 2.0, 3.0))

    def test_tensor_to_proto_keeps_source_alive(self):
        # ``raw_data`` borrows the tensor's byte buffer (zero-copy); the
        # ``keep_alive`` policy must keep the source tensor alive so the
        # borrowed view stays valid after the Python handle is dropped.
        tp = rt.tensor_to_proto(_make_int32_tensor("v", [7, 8, 9]))
        self.assertEqual(struct.unpack("<3i", bytes(tp.raw_data)), (7, 8, 9))

    def test_tensor_to_proto_string_tensor(self):
        sp = TensorProto()
        sp.name = "s"
        sp.dims.append(2)
        sp.data_type = int(TensorProto.STRING)
        sp.string_data.append(b"abc")
        sp.string_data.append(b"de")
        t = rt.tensor_from_proto(sp)
        tp = rt.tensor_to_proto(t)
        self.assertEqual(int(tp.data_type), int(TensorProto.STRING))
        self.assertEqual(list(tp.string_data), [b"abc", b"de"])


class TestTensorToNumpy(ExtTestCase):
    def test_tensor_to_numpy_returns_raw_uint8_view(self):
        t = _make_float_tensor("x", [1.0, 2.0, 3.0])
        raw = rt.tensor_to_numpy(t)
        self.assertEqual(raw.dtype, np.uint8)
        self.assertEqual(raw.ndim, 1)
        self.assertEqual(raw.shape, (12,))
        np.testing.assert_array_equal(raw.view(np.float32), np.array([1.0, 2.0, 3.0], np.float32))

    def test_tensor_to_numpy_is_zero_copy(self):
        # The returned uint8 view borrows the tensor's bytes: it must not own
        # its data (``base`` is set) so no copy was made.
        t = _make_int32_tensor("v", [7, 8, 9])
        raw = rt.tensor_to_numpy(t)
        self.assertIsNotNone(raw.base)
        np.testing.assert_array_equal(raw.view(np.int32), np.array([7, 8, 9], np.int32))

    def test_tensor_to_numpy_keeps_source_alive(self):
        # Dropping the Python tensor handle must not invalidate the borrowed
        # view: the array keeps the source tensor alive through its ``base``.
        arr = rt.tensor_to_numpy(_make_int32_tensor("v", [7, 8, 9])).view(np.int32)
        np.testing.assert_array_equal(arr, np.array([7, 8, 9], np.int32))

    def test_tensor_to_numpy_string_tensor_raises(self):
        sp = TensorProto()
        sp.name = "s"
        sp.dims.append(1)
        sp.data_type = int(TensorProto.STRING)
        sp.string_data.append(b"abc")
        t = rt.tensor_from_proto(sp)
        with self.assertRaises(ValueError):
            rt.tensor_to_numpy(t)

    def test_tensor_to_numpy_steal_transfers_ownership(self):
        # ``steal=True`` transfers ownership of an inline-owned buffer to the
        # NumPy array: the array owns the bytes (``base`` is a capsule, not the
        # source tensor) and the source tensor is emptied.
        t = _make_int32_tensor("v", [7, 8, 9])
        raw = rt.tensor_to_numpy(t, steal=True)
        self.assertIsNotNone(raw.base)
        # The source tensor no longer holds the bytes (they were moved out).
        self.assertEqual(t.raw_data(), b"")
        # The array remains valid even though the source was emptied/dropped.
        np.testing.assert_array_equal(raw.view(np.int32), np.array([7, 8, 9], np.int32))

    def test_tensor_to_numpy_steal_survives_source_drop(self):
        # After an ownership transfer the array keeps its data through the
        # capsule, so dropping the source tensor must not invalidate it.
        arr = rt.tensor_to_numpy(_make_int32_tensor("v", [1, 2, 3]), steal=True).view(np.int32)
        np.testing.assert_array_equal(arr, np.array([1, 2, 3], np.int32))

    def test_tensor_to_numpy_default_does_not_steal(self):
        # The default (``steal=False``) borrows the bytes and leaves the source
        # tensor's buffer intact.
        t = _make_int32_tensor("v", [7, 8, 9])
        raw = rt.tensor_to_numpy(t)
        self.assertEqual(len(t.raw_data()), 12)
        np.testing.assert_array_equal(raw.view(np.int32), np.array([7, 8, 9], np.int32))


class TestTensorDLPack(ExtTestCase):
    def _make_tensor(self, dtype_enum, array: np.ndarray):
        tp = TensorProto()
        tp.name = "x"
        tp.data_type = int(dtype_enum)
        tp.dims.extend(array.shape)
        tp.raw_data = array.tobytes()
        return rt.tensor_from_proto(tp)

    def test_dlpack_device_is_cpu(self):
        t = _make_float_tensor("x", [1.0, 2.0, 3.0])
        # DLPack device tuple is (device_type, device_id); 1 == kDLCPU.
        self.assertEqual(t.__dlpack_device__(), (1, 0))

    def test_dlpack_returns_capsule(self):
        t = _make_float_tensor("x", [1.0, 2.0, 3.0])
        capsule = t.__dlpack__()
        self.assertEqual(type(capsule).__name__, "PyCapsule")

    def test_from_dlpack_float(self):
        expected = np.arange(6, dtype=np.float32).reshape(2, 3)
        out = np.from_dlpack(self._make_tensor(TensorProto.FLOAT, expected))
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.shape, (2, 3))
        np.testing.assert_array_equal(out, expected)

    def test_from_dlpack_is_zero_copy(self):
        # The exported buffer is shared with the source tensor: the resulting
        # array must not own its data (``base`` is set) so no copy was made.
        out = np.from_dlpack(_make_int32_tensor("v", [7, 8, 9]))
        self.assertIsNotNone(out.base)
        np.testing.assert_array_equal(out, np.array([7, 8, 9], np.int32))

    def test_from_dlpack_keeps_source_alive(self):
        # Dropping the Python tensor handle must not invalidate the shared
        # buffer: the array keeps the source tensor alive through the capsule.
        out = np.from_dlpack(_make_int32_tensor("v", [7, 8, 9]))
        np.testing.assert_array_equal(out, np.array([7, 8, 9], np.int32))

    def test_from_dlpack_various_dtypes(self):
        cases = [
            (TensorProto.DOUBLE, np.array([1.5, 2.5], dtype=np.float64)),
            (TensorProto.FLOAT16, np.array([1.5, 2.5], dtype=np.float16)),
            (TensorProto.INT8, np.array([-1, 2, 3], dtype=np.int8)),
            (TensorProto.UINT8, np.array([1, 2, 3], dtype=np.uint8)),
            (TensorProto.INT16, np.array([1, -2], dtype=np.int16)),
            (TensorProto.UINT16, np.array([1, 2], dtype=np.uint16)),
            (TensorProto.INT32, np.array([1, -2], dtype=np.int32)),
            (TensorProto.UINT32, np.array([1, 2], dtype=np.uint32)),
            (TensorProto.INT64, np.array([[1, 2], [3, 4]], dtype=np.int64)),
            (TensorProto.UINT64, np.array([1, 2], dtype=np.uint64)),
        ]
        for dtype_enum, array in cases:
            with self.subTest(dtype=dtype_enum):
                out = np.from_dlpack(self._make_tensor(dtype_enum, array))
                self.assertEqual(out.dtype, array.dtype)
                np.testing.assert_array_equal(out, array)

    def test_from_dlpack_bool(self):
        expected = np.array([True, False, True])
        out = np.from_dlpack(self._make_tensor(TensorProto.BOOL, expected))
        self.assertEqual(out.dtype, np.bool_)
        np.testing.assert_array_equal(out, expected)

    def test_from_dlpack_scalar(self):
        out = np.from_dlpack(self._make_tensor(TensorProto.INT32, np.array(7, dtype=np.int32)))
        self.assertEqual(out.shape, ())
        self.assertEqual(int(out), 7)

    def test_backend_test_tensor_from_dlpack(self):
        cases = bt.collect_test_cases("Abs")
        self.assertGreater(len(cases), 0)
        tensor = cases[0].data_sets[0].inputs[0]
        out = np.from_dlpack(tensor)
        self.assertEqual(out.shape, tuple(tensor.shape))
        self.assertEqual(out.size, tensor.element_count())

    def test_dlpack_string_tensor_raises(self):
        sp = TensorProto()
        sp.name = "s"
        sp.dims.append(1)
        sp.data_type = int(TensorProto.STRING)
        sp.string_data.append(b"abc")
        t = rt.tensor_from_proto(sp)
        with self.assertRaises(ValueError):
            t.__dlpack__()
        with self.assertRaises(ValueError):
            t.__dlpack_device__()


class TestSubgraphEventGraphName(ExtTestCase):
    """Verify that events produced inside subgraphs carry the correct
    ``subgraph_node_index`` and ``subgraph_attr_name`` fields, and that
    top-level events have ``subgraph_node_index == -1`` and an empty
    ``subgraph_attr_name``."""

    def _build_loop_model(self) -> object:
        """Builds a Loop model: Loop(M=2, cond=true, s_init=0.0) -> s_final.

        The body adds 1.0 to s_in each iteration.
        """
        from onnx_light.onnx import helper, TensorProto

        one_init = helper.make_tensor("one", TensorProto.FLOAT, [], [1.0])
        add = helper.make_node("Add", ["s_in", "one"], ["s_out"])
        body = helper.make_graph(
            [add],
            "loop_body",
            [
                helper.make_tensor_value_info("iter", TensorProto.INT64, []),
                helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
                helper.make_tensor_value_info("s_in", TensorProto.FLOAT, []),
            ],
            [
                helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
                helper.make_tensor_value_info("s_out", TensorProto.FLOAT, []),
                helper.make_tensor_value_info("s_out", TensorProto.FLOAT, []),
            ],
            initializer=[one_init],
        )
        loop_node = helper.make_node("Loop", ["M", "cond", "s_init"], ["s_final", "scan_out"])
        loop_node.attribute.append(helper.make_attribute("body", body))
        graph = helper.make_graph(
            [loop_node],
            "main",
            [
                helper.make_tensor_value_info("M", TensorProto.INT64, []),
                helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
                helper.make_tensor_value_info("s_init", TensorProto.FLOAT, []),
            ],
            [
                helper.make_tensor_value_info("s_final", TensorProto.FLOAT, []),
                helper.make_tensor_value_info("scan_out", TensorProto.FLOAT, None),
            ],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        model.ir_version = 10
        return model

    def _build_if_model(self) -> object:
        """Builds an If model: cond -> If(then_branch, else_branch)."""
        from onnx_light.onnx import helper, TensorProto

        def _const_branch(name: str, val: float) -> object:
            init = helper.make_tensor(name, TensorProto.FLOAT, [], [val])
            add = helper.make_node("Add", [name, name], ["out"])
            return helper.make_graph(
                [add],
                name + "_g",
                [],
                [helper.make_tensor_value_info("out", TensorProto.FLOAT, [])],
                initializer=[init],
            )

        if_node = helper.make_node("If", ["cond"], ["z"])
        if_node.attribute.append(helper.make_attribute("then_branch", _const_branch("t", 1.0)))
        if_node.attribute.append(helper.make_attribute("else_branch", _const_branch("e", 2.0)))
        graph = helper.make_graph(
            [if_node],
            "main",
            [helper.make_tensor_value_info("cond", TensorProto.BOOL, [])],
            [helper.make_tensor_value_info("z", TensorProto.FLOAT, [])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        model.ir_version = 10
        return model

    def test_top_level_events_have_empty_subgraph_attr_name(self):
        """A plain model without subgraphs has all events with empty subgraph_attr_name."""
        model = parser.parse_model(_MODEL_SRC)
        ctx = rt.RuntimeContext(rt.KernelContext(rt.default_opset(18)), events_enabled=True)
        ctx.set("x", _make_float_tensor("x", [-1.0, 2.0, -3.5]))
        ctx.set("z", _make_float_tensor("z", [10.0, 20.0, 30.0]))
        ctx.clear_events()
        _run_model(model, ctx)

        for ev in ctx.events():
            self.assertEqual(
                ev.subgraph_attr_name,
                "",
                f"Expected empty subgraph_attr_name for top-level event, "
                f"got: {ev.subgraph_attr_name!r}",
            )
            self.assertEqual(ev.subgraph_node_index, -1)

    def test_loop_subgraph_events_carry_body_attr_name(self):
        """At least one event from a Loop body carries subgraph_attr_name='body'."""
        import struct

        model = self._build_loop_model()
        ctx = rt.RuntimeContext(rt.KernelContext(rt.default_opset(18)), events_enabled=True)

        # Set inputs
        m_tp = TensorProto()
        m_tp.name = "M"
        m_tp.data_type = int(TensorProto.INT64)
        m_tp.raw_data = struct.pack("<q", 2)
        ctx.set("M", rt.tensor_from_proto(m_tp))

        cond_tp = TensorProto()
        cond_tp.name = "cond"
        cond_tp.data_type = int(TensorProto.BOOL)
        cond_tp.raw_data = struct.pack("B", 1)
        ctx.set("cond", rt.tensor_from_proto(cond_tp))

        ctx.set("s_init", _make_float_tensor("s_init", [0.0]))
        ctx.clear_events()
        _run_model(model, ctx)

        body_events = [ev for ev in ctx.events() if ev.subgraph_attr_name == "body"]
        self.assertGreater(len(body_events), 0, "No event with subgraph_attr_name='body' found")
        # The Loop node is node 0 in the main graph.
        self.assertTrue(all(ev.subgraph_node_index == 0 for ev in body_events))

    def test_if_then_branch_events_carry_attr_name(self):
        """Events from the then_branch of an If must carry subgraph_attr_name='then_branch'."""
        model = self._build_if_model()
        ctx = rt.RuntimeContext(rt.KernelContext(rt.default_opset(18)), events_enabled=True)

        cond_tp = TensorProto()
        cond_tp.name = "cond"
        cond_tp.data_type = int(TensorProto.BOOL)
        import struct

        cond_tp.raw_data = struct.pack("B", 1)  # true
        ctx.set("cond", rt.tensor_from_proto(cond_tp))
        ctx.clear_events()
        _run_model(model, ctx)

        then_events = [ev for ev in ctx.events() if ev.subgraph_attr_name == "then_branch"]
        self.assertGreater(
            len(then_events), 0, "No event with subgraph_attr_name='then_branch' found"
        )
        self.assertTrue(all(ev.subgraph_node_index == 0 for ev in then_events))

    def test_if_else_branch_events_carry_attr_name(self):
        """Events from the else_branch of an If must carry subgraph_attr_name='else_branch'."""
        model = self._build_if_model()
        ctx = rt.RuntimeContext(rt.KernelContext(rt.default_opset(18)), events_enabled=True)

        cond_tp = TensorProto()
        cond_tp.name = "cond"
        cond_tp.data_type = int(TensorProto.BOOL)
        import struct

        cond_tp.raw_data = struct.pack("B", 0)  # false
        ctx.set("cond", rt.tensor_from_proto(cond_tp))
        ctx.clear_events()
        _run_model(model, ctx)

        else_events = [ev for ev in ctx.events() if ev.subgraph_attr_name == "else_branch"]
        self.assertGreater(
            len(else_events), 0, "No event with subgraph_attr_name='else_branch' found"
        )
        self.assertTrue(all(ev.subgraph_node_index == 0 for ev in else_events))

    def test_subgraph_fields_exposed_in_as_dict(self):
        """The ``subgraph_node_index`` and ``subgraph_attr_name`` fields must be in as_dict()."""
        model = parser.parse_model(_MODEL_SRC)
        ctx = rt.RuntimeContext(rt.KernelContext(rt.default_opset(18)), events_enabled=True)
        ctx.set("x", _make_float_tensor("x", [-1.0, 2.0, -3.5]))
        ctx.set("z", _make_float_tensor("z", [10.0, 20.0, 30.0]))
        ctx.clear_events()
        _run_model(model, ctx)

        for ev in ctx.events():
            d = ev.as_dict()
            self.assertIn("subgraph_node_index", d)
            self.assertIn("subgraph_attr_name", d)
            self.assertEqual(d["subgraph_node_index"], -1)
            self.assertEqual(d["subgraph_attr_name"], "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
