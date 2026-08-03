# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for :class:`onnx_light.reference.ReferenceEvaluator`.

The evaluator is a thin Python layer on top of the C++ kernels Python
API (``onnx_light.onnx_py._onnxpykernels.runtime``). These tests cover
the numpy ↔ runtime ``Tensor`` round-trip, the supported input shapes
(``ModelProto`` / ``GraphProto`` / ``FunctionProto`` / bytes / path)
and the ``run`` calling convention.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np


from onnx_light.ext_test_case import ExtTestCase, import_or_skip
import onnx_light.onnx as onnxl
from onnx_light.onnx import numpy_helper
from onnx_light.onnx_lib import parser

# The reference runtime is only available in the full build; skip this module on
# a reduced build (ONNX_LIGHT_BUILD_KERNELS=OFF).
ReferenceEvaluator = import_or_skip("onnx_light.onnx.reference", "ReferenceEvaluator")

_ABS_ADD_MODEL_SRC = (
    '<ir_version: 10, opset_import: ["" : 18]>\n'
    "agraph (float[3] x, float[3] z) => (float[3] y) {\n"
    "  t = Abs(x)\n"
    "  y = Add(t, z)\n"
    "}\n"
)


_INITIALIZER_MODEL_SRC = (
    '<ir_version: 10, opset_import: ["" : 18]>\n'
    "agraph (float[3] x) => (float[3] y) <float[3] b = {1.0, 2.0, 3.0}>\n"
    "{ y = Add(x, b) }\n"
)


class TestReferenceEvaluator(ExtTestCase):
    def test_metadata(self):
        model = parser.parse_model(_ABS_ADD_MODEL_SRC)
        sess = ReferenceEvaluator(model)
        self.assertEqual(sess.input_names, ["x", "z"])
        self.assertEqual(sess.output_names, ["y"])
        self.assertEqual(sess.opsets, {"": 18})

    def test_run_default_output_names(self):
        model = parser.parse_model(_ABS_ADD_MODEL_SRC)
        sess = ReferenceEvaluator(model)
        x = np.array([-1.0, 2.0, -3.5], dtype=np.float32)
        z = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        (y,) = sess.run(None, {"x": x, "z": z})
        np.testing.assert_array_equal(y, np.array([11.0, 22.0, 33.5], dtype=np.float32))

    def test_run_explicit_output_names(self):
        model = parser.parse_model(_ABS_ADD_MODEL_SRC)
        sess = ReferenceEvaluator(model)
        x = np.array([-4.0, 5.0, -6.0], dtype=np.float32)
        z = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        (y,) = sess.run(["y"], {"x": x, "z": z})
        np.testing.assert_array_equal(y, np.array([5.0, 6.0, 7.0], dtype=np.float32))

    def test_run_intermediate_output(self):
        # ``t`` is an internal node output (not declared in graph.output) but
        # is still observable through the runtime context.
        model = parser.parse_model(_ABS_ADD_MODEL_SRC)
        sess = ReferenceEvaluator(model)
        x = np.array([-1.0, 1.0, -2.0], dtype=np.float32)
        z = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        (t,) = sess.run(["t"], {"x": x, "z": z})
        np.testing.assert_array_equal(t, np.array([1.0, 1.0, 2.0], dtype=np.float32))

    def test_initializer_supplies_input(self):
        model = parser.parse_model(_INITIALIZER_MODEL_SRC)
        sess = ReferenceEvaluator(model)
        # ``b`` is provided as an initializer, so it must not appear in
        # ``input_names`` -- only ``x`` is required from the caller.
        self.assertEqual(sess.input_names, ["x"])
        (y,) = sess.run(None, {"x": np.array([10.0, 20.0, 30.0], dtype=np.float32)})
        np.testing.assert_array_equal(y, np.array([11.0, 22.0, 33.0], dtype=np.float32))

    def test_construct_from_bytes(self):
        model = parser.parse_model(_ABS_ADD_MODEL_SRC)
        sess = ReferenceEvaluator(model.SerializeToString())
        self.assertEqual(sess.input_names, ["x", "z"])
        (y,) = sess.run(
            None,
            {
                "x": np.array([0.0, 0.0, 0.0], dtype=np.float32),
                "z": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            },
        )
        np.testing.assert_array_equal(y, np.array([1.0, 2.0, 3.0], dtype=np.float32))

    def test_construct_from_path(self):
        model = parser.parse_model(_ABS_ADD_MODEL_SRC)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "model.onnx")
            with open(path, "wb") as f:
                f.write(model.SerializeToString())
            sess = ReferenceEvaluator(path)
            (y,) = sess.run(
                None,
                {
                    "x": np.array([-1.0, -1.0, -1.0], dtype=np.float32),
                    "z": np.array([0.0, 0.0, 0.0], dtype=np.float32),
                },
            )
            np.testing.assert_array_equal(y, np.ones(3, dtype=np.float32))

    def test_construct_from_graph(self):
        model = parser.parse_model(_INITIALIZER_MODEL_SRC)
        sess = ReferenceEvaluator(model.graph)
        # Graph-only construction has no opset_import metadata.
        self.assertEqual(sess.opsets, {})
        (y,) = sess.run(None, {"x": np.array([0.0, 0.0, 0.0], dtype=np.float32)})
        np.testing.assert_array_equal(y, np.array([1.0, 2.0, 3.0], dtype=np.float32))

    def test_events_empty_before_run(self):
        model = parser.parse_model(_ABS_ADD_MODEL_SRC)
        sess = ReferenceEvaluator(model)
        self.assertEqual(sess.events(), [])

    def test_verbose_must_be_non_negative_int(self):
        model = parser.parse_model(_ABS_ADD_MODEL_SRC)
        with self.assertRaises(TypeError):
            ReferenceEvaluator(model, verbose=1.5)
        with self.assertRaises(ValueError):
            ReferenceEvaluator(model, verbose=-1)

    def test_events_after_run(self):
        # After run, events() should return a non-empty list whose entries
        # have an as_dict() method with the expected keys.
        model = parser.parse_model(_ABS_ADD_MODEL_SRC)
        sess = ReferenceEvaluator(model, events_enabled=True)
        x = np.array([-1.0, 2.0, -3.5], dtype=np.float32)
        z = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        sess.run(None, {"x": x, "z": z})
        events = sess.events()
        self.assertGreater(len(events), 0)
        for ev in events:
            d = ev.as_dict()
            for key in ("action", "kind", "name", "data_type", "shape"):
                self.assertIn(key, d)

    def test_verbose_prints_node_progress(self):
        code = f"""
import numpy as np
from onnx_light.onnx_lib import parser
from onnx_light.onnx.reference import ReferenceEvaluator

model = parser.parse_model({ _ABS_ADD_MODEL_SRC!r })
sess = ReferenceEvaluator(model, verbose=1)
sess.run(
    None,
    {{
        "x": np.array([-1.0, 2.0, -3.5], dtype=np.float32),
        "z": np.array([10.0, 20.0, 30.0], dtype=np.float32),
    }},
)
"""
        proc = subprocess.run(
            [sys.executable, "-c", code], check=True, capture_output=True, text=True
        )
        out = proc.stdout
        self.assertIn("[ReferenceEvaluator] #0 Abs(x) -> (t)", out)
        self.assertIn("[ReferenceEvaluator] #1 Add(t, z) -> (y)", out)

    def test_events_contain_intermediate(self):
        # The intermediate tensor ``t = Abs(x)`` must appear in the event log.
        model = parser.parse_model(_ABS_ADD_MODEL_SRC)
        sess = ReferenceEvaluator(model, events_enabled=True)
        sess.run(None, {"x": np.zeros(3, dtype=np.float32), "z": np.zeros(3, dtype=np.float32)})
        names = [ev.as_dict()["name"] for ev in sess.events()]
        self.assertIn("t", names)

    def test_events_reset_on_new_run(self):
        # Each run() overwrites the stored context, so events() always reflects
        # the most recent execution.
        model = parser.parse_model(_ABS_ADD_MODEL_SRC)
        sess = ReferenceEvaluator(model)
        x = np.zeros(3, dtype=np.float32)
        z = np.zeros(3, dtype=np.float32)
        sess.run(None, {"x": x, "z": z})
        first_count = len(sess.events())
        sess.run(None, {"x": x, "z": z})
        self.assertEqual(len(sess.events()), first_count)

        model = parser.parse_model(_ABS_ADD_MODEL_SRC)
        sess = ReferenceEvaluator(model)
        with self.assertRaises(ValueError) as ctx:
            sess.run(None, {"x": np.zeros(3, dtype=np.float32)})
        self.assertIn("Missing input", str(ctx.exception))
        self.assertIn("z", str(ctx.exception))

    def test_run_reuses_context_across_calls(self):
        # The RuntimeContext is built once in the constructor and reused for
        # every run(). Repeated runs with different inputs must therefore
        # produce independent, correct results (the per-run state is reset).
        model = parser.parse_model(_ABS_ADD_MODEL_SRC)
        sess = ReferenceEvaluator(model)
        x1 = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
        z1 = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        (y1,) = sess.run(None, {"x": x1, "z": z1})
        self.assertEqualArray(y1, np.abs(x1) + z1)

        x2 = np.array([4.0, -5.0, 6.0], dtype=np.float32)
        z2 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
        (y2,) = sess.run(None, {"x": x2, "z": z2})
        self.assertEqualArray(y2, np.abs(x2) + z2)

        # Running the first feed again still yields the original result.
        (y1b,) = sess.run(None, {"x": x1, "z": z1})
        self.assertEqualArray(y1b, np.abs(x1) + z1)

    def test_release_intermediates_removes_unused_and_logs_event(self):
        # With release_intermediates=True, the intermediate "t" is removed
        # from the runtime context as soon as the last node that references
        # it (Add) finishes. The graph output "y" is preserved.
        model = parser.parse_model(_ABS_ADD_MODEL_SRC)
        sess = ReferenceEvaluator(model, events_enabled=True, release_intermediates=True)
        x = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
        z = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        (y,) = sess.run(["y"], {"x": x, "z": z})
        np.testing.assert_array_equal(y, np.array([11.0, 22.0, 33.0], dtype=np.float32))

        actions = [ev.as_dict() for ev in sess.events()]
        removed_names = [d["name"] for d in actions if d["action"] == "remove"]
        self.assertIn("t", removed_names)
        # "y" (graph output) must NOT have been removed.
        self.assertNotIn("y", removed_names)

    def test_release_intermediates_disabled_keeps_intermediates(self):
        # With release_intermediates=False, every intermediate stays observable after the run.
        model = parser.parse_model(_ABS_ADD_MODEL_SRC)
        sess = ReferenceEvaluator(model, events_enabled=True, release_intermediates=False)
        sess.run(None, {"x": np.zeros(3, dtype=np.float32), "z": np.zeros(3, dtype=np.float32)})
        actions = [ev.as_dict() for ev in sess.events()]
        removed_names = [d["name"] for d in actions if d["action"] == "remove"]
        self.assertEqual(removed_names, [])

    def test_unknown_output_raises(self):
        model = parser.parse_model(_ABS_ADD_MODEL_SRC)
        sess = ReferenceEvaluator(model)
        with self.assertRaises(RuntimeError):
            sess.run(
                ["nonexistent"],
                {"x": np.zeros(3, dtype=np.float32), "z": np.zeros(3, dtype=np.float32)},
            )

    def test_feed_inputs_must_be_dict(self):
        model = parser.parse_model(_ABS_ADD_MODEL_SRC)
        sess = ReferenceEvaluator(model)
        with self.assertRaises(TypeError):
            sess.run(None, [np.zeros(3, dtype=np.float32)])  # type: ignore[arg-type]

    def test_unsupported_proto_type(self):
        with self.assertRaises(TypeError):
            ReferenceEvaluator(42)  # type: ignore[arg-type]

    def test_multi_dtype_round_trip(self):
        # Exercise the numpy <-> Tensor conversion for several non-float32
        # dtypes by running ``Add`` on each.
        src = (
            '<ir_version: 10, opset_import: ["" : 18]>\n'
            "ag (int16[2,3] a0, int16[2,3] a1, uint8[2] b0, uint8[2] b1) => "
            "(int16[2,3] y0, uint8[2] y1) {\n"
            "  y0 = Add(a0, a1)\n"
            "  y1 = Add(b0, b1)\n"
            "}\n"
        )
        model = parser.parse_model(src)
        sess = ReferenceEvaluator(model)
        a0 = np.array([[1, 2, 3], [-1, -2, -3]], dtype=np.int16)
        a1 = np.array([[10, 20, 30], [-10, -20, -30]], dtype=np.int16)
        b0 = np.array([1, 2], dtype=np.uint8)
        b1 = np.array([3, 4], dtype=np.uint8)
        y0, y1 = sess.run(None, {"a0": a0, "a1": a1, "b0": b0, "b1": b1})
        np.testing.assert_array_equal(y0, a0 + a1)
        np.testing.assert_array_equal(y1, b0 + b1)
        self.assertEqual(y0.dtype, np.int16)
        self.assertEqual(y1.dtype, np.uint8)

    def test_dict_vectorizer_int64_float(self):
        # ``ai.onnx.ml::DictVectorizer`` consumes a ``map(int64, float)``
        # graph input. ReferenceEvaluator exposes the original input name
        # in ``input_names`` and accepts a Python ``dict`` under that name.
        model = onnxl.ModelProto()
        model.ir_version = 10
        op = model.opset_import.add()
        op.domain = ""
        op.version = 13
        op_ml = model.opset_import.add()
        op_ml.domain = "ai.onnx.ml"
        op_ml.version = 1
        graph = model.graph
        graph.name = "dv"
        x = graph.input.add()
        x.name = "x"
        mt = x.type.map_type
        mt.key_type = int(onnxl.TensorProto.INT64)
        mt.value_type.tensor_type.elem_type = int(onnxl.TensorProto.FLOAT)
        y = graph.output.add()
        y.name = "y"
        y.type.tensor_type.elem_type = int(onnxl.TensorProto.FLOAT)
        node = graph.node.add()
        node.op_type = "DictVectorizer"
        node.domain = "ai.onnx.ml"
        node.input.append("x")
        node.output.append("y")
        attr = node.attribute.add()
        attr.name = "int64_vocabulary"
        attr.type = onnxl.AttributeProto.INTS
        attr.ints.extend([10, 20, 30])

        sess = ReferenceEvaluator(model)
        # Map-typed inputs are listed under their original graph-input name.
        self.assertEqual(sess.input_names, ["x"])

        # Feed a Python dict under the original input name.
        (out_dict,) = sess.run(None, {"x": {10: 1.5, 30: 2.5}})
        np.testing.assert_array_equal(out_dict, np.array([1.5, 0.0, 2.5], dtype=np.float32))

        # A size-1 numpy object array wrapping the dict is unwrapped too
        # (1-D shape-(1,) variant; see also the 0-D variant below).
        (out_obj,) = sess.run(None, {"x": np.array([{10: 1.5, 30: 2.5}], dtype=object)})
        np.testing.assert_array_equal(out_obj, np.array([1.5, 0.0, 2.5], dtype=np.float32))

        # Regression test for issue #2577: a 0-D numpy object array produced
        # by ``np.asarray(some_dict, dtype=object)`` must also be unwrapped and
        # accepted.
        wrapped_0d = np.asarray({10: 1.5, 30: 2.5}, dtype=object)
        (out_0d,) = sess.run(None, {"x": wrapped_0d})
        np.testing.assert_array_equal(out_0d, np.array([1.5, 0.0, 2.5], dtype=np.float32))

    def test_cast_map_int64_float_dense_dict_feed(self):
        # Regression test for issue #2576: a ``map(int64, float)`` input fed to
        # ``ai.onnx.ml::CastMap`` as a Python ``dict`` under its original name
        # ``x`` must be accepted.
        model = onnxl.ModelProto()
        model.ir_version = 10
        op = model.opset_import.add()
        op.domain = ""
        op.version = 13
        op_ml = model.opset_import.add()
        op_ml.domain = "ai.onnx.ml"
        op_ml.version = 1
        graph = model.graph
        graph.name = "cm"
        x = graph.input.add()
        x.name = "x"
        mt = x.type.map_type
        mt.key_type = int(onnxl.TensorProto.INT64)
        mt.value_type.tensor_type.elem_type = int(onnxl.TensorProto.FLOAT)
        y = graph.output.add()
        y.name = "y"
        y.type.tensor_type.elem_type = int(onnxl.TensorProto.FLOAT)
        node = graph.node.add()
        node.op_type = "CastMap"
        node.domain = "ai.onnx.ml"
        node.input.append("x")
        node.output.append("y")
        cast_to = node.attribute.add()
        cast_to.name = "cast_to"
        cast_to.type = onnxl.AttributeProto.STRING
        cast_to.s = b"TO_FLOAT"
        map_form = node.attribute.add()
        map_form.name = "map_form"
        map_form.type = onnxl.AttributeProto.STRING
        map_form.s = b"DENSE"

        sess = ReferenceEvaluator(model)
        self.assertEqual(sess.input_names, ["x"])
        # Keys are deliberately unsorted; CastMap DENSE sorts them ascending.
        (out,) = sess.run(None, {"x": {2: 2.5, 0: 0.5, 1: 1.5}})
        np.testing.assert_array_equal(out, np.array([0.5, 1.5, 2.5], dtype=np.float32))

    def test_lstm_layout1_matches_layout0(self):
        # Regression test for ``test_cc_lstm_batchwise``: the LSTM kernel
        # itself only implements ``layout=0`` so the dispatch table
        # permutes X / initial_h / initial_c on the way in and Y / Y_h on
        # the way out when ``layout=1`` is requested. Running both
        # layouts on the same logical inputs must produce the same
        # outputs (modulo the axis permutation).
        from onnx_light.onnx import TensorProto, helper

        batch, seq, inp, hid = 2, 3, 4, 5
        rng = np.random.default_rng(0)
        weights = rng.standard_normal((1, 4 * hid, inp), dtype=np.float32) * 0.1
        recur = rng.standard_normal((1, 4 * hid, hid), dtype=np.float32) * 0.1
        x_layout0 = rng.standard_normal((seq, batch, inp), dtype=np.float32)
        x_layout1 = np.transpose(x_layout0, (1, 0, 2)).copy()

        def build_model(layout: int, x_shape: list[int]):
            node = helper.make_node(
                "LSTM", ["X", "W", "R"], ["Y", "Y_h"], hidden_size=hid, layout=layout
            )
            graph = helper.make_graph(
                [node],
                "g",
                [
                    helper.make_tensor_value_info("X", TensorProto.FLOAT, x_shape),
                    helper.make_tensor_value_info("W", TensorProto.FLOAT, [1, 4 * hid, inp]),
                    helper.make_tensor_value_info("R", TensorProto.FLOAT, [1, 4 * hid, hid]),
                ],
                [
                    helper.make_tensor_value_info("Y", TensorProto.FLOAT, None),
                    helper.make_tensor_value_info("Y_h", TensorProto.FLOAT, None),
                ],
            )
            return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 22)])

        sess0 = ReferenceEvaluator(build_model(0, [seq, batch, inp]))
        sess1 = ReferenceEvaluator(build_model(1, [batch, seq, inp]))
        y0, y_h0 = sess0.run(None, {"X": x_layout0, "W": weights, "R": recur})
        y1, y_h1 = sess1.run(None, {"X": x_layout1, "W": weights, "R": recur})

        self.assertEqual(y0.shape, (seq, 1, batch, hid))
        self.assertEqual(y1.shape, (batch, seq, 1, hid))
        self.assertEqual(y_h0.shape, (1, batch, hid))
        self.assertEqual(y_h1.shape, (batch, 1, hid))
        np.testing.assert_allclose(np.transpose(y0, (2, 0, 1, 3)), y1, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.transpose(y_h0, (1, 0, 2)), y_h1, rtol=1e-5, atol=1e-6)

    def test_gru_layout1_matches_layout0(self):
        # Regression test for ``test_cc_gru_batchwise``: the GRU kernel
        # supports both ``layout=0`` and ``layout=1`` (the latter permutes
        # X / initial_h on the way in and Y / Y_h on the way out).
        # Running both layouts on the same logical inputs must produce
        # the same outputs (modulo the axis permutation).
        from onnx_light.onnx import TensorProto, helper

        batch, seq, inp, hid = 2, 3, 4, 5
        rng = np.random.default_rng(0)
        weights = rng.standard_normal((1, 3 * hid, inp), dtype=np.float32) * 0.1
        recur = rng.standard_normal((1, 3 * hid, hid), dtype=np.float32) * 0.1
        x_layout0 = rng.standard_normal((seq, batch, inp), dtype=np.float32)
        x_layout1 = np.transpose(x_layout0, (1, 0, 2)).copy()

        def build_model(layout: int, x_shape: list[int]):
            node = helper.make_node(
                "GRU", ["X", "W", "R"], ["Y", "Y_h"], hidden_size=hid, layout=layout
            )
            graph = helper.make_graph(
                [node],
                "g",
                [
                    helper.make_tensor_value_info("X", TensorProto.FLOAT, x_shape),
                    helper.make_tensor_value_info("W", TensorProto.FLOAT, [1, 3 * hid, inp]),
                    helper.make_tensor_value_info("R", TensorProto.FLOAT, [1, 3 * hid, hid]),
                ],
                [
                    helper.make_tensor_value_info("Y", TensorProto.FLOAT, None),
                    helper.make_tensor_value_info("Y_h", TensorProto.FLOAT, None),
                ],
            )
            return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 22)])

        sess0 = ReferenceEvaluator(build_model(0, [seq, batch, inp]))
        sess1 = ReferenceEvaluator(build_model(1, [batch, seq, inp]))
        y0, y_h0 = sess0.run(None, {"X": x_layout0, "W": weights, "R": recur})
        y1, y_h1 = sess1.run(None, {"X": x_layout1, "W": weights, "R": recur})

        self.assertEqual(y0.shape, (seq, 1, batch, hid))
        self.assertEqual(y1.shape, (batch, seq, 1, hid))
        self.assertEqual(y_h0.shape, (1, batch, hid))
        self.assertEqual(y_h1.shape, (batch, 1, hid))
        np.testing.assert_allclose(np.transpose(y0, (2, 0, 1, 3)), y1, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.transpose(y_h0, (1, 0, 2)), y_h1, rtol=1e-5, atol=1e-6)

    def test_loop_zero_trip_count(self):
        # Regression test for ``test_cc_loop_zero_trip_count``: when ``M = 0``
        # the loop runs zero iterations and the scan output is empty along
        # its leading axis. The kernel has no per-iteration template to seed
        # the dtype/shape from, so ``RunLoopNode`` must patch the empty scan
        # output's dtype/trailing shape from the body's declared output
        # value-info; otherwise the downstream numpy conversion raises
        # ``"The element type in the input tensor is UNDEFINED."``.
        from onnx_light.onnx_lib.backend.test.case import get_test_case

        tc = get_test_case("test_cc_loop_zero_trip_count")
        self.assertIsNotNone(tc)
        inputs, outputs = tc.data_sets[0]
        sess = ReferenceEvaluator(tc.model)
        got = sess.run(None, dict(zip(sess.input_names, inputs)))
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0].dtype, outputs[0].dtype)
        self.assertEqual(got[0].shape, outputs[0].shape)
        np.testing.assert_array_equal(got[0], outputs[0])

    def test_flexattention_relative_positional(self):
        # Regression test for ``test_cc_flexattention_relative_positional``:
        # the ``score_mod`` subgraph computes ``q_idx - k_idx`` on INT64
        # index tensors produced by ``Range``/``Reshape``. Earlier
        # versions of :cpp:class:`kernel::Sub` rejected INT64 inputs with
        # ``kernel::Sub only supports FLOAT, INT8, INT16, UINT8, UINT16,
        # UINT32 and UINT64 inputs.``, which broke the FlexAttention path
        # whenever a ``score_mod`` subgraph subtracted query/key indices.
        from onnx_light.onnx_lib.backend.test.case import get_test_case

        tc = get_test_case("test_cc_flexattention_relative_positional")
        self.assertIsNotNone(tc)
        inputs, outputs = tc.data_sets[0]
        sess = ReferenceEvaluator(tc.model)
        got = sess.run(None, dict(zip(sess.input_names, inputs)))
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0].dtype, outputs[0].dtype)
        self.assertEqual(got[0].shape, outputs[0].shape)
        np.testing.assert_allclose(got[0], outputs[0], rtol=tc.rtol, atol=tc.atol)

    def test_sequence_erase_pos1_returns_list(self):
        # Regression test for test_cc_sequence_erase_pos1: the graph output
        # is a SequenceType, so ReferenceEvaluator.run() must return a list
        # of numpy arrays rather than raise "Output was not produced".
        from onnx_light.onnx_lib.backend.test.case import get_test_case

        tc = get_test_case("test_cc_sequence_erase_pos1")
        self.assertIsNotNone(tc)
        inputs, outputs = tc.data_sets[0]
        sess = ReferenceEvaluator(tc.model)
        got = sess.run(None, dict(zip(sess.input_names, inputs)))
        self.assertEqual(len(got), 1)
        # Sequence output is a list of numpy arrays.
        self.assertIsInstance(got[0], list)
        # Erasing position 1 from [a, b, c] leaves [a, c], so 2 elements.
        self.assertEqual(len(got[0]), 2)
        # The expected value in the data set is the stacked tensor [a, c].
        expected_stacked = outputs[0]
        np.testing.assert_allclose(got[0][0], expected_stacked[0], rtol=tc.rtol, atol=tc.atol)
        np.testing.assert_allclose(got[0][1], expected_stacked[1], rtol=tc.rtol, atol=tc.atol)

    def test_sequence_erase_default_returns_list(self):
        # Regression test for test_cc_sequence_erase_default: the graph output
        # is a SequenceType, so ReferenceEvaluator.run() must return a list
        # of numpy arrays rather than raise "Output was not produced".
        from onnx_light.onnx_lib.backend.test.case import get_test_case

        tc = get_test_case("test_cc_sequence_erase_default")
        self.assertIsNotNone(tc)
        inputs, outputs = tc.data_sets[0]
        sess = ReferenceEvaluator(tc.model)
        got = sess.run(None, dict(zip(sess.input_names, inputs)))
        self.assertEqual(len(got), 1)
        # Sequence output is a list of numpy arrays.
        self.assertIsInstance(got[0], list)
        # Default erase removes the last element from [a, b, c], leaving [a, b].
        self.assertEqual(len(got[0]), 2)
        # The expected value in the data set is the stacked tensor [a, b].
        expected_stacked = outputs[0]
        np.testing.assert_allclose(got[0][0], expected_stacked[0], rtol=tc.rtol, atol=tc.atol)
        np.testing.assert_allclose(got[0][1], expected_stacked[1], rtol=tc.rtol, atol=tc.atol)

    def test_split_to_sequence_1_returns_list(self):
        # Regression test for test_cc_split_to_sequence_1: the graph output of
        # SplitToSequence is a SequenceType, so ReferenceEvaluator.run() must
        # return a list of numpy arrays rather than raise "Output was not
        # produced". The scalar 'split' input (value 2) splits the [3, 6] data
        # along axis=1 into three [3, 2] chunks.
        from onnx_light.onnx_lib.backend.test.case import get_test_case

        tc = get_test_case("test_cc_split_to_sequence_1")
        self.assertIsNotNone(tc)
        inputs, outputs = tc.data_sets[0]
        sess = ReferenceEvaluator(tc.model)
        got = sess.run(None, dict(zip(sess.input_names, inputs)))
        self.assertEqual(len(got), 1)
        # Sequence output is a list of numpy arrays.
        self.assertIsInstance(got[0], list)
        # Splitting [3, 6] by the scalar 2 along axis=1 yields three [3, 2] chunks.
        self.assertEqual(len(got[0]), 3)
        # The expected value in the data set is the sequence [c0, c1, c2].
        expected_chunks = outputs[0]
        self.assertEqual(len(expected_chunks), 3)
        for got_chunk, expected_chunk in zip(got[0], expected_chunks):
            self.assertEqual(got_chunk.shape, (3, 2))
            np.testing.assert_allclose(got_chunk, expected_chunk, rtol=tc.rtol, atol=tc.atol)

    def test_sequence_map_identity_returns_list(self):
        # Regression test for test_cc_sequence_map_identity_float: SequenceMap
        # applies an Identity body to each element of a single input sequence,
        # so ReferenceEvaluator.run() must execute the body subgraph once per
        # element and return the result as a list of numpy arrays.
        from onnx_light.onnx_lib.backend.test.case import get_test_case

        tc = get_test_case("test_cc_sequence_map_identity_float")
        self.assertIsNotNone(tc)
        inputs, outputs = tc.data_sets[0]
        sess = ReferenceEvaluator(tc.model)
        got = sess.run(None, dict(zip(sess.input_names, inputs)))
        self.assertEqual(len(got), 1)
        # Sequence output is a list of numpy arrays.
        self.assertIsInstance(got[0], list)
        # Identity over a 3-element sequence yields 3 elements unchanged.
        expected_stacked = outputs[0]
        self.assertEqual(len(got[0]), len(expected_stacked))
        for got_elem, expected_elem in zip(got[0], expected_stacked):
            np.testing.assert_allclose(got_elem, expected_elem, rtol=tc.rtol, atol=tc.atol)

    def test_sequence_map_with_sequence_graph_input(self):
        # Regression test for feeding a ``seq(tensor)`` graph input directly:
        # the SequenceMap node consumes a top-level sequence input ``x`` and the
        # ReferenceEvaluator.run() feed boundary must accept a list of numpy
        # arrays (one per element) via ``put_sequence`` rather than the
        # single-tensor ``set`` path.
        from onnx_light.onnx import helper

        body = helper.make_graph(
            [helper.make_node("Identity", ["elem"], ["out"])],
            "body",
            [helper.make_tensor_value_info("elem", onnxl.TensorProto.FLOAT, None)],
            [helper.make_tensor_value_info("out", onnxl.TensorProto.FLOAT, None)],
        )
        node = helper.make_node("SequenceMap", ["x"], ["y"], body=body)
        graph = helper.make_graph(
            [node],
            "g",
            [helper.make_tensor_sequence_value_info("x", onnxl.TensorProto.FLOAT, None)],
            [helper.make_tensor_sequence_value_info("y", onnxl.TensorProto.FLOAT, None)],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

        sess = ReferenceEvaluator(model)
        self.assertEqual(sess.input_names, ["x"])
        elements = [
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0, 5.0], dtype=np.float32),
        ]
        got = sess.run(None, {"x": elements})
        self.assertEqual(len(got), 1)
        self.assertIsInstance(got[0], list)
        self.assertEqual(len(got[0]), len(elements))
        for got_elem, expected_elem in zip(got[0], elements):
            np.testing.assert_allclose(got_elem, expected_elem)

    def test_sequence_input_must_be_list(self):
        # A ``seq(tensor)`` graph input fed as a bare array (instead of a list of
        # arrays) is a caller error and must raise a clear ``TypeError``.
        from onnx_light.onnx import helper

        body = helper.make_graph(
            [helper.make_node("Identity", ["elem"], ["out"])],
            "body",
            [helper.make_tensor_value_info("elem", onnxl.TensorProto.FLOAT, None)],
            [helper.make_tensor_value_info("out", onnxl.TensorProto.FLOAT, None)],
        )
        node = helper.make_node("SequenceMap", ["x"], ["y"], body=body)
        graph = helper.make_graph(
            [node],
            "g",
            [helper.make_tensor_sequence_value_info("x", onnxl.TensorProto.FLOAT, None)],
            [helper.make_tensor_sequence_value_info("y", onnxl.TensorProto.FLOAT, None)],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        sess = ReferenceEvaluator(model)
        with self.assertRaises(TypeError):
            sess.run(None, {"x": np.array([1.0, 2.0], dtype=np.float32)})

        # Regression test for test_cc_sequence_map_identity_2_sequences: the
        # SequenceMap node consumes two input sequences and produces two output
        # sequences (y0, y1) through a two-input/two-output Identity body, so
        # ReferenceEvaluator.run() must return one list of numpy arrays per
        # output sequence.
        from onnx_light.onnx_lib.backend.test.case import get_test_case

        tc = get_test_case("test_cc_sequence_map_identity_2_sequences")
        self.assertIsNotNone(tc)
        inputs, outputs = tc.data_sets[0]
        sess = ReferenceEvaluator(tc.model)
        got = sess.run(None, dict(zip(sess.input_names, inputs)))
        self.assertEqual(len(got), 2)
        for got_seq, expected_stacked in zip(got, outputs):
            self.assertIsInstance(got_seq, list)
            self.assertEqual(len(got_seq), len(expected_stacked))
            for got_elem, expected_elem in zip(got_seq, expected_stacked):
                np.testing.assert_allclose(got_elem, expected_elem, rtol=tc.rtol, atol=tc.atol)

    def test_image_decoder_decode_bmp_rgb(self):
        # Regression test for ``test_cc_image_decoder_decode_bmp_rgb``: the
        # ``ImageDecoder`` kernel must decode the BMP bytestream and return
        # the correct ``(32, 32, 3)`` uint8 tensor rather than the empty-matrix
        # fallback ``(0, 0, 3)`` returned by earlier versions of the kernel.
        from onnx_light.onnx_lib.backend.test.case import get_test_case

        tc = get_test_case("test_cc_image_decoder_decode_bmp_rgb")
        self.assertIsNotNone(tc)
        inputs, outputs = tc.data_sets[0]
        sess = ReferenceEvaluator(tc.model)
        got = sess.run(None, dict(zip(sess.input_names, inputs)))
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0].dtype, np.uint8)
        self.assertEqual(got[0].shape, (32, 32, 3))
        self.assertEqual(got[0].shape, outputs[0].shape)
        np.testing.assert_array_equal(got[0], outputs[0])

    def _check_image_decoder_jpeg(self, test_name, expected_shape, expected_channels):
        # Regression test for the JPEG variants of ``test_cc_image_decoder_*``:
        # earlier versions of the ``ImageDecoder`` kernel returned the
        # empty-matrix fallback ``(0, 0, C)`` for any non-BMP bytestream.
        # The baseline JFIF decoder must return the correct shape and pixel
        # values that closely match the upstream Pillow reference (small
        # integer rounding differences are tolerated since the reference is
        # produced by libjpeg-turbo).
        from onnx_light.onnx_lib.backend.test.case import get_test_case

        tc = get_test_case(test_name)
        self.assertIsNotNone(tc)
        inputs, outputs = tc.data_sets[0]
        sess = ReferenceEvaluator(tc.model)
        got = sess.run(None, dict(zip(sess.input_names, inputs)))
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0].dtype, np.uint8)
        self.assertEqual(got[0].shape, expected_shape)
        self.assertEqual(got[0].shape[-1], expected_channels)
        self.assertEqual(got[0].shape, outputs[0].shape)
        diff = np.abs(got[0].astype(np.int32) - outputs[0].astype(np.int32))
        # Allow at most a handful of unit-level differences caused by the
        # integer YCbCr→RGB conversion and chroma upsampling rounding.
        self.assertLessEqual(int(diff.max()), 2)

    def test_image_decoder_decode_jpeg_rgb(self):
        self._check_image_decoder_jpeg("test_cc_image_decoder_decode_jpeg_rgb", (32, 32, 3), 3)

    def test_image_decoder_decode_jpeg_bgr(self):
        self._check_image_decoder_jpeg("test_cc_image_decoder_decode_jpeg_bgr", (32, 32, 3), 3)

    def test_image_decoder_decode_jpeg_grayscale(self):
        self._check_image_decoder_jpeg(
            "test_cc_image_decoder_decode_jpeg_grayscale", (32, 32, 1), 1
        )

    def test_image_decoder_decode_png_rgb(self):
        # Regression test for ``test_cc_image_decoder_decode_png_rgb``: the
        # ``ImageDecoder`` kernel must inflate the PNG bytestream (deflate +
        # PNG filters) and return the correct ``(32, 32, 3)`` uint8 tensor
        # rather than the empty-matrix fallback ``(0, 0, 3)`` returned by
        # earlier versions of the kernel that lacked a PNG decoder.
        from onnx_light.onnx_lib.backend.test.case import get_test_case

        tc = get_test_case("test_cc_image_decoder_decode_png_rgb")
        self.assertIsNotNone(tc)
        inputs, outputs = tc.data_sets[0]
        sess = ReferenceEvaluator(tc.model)
        got = sess.run(None, dict(zip(sess.input_names, inputs)))
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0].dtype, np.uint8)
        self.assertEqual(got[0].shape, (32, 32, 3))
        self.assertEqual(got[0].shape, outputs[0].shape)
        np.testing.assert_array_equal(got[0], outputs[0])

    def test_image_decoder_decode_pnm_rgb(self):
        # Regression test for ``test_cc_image_decoder_decode_pnm_rgb``: the
        # ``ImageDecoder`` kernel must decode the binary PPM (``P6``)
        # bytestream and return the correct ``(32, 32, 3)`` uint8 tensor
        # rather than the empty-matrix fallback ``(0, 0, 3)`` returned by
        # earlier versions of the kernel that lacked a PNM decoder.
        from onnx_light.onnx_lib.backend.test.case import get_test_case

        tc = get_test_case("test_cc_image_decoder_decode_pnm_rgb")
        self.assertIsNotNone(tc)
        inputs, outputs = tc.data_sets[0]
        sess = ReferenceEvaluator(tc.model)
        got = sess.run(None, dict(zip(sess.input_names, inputs)))
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0].dtype, np.uint8)
        self.assertEqual(got[0].shape, (32, 32, 3))
        self.assertEqual(got[0].shape, outputs[0].shape)
        np.testing.assert_array_equal(got[0], outputs[0])

    def _check_resize_backend_case(self, test_name):
        # Regression test for the ``Resize`` ``align_corners`` downsample
        # variants: the ONNX reference uses ``output_width = scale *
        # input_width`` (a float) in the denominator of the coordinate
        # transformation, so sample positions land on non-integer indices
        # when ``scale * input_width`` is fractional. The C++ ``Resize``
        # kernel mirrors that convention; this test locks in bit-exact
        # agreement with the upstream backend reference outputs.
        from onnx_light.onnx_lib.backend.test.case import get_test_case

        tc = get_test_case(test_name)
        self.assertIsNotNone(tc)
        inputs, outputs = tc.data_sets[0]
        sess = ReferenceEvaluator(tc.model)
        got = sess.run(None, dict(zip(sess.input_names, inputs)))
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0].shape, outputs[0].shape)
        np.testing.assert_allclose(got[0], outputs[0], rtol=tc.rtol, atol=tc.atol)

    def test_resize_downsample_scales_linear_align_corners(self):
        self._check_resize_backend_case("test_resize_downsample_scales_linear_align_corners")

    def test_resize_downsample_scales_cubic_align_corners(self):
        self._check_resize_backend_case("test_resize_downsample_scales_cubic_align_corners")

    @staticmethod
    def _tiny_llm_inputs(bfloat16=None):
        # Builds a deterministic set of valid inputs for the ``tiny_llm``
        # decoder. The shape-inference test case ships no ``data_sets``, so
        # the inputs are generated here. ``input_ids`` indexes the 32-row
        # embedding table and the attention mask covers ``seq + past`` keys.
        rng = np.random.RandomState(0)
        batch, seq, past = 1, 3, 2
        float_dtype = np.float32 if bfloat16 is None else bfloat16
        return {
            "input_ids": rng.randint(0, 32, (batch, seq)).astype(np.int64),
            "attention_mask": np.ones((batch, seq + past), dtype=np.int64),
            "past_key": rng.rand(batch, 4, past, 4).astype(float_dtype),
            "past_value": rng.rand(batch, 4, past, 4).astype(float_dtype),
        }

    @staticmethod
    def _model_to_bfloat16(model, bfloat16):
        # Rewrites every FLOAT tensor in ``model`` to BFLOAT16: initializers,
        # ``Constant`` value tensors, ``Cast`` ``to`` attributes and the
        # element types of graph inputs/outputs/value_info. The model is
        # round-tripped through serialization first because the bound
        # ``ModelProto`` cannot be deep-copied.
        FLOAT = int(onnxl.TensorProto.FLOAT)
        BFLOAT16 = int(onnxl.TensorProto.BFLOAT16)
        converted = onnxl.ModelProto()
        converted.ParseFromString(model.SerializeToString())
        graph = converted.graph
        new_initializers = []
        for init in graph.initializer:
            if init.data_type == FLOAT:
                array = numpy_helper.to_array(init).astype(bfloat16)
                new_initializers.append(numpy_helper.from_array(array, init.name))
            else:
                new_initializers.append(init)
        del graph.initializer[:]
        graph.initializer.extend(new_initializers)
        for value_info in list(graph.input) + list(graph.output) + list(graph.value_info):
            tensor_type = value_info.type.tensor_type
            if tensor_type.elem_type == FLOAT:
                tensor_type.elem_type = BFLOAT16
        for node in graph.node:
            if node.op_type == "Cast":
                for attr in node.attribute:
                    if attr.name == "to" and attr.i == FLOAT:
                        attr.i = BFLOAT16
            elif node.op_type == "Constant":
                for attr in node.attribute:
                    if attr.name == "value" and attr.t.data_type == FLOAT:
                        array = numpy_helper.to_array(attr.t).astype(bfloat16)
                        attr.t.CopyFrom(numpy_helper.from_array(array))
        return converted

    def test_tiny_llm_float32(self):
        # Runs the ``tiny_llm`` Llama-style decoder end to end in float32 and
        # locks in the output ranks/dtypes (logits and the updated KV cache).
        from onnx_light.onnx_lib.backend.test.case import get_test_case

        tc = get_test_case("test_cc_shape_inference_tiny_llm")
        self.assertIsNotNone(tc)
        sess = ReferenceEvaluator(tc.model)
        logits, present_key, present_value = sess.run(None, self._tiny_llm_inputs())
        self.assertEqual(logits.dtype, np.float32)
        self.assertEqual(logits.shape, (1, 3, 32))
        self.assertEqual(present_key.shape, (1, 4, 5, 4))
        self.assertEqual(present_value.shape, (1, 4, 5, 4))

    def test_tiny_llm_reuses_runtime_session(self):
        # The evaluator caches and reuses a single RuntimeSession across
        # run() calls (so the per-node kernel resolution done on the session's
        # first Run is not repeated on every call), and repeated runs on the
        # same inputs return bit-identical results.
        from onnx_light.onnx_lib.backend.test.case import get_test_case

        tc = get_test_case("test_cc_shape_inference_tiny_llm")
        self.assertIsNotNone(tc)
        sess = ReferenceEvaluator(tc.model)
        inputs = self._tiny_llm_inputs()

        # run() returns zero-copy views over the runtime's tensor buffers,
        # which the allocator reuses on the next run(); copies the first run's
        # outputs so they survive the second run() before comparing.
        first = [np.array(out, copy=True) for out in sess.run(None, inputs)]
        # A single session is cached after the first run.
        self.assertEqual(len(sess._sessions), 1)
        cached = next(iter(sess._sessions.values()))

        second = sess.run(None, inputs)
        # The same (plan, session) entry is reused, not rebuilt.
        self.assertEqual(len(sess._sessions), 1)
        self.assertIs(next(iter(sess._sessions.values())), cached)
        for expected, actual in zip(first, second):
            np.testing.assert_array_equal(expected, actual)

        # Companion to ``test_tiny_llm_float32``: the same decoder runs in
        # BFLOAT16 (exercising the half-precision promote/demote paths of
        # RMSNormalization, Attention and the elementwise kernels) and the
        # de-quantized outputs stay close to the float32 reference.
        from onnx_light.onnx_lib.backend.test.case import get_test_case

        bfloat16 = import_or_skip("ml_dtypes", "bfloat16")
        tc = get_test_case("test_cc_shape_inference_tiny_llm")
        self.assertIsNotNone(tc)

        ref = ReferenceEvaluator(tc.model).run(None, self._tiny_llm_inputs())

        model_bf16 = self._model_to_bfloat16(tc.model, bfloat16)
        sess = ReferenceEvaluator(model_bf16)
        got = sess.run(None, self._tiny_llm_inputs(bfloat16))
        self.assertEqual(len(got), 3)
        for tensor in got:
            self.assertEqual(tensor.dtype, bfloat16)
        self.assertEqual(got[0].shape, (1, 3, 32))
        self.assertEqual(got[1].shape, (1, 4, 5, 4))
        self.assertEqual(got[2].shape, (1, 4, 5, 4))
        for expected, actual in zip(ref, got):
            np.testing.assert_allclose(expected, actual.astype(np.float32), rtol=0.05, atol=0.05)

    def test_resize_downsample_scales_linear_antialias(self):
        self._check_resize_backend_case("test_resize_downsample_scales_linear_antialias")

    def test_resize_downsample_scales_cubic_antialias(self):
        self._check_resize_backend_case("test_resize_downsample_scales_cubic_antialias")

    def test_resize_downsample_sizes_linear_antialias(self):
        self._check_resize_backend_case("test_resize_downsample_sizes_linear_antialias")

    def test_resize_downsample_sizes_cubic_antialias(self):
        self._check_resize_backend_case("test_resize_downsample_sizes_cubic_antialias")


class TestReferenceEvaluatorTensorConversion(ExtTestCase):
    """Tests for the runtime ``Tensor`` -> numpy conversion helper, which
    imports standard dtypes through the DLPack exchange protocol."""

    @classmethod
    def setUpClass(cls):
        from onnx_light._reference import _evaluator

        cls._evaluator = _evaluator
        cls._rt = _evaluator._runtime

    def _make_tensor(self, dtype_enum, array):
        tp = onnxl.TensorProto()
        tp.name = "x"
        tp.data_type = int(dtype_enum)
        tp.dims.extend(array.shape)
        tp.raw_data = array.tobytes()
        return self._rt.tensor_from_proto(tp)

    def test_dlpack_dtypes_exclude_subbyte_and_string(self):
        # Sub-byte packed types, STRING, and bfloat16/float8 must not be routed
        # through ``numpy.from_dlpack`` (no stock NumPy dtype).
        dlpack = self._evaluator._DLPACK_DTYPES
        self.assertIn(int(onnxl.TensorProto.FLOAT), dlpack)
        self.assertIn(int(onnxl.TensorProto.BOOL), dlpack)
        self.assertNotIn(int(onnxl.TensorProto.STRING), dlpack)
        self.assertNotIn(int(onnxl.TensorProto.BFLOAT16), dlpack)

    def test_dlpack_conversion_is_zero_copy(self):
        # Standard dtypes take the DLPack path: the resulting array shares the
        # tensor's buffer (``base`` is set) and matches shape/dtype/values.
        expected = np.arange(6, dtype=np.float32).reshape(2, 3)
        out = self._evaluator._cpp_tensor_to_numpy(
            self._make_tensor(onnxl.TensorProto.FLOAT, expected)
        )
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.shape, (2, 3))
        self.assertIsNotNone(out.base)
        np.testing.assert_array_equal(out, expected)

    def test_dlpack_conversion_various_dtypes(self):
        cases = [
            (onnxl.TensorProto.DOUBLE, np.array([1.5, 2.5], dtype=np.float64)),
            (onnxl.TensorProto.INT64, np.array([[1, 2], [3, 4]], dtype=np.int64)),
            (onnxl.TensorProto.UINT8, np.array([1, 2, 3], dtype=np.uint8)),
            (onnxl.TensorProto.BOOL, np.array([True, False, True])),
        ]
        for dtype_enum, array in cases:
            with self.subTest(dtype=dtype_enum):
                out = self._evaluator._cpp_tensor_to_numpy(self._make_tensor(dtype_enum, array))
                self.assertEqual(out.dtype, array.dtype)
                np.testing.assert_array_equal(out, array)

    def test_numpy_to_cpp_tensor_calls_ascontiguousarray_only_when_needed(self):
        contiguous = np.arange(6, dtype=np.float32).reshape(2, 3)
        non_contiguous = contiguous.T

        with (
            unittest.mock.patch.object(
                self._evaluator.np,
                "ascontiguousarray",
                wraps=self._evaluator.np.ascontiguousarray,
            ) as mocked_contiguous,
            unittest.mock.patch.object(
                self._evaluator._runtime,
                "tensor_from_numpy",
                wraps=self._evaluator._runtime.tensor_from_numpy,
            ) as mocked_from_numpy,
        ):
            self._evaluator._numpy_to_cpp_tensor("x", contiguous)
            self.assertEqual(mocked_contiguous.call_count, 0)
            self.assertEqual(mocked_from_numpy.call_args.kwargs.get("copy"), False)
            self._evaluator._numpy_to_cpp_tensor("x", non_contiguous)
            self.assertEqual(mocked_contiguous.call_count, 1)
            self.assertEqual(mocked_from_numpy.call_args.kwargs.get("copy"), False)

    def test_bfloat16_uses_ml_dtypes_fallback(self):
        # bfloat16 has no stock NumPy dtype, so it bypasses the DLPack path and
        # is reinterpreted as ``ml_dtypes.bfloat16``.
        ml_dtypes = import_or_skip("ml_dtypes", "bfloat16")
        expected = np.array([1.5, -2.5, 3.0], dtype=ml_dtypes)
        raw = expected.view(np.uint8).ravel()
        t = self._rt.tensor_from_numpy("x", int(onnxl.TensorProto.BFLOAT16), [3], raw)
        out = self._evaluator._cpp_tensor_to_numpy(t)
        self.assertEqual(out.dtype, ml_dtypes)
        np.testing.assert_array_equal(out, expected)


class TestReferenceEvaluatorCustomKernels(ExtTestCase):
    """Tests for :meth:`ReferenceEvaluator.register_custom_kernel`."""

    _CUSTOM_MODEL_SRC = (
        '<ir_version: 10, opset_import: ["" : 18, "my.domain" : 1]>\n'
        "agraph (float[3] x) => (float[3] y) {\n"
        "  y = my.domain.Square(x)\n"
        "}\n"
    )

    def test_custom_kernel_basic(self):
        model = parser.parse_model(self._CUSTOM_MODEL_SRC)
        sess = ReferenceEvaluator(model)
        sess.register_custom_kernel("my.domain", "Square", lambda node, x: x * x)
        x = np.array([-1.0, 2.0, -3.5], dtype=np.float32)
        (y,) = sess.run(None, {"x": x})
        np.testing.assert_array_equal(y, x * x)

    def test_custom_kernel_called_each_run(self):
        # The custom kernel is dispatched on every run() call; verify by
        # using a stateful counter callable.
        model = parser.parse_model(self._CUSTOM_MODEL_SRC)
        sess = ReferenceEvaluator(model)
        calls = []

        def kernel(node, x):
            calls.append(x.shape)
            return x + 1

        sess.register_custom_kernel("my.domain", "Square", kernel)
        sess.run(None, {"x": np.zeros(3, dtype=np.float32)})
        sess.run(None, {"x": np.zeros(3, dtype=np.float32)})
        self.assertEqual(len(calls), 2)

    def test_custom_kernel_overrides_builtin(self):
        # Registering a custom kernel for an existing built-in op overrides it.
        src = (
            '<ir_version: 10, opset_import: ["" : 18]>\n'
            "agraph (float[3] x) => (float[3] y) { y = Abs(x) }\n"
        )
        sess = ReferenceEvaluator(parser.parse_model(src))
        sess.register_custom_kernel("", "Abs", lambda node, x: x + 100.0)
        (y,) = sess.run(None, {"x": np.array([-1.0, 2.0, -3.5], dtype=np.float32)})
        np.testing.assert_array_equal(y, np.array([99.0, 102.0, 96.5], dtype=np.float32))

    def test_custom_kernel_registered_after_run(self):
        # Registering a custom kernel after a first run() must take effect on
        # the next run: register_custom_kernel invalidates the cached
        # RuntimeSession so its per-node kernels are re-resolved.
        model = parser.parse_model(self._CUSTOM_MODEL_SRC)
        sess = ReferenceEvaluator(model)
        x = np.array([-1.0, 2.0, -3.5], dtype=np.float32)

        sess.register_custom_kernel("my.domain", "Square", lambda node, v: v * v)
        (first,) = sess.run(None, {"x": x})
        np.testing.assert_array_equal(first, x * x)

        # Re-register a different kernel for the same op; the next run must use
        # it rather than the kernel cached in the previous run's session.
        sess.register_custom_kernel("my.domain", "Square", lambda node, v: v + 1.0)
        (second,) = sess.run(None, {"x": x})
        np.testing.assert_array_equal(second, x + 1.0)

    def test_custom_kernel_multi_output(self):
        src = (
            '<ir_version: 10, opset_import: ["" : 18, "my.domain" : 1]>\n'
            "agraph (float[3] x) => (float[3] a, float[3] b) {\n"
            "  a, b = my.domain.Split2(x)\n"
            "}\n"
        )
        sess = ReferenceEvaluator(parser.parse_model(src))
        sess.register_custom_kernel("my.domain", "Split2", lambda node, x: (x + 1, x - 1))
        a, b = sess.run(None, {"x": np.array([1.0, 2.0, 3.0], dtype=np.float32)})
        np.testing.assert_array_equal(a, np.array([2.0, 3.0, 4.0], dtype=np.float32))
        np.testing.assert_array_equal(b, np.array([0.0, 1.0, 2.0], dtype=np.float32))

    def test_custom_kernel_reads_attributes(self):
        # The callback receives the NodeProto, so it can inspect attributes.
        src = (
            '<ir_version: 10, opset_import: ["" : 18, "my.domain" : 1]>\n'
            "agraph (float[3] x) => (float[3] y) {\n"
            "  y = my.domain.Scale<factor = 3.0>(x)\n"
            "}\n"
        )
        sess = ReferenceEvaluator(parser.parse_model(src))

        def scale(node, x):
            attr = next(a for a in node.attribute if str(a.name) == "factor")
            return x * float(attr.f)

        sess.register_custom_kernel("my.domain", "Scale", scale)
        (y,) = sess.run(None, {"x": np.array([1.0, 2.0, 3.0], dtype=np.float32)})
        np.testing.assert_array_equal(y, np.array([3.0, 6.0, 9.0], dtype=np.float32))

    def test_custom_kernel_wrong_output_count_raises(self):
        model = parser.parse_model(self._CUSTOM_MODEL_SRC)
        sess = ReferenceEvaluator(model)
        sess.register_custom_kernel("my.domain", "Square", lambda node, x: (x, x))
        with self.assertRaises(ValueError) as ctx:
            sess.run(None, {"x": np.zeros(3, dtype=np.float32)})
        self.assertIn("returned 2 output", str(ctx.exception))

    def test_custom_kernel_chained_with_builtins(self):
        # Mix a custom op with built-in ops in the same graph.
        src = (
            '<ir_version: 10, opset_import: ["" : 18, "my.domain" : 1]>\n'
            "agraph (float[3] x) => (float[3] y) {\n"
            "  t = my.domain.Square(x)\n"
            "  y = Add(t, x)\n"
            "}\n"
        )
        sess = ReferenceEvaluator(parser.parse_model(src))
        sess.register_custom_kernel("my.domain", "Square", lambda node, x: x * x)
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        (y,) = sess.run(None, {"x": x})
        np.testing.assert_array_equal(y, x * x + x)


class TestReferenceEvaluatorUnregisterCustomKernels(ExtTestCase):
    """Tests for :meth:`ReferenceEvaluator.unregister_custom_kernel`."""

    def test_unregister_restores_builtin(self):
        # Overriding a built-in op and then unregistering restores the original.
        src = (
            '<ir_version: 10, opset_import: ["" : 18]>\n'
            "agraph (float[3] x) => (float[3] y) { y = Abs(x) }\n"
        )
        sess = ReferenceEvaluator(parser.parse_model(src))
        x = np.array([-1.0, 2.0, -3.5], dtype=np.float32)

        sess.register_custom_kernel("", "Abs", lambda node, v: v + 100.0)
        (overridden,) = sess.run(None, {"x": x})
        np.testing.assert_array_equal(overridden, x + 100.0)

        self.assertTrue(sess.unregister_custom_kernel("", "Abs"))
        (restored,) = sess.run(None, {"x": x})
        np.testing.assert_array_equal(restored, np.abs(x))

    def test_unregister_returns_false_when_absent(self):
        src = (
            '<ir_version: 10, opset_import: ["" : 18]>\n'
            "agraph (float[3] x) => (float[3] y) { y = Abs(x) }\n"
        )
        sess = ReferenceEvaluator(parser.parse_model(src))
        self.assertFalse(sess.unregister_custom_kernel("", "Abs"))
        self.assertFalse(sess.unregister_custom_kernel("my.domain", "Missing"))

    def test_unregister_domain_normalisation(self):
        # The empty domain is normalised to "ai.onnx", so a kernel registered
        # with "" can be unregistered with "ai.onnx" and vice versa.
        src = (
            '<ir_version: 10, opset_import: ["" : 18]>\n'
            "agraph (float[3] x) => (float[3] y) { y = Abs(x) }\n"
        )
        sess = ReferenceEvaluator(parser.parse_model(src))
        sess.register_custom_kernel("", "Abs", lambda node, v: v + 1.0)
        self.assertTrue(sess.unregister_custom_kernel("ai.onnx", "Abs"))
        (restored,) = sess.run(None, {"x": np.array([-1.0, 2.0, -3.5], dtype=np.float32)})
        np.testing.assert_array_equal(restored, np.array([1.0, 2.0, 3.5], dtype=np.float32))

    def test_unregister_removes_user_domain_kernel(self):
        # Unregistering a custom-only op (no built-in) makes the graph fail
        # again with an unsupported op error.
        model = parser.parse_model(
            '<ir_version: 10, opset_import: ["" : 18, "my.domain" : 1]>\n'
            "agraph (float[3] x) => (float[3] y) {\n"
            "  y = my.domain.Square(x)\n"
            "}\n"
        )
        sess = ReferenceEvaluator(model)
        sess.register_custom_kernel("my.domain", "Square", lambda node, v: v * v)
        x = np.array([-1.0, 2.0, -3.5], dtype=np.float32)
        (y,) = sess.run(None, {"x": x})
        np.testing.assert_array_equal(y, x * x)

        self.assertTrue(sess.unregister_custom_kernel("my.domain", "Square"))
        with self.assertRaises(ValueError) as ctx:
            sess.run(None, {"x": x})
        self.assertIn("unsupported op_type", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
