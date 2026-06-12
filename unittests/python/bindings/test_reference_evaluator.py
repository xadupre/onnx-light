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
import tempfile
import unittest

import numpy as np

from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx_lib import parser
from onnx_light.onnx.reference import ReferenceEvaluator

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

    def test_loop_zero_trip_count(self):
        # Regression test for ``test_cc_loop_zero_trip_count``: when ``M = 0``
        # the loop runs zero iterations and the scan output is empty along
        # its leading axis. The kernel has no per-iteration template to seed
        # the dtype/shape from, so ``RunLoopNode`` must patch the empty scan
        # output's dtype/trailing shape from the body's declared output
        # value-info; otherwise the downstream numpy conversion raises
        # ``"The element type in the input tensor is UNDEFINED."``.
        from onnx_light.onnx_lib.backend.test.case import collect_test_case

        tc = collect_test_case().get("test_cc_loop_zero_trip_count")
        self.assertIsNotNone(tc)
        inputs, outputs = tc.data_sets[0]
        sess = ReferenceEvaluator(tc.model)
        got = sess.run(None, dict(zip(sess.input_names, inputs)))
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0].dtype, outputs[0].dtype)
        self.assertEqual(got[0].shape, outputs[0].shape)
        np.testing.assert_array_equal(got[0], outputs[0])

    def test_sequence_erase_pos1_returns_list(self):
        # Regression test for test_cc_sequence_erase_pos1: the graph output
        # is a SequenceType, so ReferenceEvaluator.run() must return a list
        # of numpy arrays rather than raise "Output was not produced".
        from onnx_light.onnx_lib.backend.test.case import collect_test_case

        tc = collect_test_case().get("test_cc_sequence_erase_pos1")
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

    def test_image_decoder_decode_bmp_rgb(self):
        # Regression test for ``test_cc_image_decoder_decode_bmp_rgb``: the
        # ``ImageDecoder`` kernel must decode the BMP bytestream and return
        # the correct ``(32, 32, 3)`` uint8 tensor rather than the empty-matrix
        # fallback ``(0, 0, 3)`` returned by earlier versions of the kernel.
        from onnx_light.onnx_lib.backend.test.case import collect_test_case

        tc = collect_test_case().get("test_cc_image_decoder_decode_bmp_rgb")
        self.assertIsNotNone(tc)
        inputs, outputs = tc.data_sets[0]
        sess = ReferenceEvaluator(tc.model)
        got = sess.run(None, dict(zip(sess.input_names, inputs)))
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0].dtype, np.uint8)
        self.assertEqual(got[0].shape, (32, 32, 3))
        self.assertEqual(got[0].shape, outputs[0].shape)
        np.testing.assert_array_equal(got[0], outputs[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
