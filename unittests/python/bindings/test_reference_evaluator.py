# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for :class:`onnx_light.reference.ReferenceEvaluator`.

The evaluator is a thin Python layer on top of the C++ kernels Python
API (``onnx_light.onnx_py._onnxkernels.runtime``). These tests cover
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
from onnx_light.reference import ReferenceEvaluator

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

    def test_missing_input_raises(self):
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
