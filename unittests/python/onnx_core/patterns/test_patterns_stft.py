# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests the STFT fusion pattern."""

from __future__ import annotations

import unittest

import numpy as np

from onnx_light.ext_test_case import ExtTestCase, import_or_skip
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh
from onnx_light.onnx import TensorProto
from onnx_light.onnx_core import optimization

ReferenceEvaluator = import_or_skip("onnx_light.onnx.reference", "ReferenceEvaluator")


def _make_model(dtype, *, onesided, frame_length):
    np_dtype = np.float32 if dtype == TensorProto.FLOAT else np.float64
    frame_step = 2
    bins = frame_length // 2 + 1 if onesided else frame_length
    window = np.hanning(frame_length + 2)[1:-1].astype(np_dtype)
    frequencies = np.arange(bins, dtype=np.float64)[:, None]
    samples = np.arange(frame_length, dtype=np.float64)[None, :]
    angles = -2.0 * np.pi * frequencies * samples / frame_length
    real_weights = (np.cos(angles) * window).astype(np_dtype)[:, None, :]
    imag_weights = (np.sin(angles) * window).astype(np_dtype)[:, None, :]

    initializers = [
        onh.from_array(real_weights, name="real_weights"),
        onh.from_array(imag_weights, name="imag_weights"),
        onh.from_array(np.asarray([3], dtype=np.int64), name="axes"),
    ]
    transpose_signal = oh.make_node("Transpose", ["signal"], ["transposed"], perm=[0, 2, 1])
    real = oh.make_node("Conv", ["transposed", "real_weights"], ["real"], strides=[frame_step])
    imag = oh.make_node("Conv", ["transposed", "imag_weights"], ["imag"], strides=[frame_step])
    real_unsqueeze = oh.make_node("Unsqueeze", ["real", "axes"], ["real_u"])
    imag_unsqueeze = oh.make_node("Unsqueeze", ["imag", "axes"], ["imag_u"])
    concat = oh.make_node("Concat", ["real_u", "imag_u"], ["complex"], axis=3)
    transpose_output = oh.make_node(
        "Transpose", ["complex"], ["output"], perm=[0, 2, 1, 3], name="final"
    )
    graph = oh.make_graph(
        [transpose_signal, real, imag, real_unsqueeze, imag_unsqueeze, concat, transpose_output],
        "stft_convolution",
        [oh.make_tensor_value_info("signal", dtype, [2, 13, 1])],
        [
            oh.make_tensor_value_info(
                "output", dtype, [2, 1 + (13 - frame_length) // frame_step, bins, 2]
            )
        ],
        initializers,
    )
    return oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)], ir_version=10)


def _optimize(model):
    builder = optimization.GraphBuilder(model)
    graph = optimization.GraphGraph(
        builder, [optimization.STFTFusionPattern()], use_global_patterns=False
    )
    rewrites = graph.optimize()
    return builder.to_onnx("model"), rewrites


class TestSTFTFusionPattern(ExtTestCase):
    def test_registry_name(self):
        patterns = optimization.standard_patterns(["STFTFusion"])
        self.assertEqual(len(patterns), 1)
        self.assertIsInstance(patterns[0], optimization.STFTFusionPattern)

    def test_numerical_equivalence(self):
        random = np.random.default_rng(42)
        for onesided in (False, True):
            for frame_length in (5, 6):
                with self.subTest(onesided=onesided, frame_length=frame_length):
                    model = _make_model(
                        TensorProto.FLOAT, onesided=onesided, frame_length=frame_length
                    )
                    optimized, rewrites = _optimize(model)
                    self.assertEqual(
                        [rewrite.pattern_name for rewrite in rewrites], ["STFTFusion"]
                    )
                    self.assertEqual([node.op_type for node in optimized.graph.node], ["STFT"])
                    self.assertEqual(optimized.graph.node[0].attribute[0].i, int(onesided))

                    signal = random.normal(size=(2, 13, 1)).astype(np.float32)
                    expected = ReferenceEvaluator(model).run(None, {"signal": signal})[0]
                    actual = ReferenceEvaluator(optimized).run(None, {"signal": signal})[0]
                    self.assertEqual(actual.dtype, signal.dtype)
                    self.assertEqual(actual.shape, expected.shape)
                    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
