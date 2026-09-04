# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests the portable STFT decomposition pattern."""

from __future__ import annotations

import unittest

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator

from onnx_light.ext_test_case import ExtTestCase
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh
from onnx_light.onnx import TensorProto
from onnx_light.onnx_core import optimization


def _make_model(dtype, *, onesided, window=True):
    np_dtype = np.float32 if dtype == TensorProto.FLOAT else np.float64
    initializers = [
        onh.from_array(np.asarray(2, dtype=np.int64), name="frame_step"),
        onh.from_array(np.asarray(5, dtype=np.int64), name="frame_length"),
    ]
    window_name = ""
    if window:
        window_name = "window"
        initializers.append(
            onh.from_array(
                np.asarray([0.25, 0.5, 1.0, 0.5, 0.25], dtype=np_dtype), name=window_name
            )
        )
    bins = 3 if onesided else 5
    node = oh.make_node(
        "STFT",
        ["signal", "frame_step", window_name, "frame_length"],
        ["output"],
        onesided=int(onesided),
    )
    graph = oh.make_graph(
        [node],
        "stft",
        [oh.make_tensor_value_info("signal", dtype, [2, 13, 1])],
        [oh.make_tensor_value_info("output", dtype, [2, 5, bins, 2])],
        initializers,
    )
    return oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)], ir_version=10)


def _optimize(model):
    builder = optimization.GraphBuilder(model)
    graph = optimization.GraphGraph(
        builder, [optimization.STFTDecompositionPattern()], use_global_patterns=False
    )
    rewrites = graph.optimize()
    return builder.to_onnx("model"), rewrites


def _upstream_model(model):
    converted = onnx.load_from_string(model.SerializeToString())
    for opset in converted.opset_import:
        if opset.domain == "ai.onnx":
            opset.domain = ""
    return converted


class TestSTFTDecompositionPattern(ExtTestCase):
    def test_numerical_equivalence(self):
        random = np.random.default_rng(42)
        for dtype, np_dtype, tolerance in (
            (TensorProto.FLOAT, np.float32, 2e-5),
            (TensorProto.DOUBLE, np.float64, 1e-11),
        ):
            for onesided in (False, True):
                for use_window in (False, True):
                    with self.subTest(dtype=dtype, onesided=onesided, use_window=use_window):
                        model = _make_model(dtype, onesided=onesided, window=use_window)
                        optimized, rewrites = _optimize(model)
                        self.assertEqual(
                            [rewrite.pattern_name for rewrite in rewrites], ["STFTDecomposition"]
                        )
                        self.assertNotIn("STFT", [node.op_type for node in optimized.graph.node])
                        self.assertEqual(
                            [node.op_type for node in optimized.graph.node],
                            [
                                "Transpose",
                                "Conv",
                                "Conv",
                                "Unsqueeze",
                                "Unsqueeze",
                                "Concat",
                                "Transpose",
                            ],
                        )
                        signal = random.normal(size=(2, 13, 1)).astype(np_dtype)
                        expected = ReferenceEvaluator(_upstream_model(model)).run(
                            None, {"signal": signal}
                        )[0]
                        actual = ReferenceEvaluator(_upstream_model(optimized)).run(
                            None, {"signal": signal}
                        )[0]
                        self.assertEqual(actual.dtype, signal.dtype)
                        self.assertEqual(actual.shape, expected.shape)
                        np.testing.assert_allclose(
                            actual, expected, atol=tolerance, rtol=tolerance
                        )

    def test_rejects_dynamic_inputs(self):
        for dynamic_index in (1, 2, 3):
            with self.subTest(dynamic_index=dynamic_index):
                model = _make_model(TensorProto.FLOAT, onesided=True)
                name = str(model.graph.node[0].input[dynamic_index])
                kept = [
                    initializer
                    for initializer in model.graph.initializer
                    if initializer.name != name
                ]
                del model.graph.initializer[:]
                model.graph.initializer.extend(kept)
                shape = [5] if dynamic_index == 2 else []
                dtype = TensorProto.FLOAT if dynamic_index == 2 else TensorProto.INT64
                model.graph.input.append(oh.make_tensor_value_info(name, dtype, shape))

                optimized, rewrites = _optimize(model)
                self.assertEqual(rewrites, [])
                self.assertEqual([node.op_type for node in optimized.graph.node], ["STFT"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
