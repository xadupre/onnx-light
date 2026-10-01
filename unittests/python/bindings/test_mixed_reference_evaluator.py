# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for the mixed native/NumPy reference evaluator."""

import numpy as np

import onnx_light.onnx.helper as oh
from onnx_light.onnx_lib import TensorProto
from onnx_light.ext_test_case import ExtTestCase, import_or_skip

MixedReferenceEvaluator = import_or_skip("onnx_light.onnx.reference", "MixedReferenceEvaluator")
ReferenceEvaluator = import_or_skip("onnx_light.onnx.reference", "ReferenceEvaluator")
_attention = import_or_skip("onnx_light._reference.ops", "_attention")


def _model(op_type, inputs, outputs, **attributes):
    """Builds a single-node model with float inputs and outputs."""
    node = oh.make_node(op_type, inputs, outputs, **attributes)
    graph = oh.make_graph(
        [node],
        "mixed",
        [
            oh.make_tensor_value_info(
                name,
                (
                    TensorProto.BOOL
                    if name == "mask"
                    else TensorProto.INT64 if name == "nonpad_kv_seqlen" else TensorProto.FLOAT
                ),
                None,
            )
            for name in inputs
            if name
        ],
        [oh.make_tensor_value_info(name, TensorProto.FLOAT, None) for name in outputs if name],
    )
    return oh.make_model(graph, opset_imports=[oh.make_operatorsetid("", 23)])


class TestMixedReferenceEvaluator(ExtTestCase):
    """Exercises the automatically registered NumPy overrides."""

    def test_attention_rejects_incomplete_cache(self):
        node = oh.make_node("Attention", ["q", "k", "v"], ["y"])
        tensor = np.ones((1, 1, 1, 1), dtype=np.float32)
        for cache in ({"past_key": tensor}, {"past_value": tensor}):
            with (
                self.subTest(cache=tuple(cache)),
                self.assertRaisesRegex(
                    ValueError, "past_key and past_value must be provided together"
                ),
            ):
                _attention(node, tensor, tensor, tensor, **cache)

    def test_inherits_and_preserves_native_ops(self):
        model = _model("Abs", ["x"], ["y"])
        evaluator = MixedReferenceEvaluator(model)
        self.assertIsInstance(evaluator, ReferenceEvaluator)
        np.testing.assert_array_equal(
            evaluator.run(None, {"x": np.array([-2, 3], dtype=np.float32)})[0], [2, 3]
        )
        self.assertEqual(
            set(evaluator._custom_kernels),
            {"ai.onnx:Gemm", "ai.onnx:MatMul", "ai.onnx:Conv", "ai.onnx:Attention"},
        )

    def test_gemm_attributes_and_optional_bias(self):
        a = np.arange(6, dtype=np.float32).reshape(3, 2)
        b = np.arange(12, dtype=np.float32).reshape(4, 3)
        c = np.ones((2, 4), dtype=np.float32)
        model = _model("Gemm", ["a", "b", "c"], ["y"], transA=1, transB=1, alpha=2.0, beta=3.0)
        evaluator = MixedReferenceEvaluator(model)
        for _ in range(2):
            np.testing.assert_allclose(
                evaluator.run(None, {"a": a, "b": b, "c": c})[0], 2 * (a.T @ b.T) + 3 * c
            )

    def test_matmul_broadcast(self):
        a = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
        b = np.ones((3, 4), dtype=np.float32)
        result = MixedReferenceEvaluator(_model("MatMul", ["a", "b"], ["y"])).run(
            None, {"a": a, "b": b}
        )[0]
        np.testing.assert_array_equal(result, a @ b)

    def test_conv_grouped_dilated_and_bias(self):
        x = np.arange(14, dtype=np.float32).reshape(1, 2, 7)
        w = np.ones((2, 1, 2), dtype=np.float32)
        b = np.array([1, -1], dtype=np.float32)
        result = MixedReferenceEvaluator(
            _model("Conv", ["x", "w", "b"], ["y"], group=2, dilations=[2], pads=[1, 0])
        ).run(None, {"x": x, "w": w, "b": b})[0]
        padded = np.pad(x, ((0, 0), (0, 0), (1, 0)))
        expected = padded[:, :, :-2] + padded[:, :, 2:] + b[None, :, None]
        np.testing.assert_array_equal(result, expected)

    def test_conv_same_lower(self):
        x = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
        w = np.ones((1, 1, 3, 3), dtype=np.float32)
        result = MixedReferenceEvaluator(
            _model("Conv", ["x", "w"], ["y"], auto_pad="SAME_LOWER", strides=[2, 2])
        ).run(None, {"x": x, "w": w})[0]
        padded = np.pad(x, ((0, 0), (0, 0), (1, 0), (1, 0)))
        expected = np.array(
            [[[[padded[0, 0, i : i + 3, j : j + 3].sum() for j in (0, 2)] for i in (0, 2)]]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(result, expected)

    def test_attention_causal_mask_cache_and_outputs(self):
        q = np.ones((1, 2, 1, 2), dtype=np.float32)
        k = np.array([[[[1, 0]]]], dtype=np.float32)
        v = np.array([[[[3, 5]]]], dtype=np.float32)
        past_key = np.array([[[[0, 1]]]], dtype=np.float32)
        past_value = np.array([[[[7, 9]]]], dtype=np.float32)
        mask = np.array([True, False])
        model = _model(
            "Attention",
            ["q", "k", "v", "mask", "past_key", "past_value"],
            ["y", "present_key", "present_value", "qk"],
            is_causal=1,
            qk_matmul_output_mode=3,
        )
        y, present_key, present_value, weights = MixedReferenceEvaluator(model).run(
            None,
            {
                "q": q,
                "k": k,
                "v": v,
                "mask": mask,
                "past_key": past_key,
                "past_value": past_value,
            },
        )
        np.testing.assert_array_equal(y, np.broadcast_to(past_value, y.shape))
        np.testing.assert_array_equal(present_key, np.concatenate([past_key, k], axis=2))
        np.testing.assert_array_equal(present_value, np.concatenate([past_value, v], axis=2))
        np.testing.assert_array_equal(weights, np.array([[[[1, 0]], [[1, 0]]]], dtype=np.float32))

    def test_attention_nonpad_causal_offset(self):
        q = np.ones((1, 1, 1, 1), dtype=np.float32)
        k = np.ones((1, 1, 3, 1), dtype=np.float32)
        v = np.array([[[[2], [4], [8]]]], dtype=np.float32)
        result = MixedReferenceEvaluator(
            _model(
                "Attention", ["q", "k", "v", "", "", "", "nonpad_kv_seqlen"], ["y"], is_causal=1
            )
        ).run(None, {"q": q, "k": k, "v": v, "nonpad_kv_seqlen": np.array([2], dtype=np.int64)})[
            0
        ]
        np.testing.assert_array_equal(result, np.array([[[[3]]]], dtype=np.float32))

    def test_attention_rank3_grouped_heads(self):
        q = np.ones((1, 2, 2), dtype=np.float32)
        k = np.ones((1, 2, 1), dtype=np.float32)
        v = np.array([[[2], [6]]], dtype=np.float32)
        y = MixedReferenceEvaluator(
            _model("Attention", ["q", "k", "v"], ["y"], q_num_heads=2, kv_num_heads=1)
        ).run(None, {"q": q, "k": k, "v": v})[0]
        np.testing.assert_array_equal(y, np.full((1, 2, 2), 4, dtype=np.float32))
