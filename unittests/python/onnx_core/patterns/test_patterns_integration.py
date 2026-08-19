# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests cross-family pattern optimization pipelines."""

from __future__ import annotations

import unittest

import numpy as np

from onnx_light.ext_test_case import ExtTestCase, import_or_skip
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh
from onnx_light.onnx import TensorProto

from onnx_light.onnx_core import optimization

ReferenceEvaluator = import_or_skip("onnx_light.onnx.reference", "ReferenceEvaluator")


def _initializer(name, values, dtype):
    """Creates an initializer from NumPy-compatible values."""
    return onh.from_array(np.asarray(values, dtype=dtype), name=name)


def _make_model(nodes, inputs, outputs, initializers=(), *, opset=18):
    """Builds a model with one default-domain opset."""
    graph = oh.make_graph(nodes, "integration", inputs, outputs, list(initializers))
    return oh.make_model(graph, opset_imports=[oh.make_opsetid("", opset)], ir_version=10)


def _optimize(model, pattern_names):
    """Runs only the requested patterns and returns the optimized model."""
    builder = optimization.GraphBuilder(model)
    graph = optimization.GraphGraph(
        builder, optimization.standard_patterns(pattern_names), use_global_patterns=False
    )
    rewrites = graph.optimize()
    return builder.to_onnx("model"), [rewrite.pattern_name for rewrite in rewrites]


def _node_types(model):
    """Returns graph node types in order."""
    return [node.op_type for node in model.graph.node]


class TestPatternsIntegration(ExtTestCase):
    def _assert_equivalent(self, model, optimized, feeds, *, atol=0.0, rtol=1e-6):
        expected = ReferenceEvaluator(model).run(None, feeds)
        got = ReferenceEvaluator(optimized).run(None, feeds)
        self.assertEqual(len(expected), len(got))
        for expected_value, got_value in zip(expected, got):
            np.testing.assert_allclose(got_value, expected_value, atol=atol, rtol=rtol)

    def test_bug_2of3_minimal_reconstruction(self):
        model = _make_model(
            [
                oh.make_node("Expand", ["X", "x_expanded_shape"], ["x_expanded"]),
                oh.make_node("Reshape", ["x_expanded", "x_reshape_shape"], ["x_reshaped"]),
                oh.make_node("Expand", ["Y", "y_expanded_shape"], ["y_expanded"]),
                oh.make_node("Reshape", ["y_expanded", "y_reshape_shape"], ["y_reshaped"]),
                oh.make_node("MatMul", ["x_reshaped", "y_reshaped"], ["product"]),
                oh.make_node("Reshape", ["product", "output_shape"], ["Z"]),
            ],
            [
                oh.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4, 5]),
                oh.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3, 5, 6]),
            ],
            [oh.make_tensor_value_info("Z", TensorProto.FLOAT, [2, 3, 4, 6])],
            [
                _initializer("x_expanded_shape", [2, 3, 4, 5], np.int64),
                _initializer("x_reshape_shape", [6, 4, 5], np.int64),
                _initializer("y_expanded_shape", [2, 3, 5, 6], np.int64),
                _initializer("y_reshape_shape", [6, 5, 6], np.int64),
                _initializer("output_shape", [2, 3, 4, 6], np.int64),
            ],
        )
        optimized, rewrites = _optimize(model, ["Expand", "ReshapeReshape", "MatMulReshape2Of3"])

        self.assertEqual(_node_types(optimized), ["MatMul"])
        self.assertEqual(rewrites.count("Expand"), 2)
        self.assertIn("MatMulReshape2Of3", rewrites)
        feeds = {
            "X": np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5) / 50,
            "Y": np.arange(180, dtype=np.float32).reshape(2, 3, 5, 6) / 60,
        }
        self._assert_equivalent(model, optimized, feeds)

    def test_layernorm_data_minimal_reconstruction_prepares_gelu(self):
        model = _make_model(
            [
                oh.make_node("ReduceMean", ["X", "axes"], ["mean"], keepdims=1),
                oh.make_node("Sub", ["X", "mean"], ["centered"]),
                oh.make_node("Pow", ["centered", "two"], ["squared"]),
                oh.make_node("ReduceMean", ["squared", "axes"], ["variance"], keepdims=1),
                oh.make_node("Add", ["variance", "epsilon"], ["variance_epsilon"]),
                oh.make_node("Sqrt", ["variance_epsilon"], ["deviation"]),
                oh.make_node("Div", ["centered", "deviation"], ["normalized"]),
                oh.make_node("Dropout", ["normalized", "ratio", "training"], ["dropped", "mask"]),
                oh.make_node("Cast", ["dropped"], ["gelu_input"], to=TensorProto.FLOAT),
                oh.make_node("Pow", ["gelu_input", "three"], ["cube"]),
                oh.make_node("Mul", ["cube", "cubic_scale"], ["scaled_cube"]),
                oh.make_node("Add", ["gelu_input", "scaled_cube"], ["sum"]),
                oh.make_node("Mul", ["sum", "sqrt_two_over_pi"], ["scaled_sum"]),
                oh.make_node("Tanh", ["scaled_sum"], ["tanh"]),
                oh.make_node("Add", ["tanh", "one"], ["tanh_plus_one"]),
                oh.make_node("Mul", ["gelu_input", "half"], ["input_half"]),
                oh.make_node("Mul", ["input_half", "tanh_plus_one"], ["Y"]),
            ],
            [oh.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])],
            [
                _initializer("axes", [-1], np.int64),
                _initializer("two", 2.0, np.float32),
                _initializer("epsilon", 1e-5, np.float32),
                _initializer("ratio", 0.0, np.float32),
                _initializer("training", False, np.bool_),
                _initializer("three", 3.0, np.float32),
                _initializer("cubic_scale", 0.044708251953125, np.float32),
                _initializer("sqrt_two_over_pi", 0.7978515625, np.float32),
                _initializer("one", 1.0, np.float32),
                _initializer("half", 0.5, np.float32),
            ],
            opset=20,
        )
        optimized, rewrites = _optimize(model, ["LayerNormalization", "Dropout", "Cast", "Gelu"])

        self.assertEqual(_node_types(optimized), ["LayerNormalization", "Gelu"])
        self.assertEqual(
            [
                name
                for name in rewrites
                if name in {"Cast", "Gelu", "LayerNormalization", "Dropout"}
            ],
            ["Cast", "Gelu", "LayerNormalization", "Dropout"],
        )
        gelu = optimized.graph.node[1]
        approximate = next(
            attribute.s for attribute in gelu.attribute if attribute.name == "approximate"
        )
        self.assertEqual(approximate, b"tanh")
        self._assert_equivalent(
            model, optimized, {"X": np.arange(8, dtype=np.float32).reshape(2, 4) - 3}, atol=2e-5
        )

    def _make_partial_rotary_model(self):
        nodes = [
            oh.make_node("Split", ["X", "outer_split"], ["head", "tail"], axis=-1),
            oh.make_node("Split", ["head"], ["left", "right"], axis=-1, num_outputs=2),
            oh.make_node("Neg", ["right"], ["negative_right"]),
            oh.make_node("Concat", ["negative_right", "left"], ["rotated"], axis=-1),
            oh.make_node("Mul", ["rotated", "sin_cache"], ["rotated_sin"]),
            oh.make_node("Mul", ["head", "cos_cache"], ["source_cos"]),
            oh.make_node("Add", ["rotated_sin", "source_cos"], ["embedded"]),
            oh.make_node("Concat", ["embedded", "tail"], ["Y"], axis=-1),
        ]
        return _make_model(
            nodes,
            [
                oh.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2, 3, 10]),
                oh.make_tensor_value_info("sin_cache", TensorProto.FLOAT, [1, 1, 3, 4]),
                oh.make_tensor_value_info("cos_cache", TensorProto.FLOAT, [1, 1, 3, 4]),
            ],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2, 3, 10])],
            [_initializer("outer_split", [4, 6], np.int64)],
            opset=20,
        )

    def _half_rotary_feeds(self, model):
        random = np.random.default_rng(0)
        return {
            value.name: random.random(
                tuple(dimension.dim_value for dimension in value.type.tensor_type.shape.dim),
                dtype=np.float32,
            )
            for value in model.graph.input
        }

    def test_rotary_prepass_preserves_trailing_slice(self):
        model = self._make_partial_rotary_model()
        optimized, rewrites = _optimize(model, ["FunctionHalfRotaryEmbedding"])

        self.assertEqual(rewrites, ["FunctionHalfRotaryEmbedding"])
        self.assertEqual(_node_types(optimized), ["Split", "HalfRotaryEmbedding", "Concat"])
        self.assertEqual(optimized.graph.node[1].domain, "intermediate")
        self._assert_equivalent(model, optimized, self._half_rotary_feeds(model))

    def test_rotary_prepass_fuses_two_shared_cache_branches(self):
        nodes = [
            oh.make_node("Sin", ["angles"], ["sin_angles"]),
            oh.make_node("Cos", ["angles"], ["cos_angles"]),
            oh.make_node("Unsqueeze", ["sin_angles", "head_axis"], ["sin_cache"]),
            oh.make_node("Unsqueeze", ["cos_angles", "head_axis"], ["cos_cache"]),
        ]
        for prefix in ("query", "key"):
            nodes.extend(
                [
                    oh.make_node(
                        "Split",
                        [prefix],
                        [f"{prefix}_left", f"{prefix}_right"],
                        axis=-1,
                        num_outputs=2,
                    ),
                    oh.make_node("Neg", [f"{prefix}_right"], [f"{prefix}_negative_right"]),
                    oh.make_node(
                        "Concat",
                        [f"{prefix}_negative_right", f"{prefix}_left"],
                        [f"{prefix}_rotated"],
                        axis=-1,
                    ),
                    oh.make_node(
                        "Mul", [f"{prefix}_rotated", "sin_cache"], [f"{prefix}_rotated_sin"]
                    ),
                    oh.make_node("Mul", [prefix, "cos_cache"], [f"{prefix}_cos"]),
                    oh.make_node(
                        "Add", [f"{prefix}_rotated_sin", f"{prefix}_cos"], [f"{prefix}_embedded"]
                    ),
                ]
            )
        model = _make_model(
            nodes,
            [
                oh.make_tensor_value_info("angles", TensorProto.FLOAT, [2, 3, 4]),
                oh.make_tensor_value_info("query", TensorProto.FLOAT, [2, 2, 3, 4]),
                oh.make_tensor_value_info("key", TensorProto.FLOAT, [2, 1, 3, 4]),
            ],
            [
                oh.make_tensor_value_info("query_embedded", TensorProto.FLOAT, [2, 2, 3, 4]),
                oh.make_tensor_value_info("key_embedded", TensorProto.FLOAT, [2, 1, 3, 4]),
            ],
            [_initializer("head_axis", [1], np.int64)],
            opset=20,
        )
        optimized, rewrites = _optimize(model, ["FunctionHalfRotaryEmbedding"])

        self.assertEqual(rewrites.count("FunctionHalfRotaryEmbedding"), 2)
        self.assertEqual(_node_types(optimized).count("HalfRotaryEmbedding"), 2)
        self.assertEqual(
            [(function.name, function.domain) for function in optimized.functions],
            [("HalfRotaryEmbedding", "intermediate")],
        )
        random = np.random.default_rng(42)
        feeds = {
            "angles": random.random((2, 3, 4), dtype=np.float32),
            "query": random.random((2, 2, 3, 4), dtype=np.float32),
            "key": random.random((2, 1, 3, 4), dtype=np.float32),
        }
        self._assert_equivalent(model, optimized, feeds)

    def test_swap_expand_reshape_prepares_rotary_prepass(self):
        model = _make_model(
            [
                oh.make_node("Expand", ["weights", "expanded_shape"], ["expanded"]),
                oh.make_node("Reshape", ["expanded", "reshape_shape"], ["reshaped"]),
                oh.make_node("Unsqueeze", ["reshaped", "cache_axis"], ["cos_cache"]),
                oh.make_node("Split", ["X"], ["left", "right"], axis=-1, num_outputs=2),
                oh.make_node("Neg", ["right"], ["negative_right"]),
                oh.make_node("Concat", ["negative_right", "left"], ["rotated"], axis=-1),
                oh.make_node("Mul", ["rotated", "sin_cache"], ["rotated_sin"]),
                oh.make_node("Mul", ["X", "cos_cache"], ["source_cos"]),
                oh.make_node("Add", ["rotated_sin", "source_cos"], ["Y"]),
            ],
            [
                oh.make_tensor_value_info("X", TensorProto.FLOAT, [2, 1, 1, 2]),
                oh.make_tensor_value_info("sin_cache", TensorProto.FLOAT, [2, 1, 1, 2]),
                oh.make_tensor_value_info("weights", TensorProto.FLOAT, [1, 2, 1]),
            ],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 1, 1, 2])],
            [
                _initializer("expanded_shape", [2, 1, 1], np.int64),
                _initializer("reshape_shape", [0, 1, -1], np.int64),
                _initializer("cache_axis", [2], np.int64),
            ],
            opset=20,
        )
        optimized, rewrites = _optimize(
            model, ["SwapExpandReshape", "FunctionHalfRotaryEmbedding"]
        )

        self.assertEqual(
            [name for name in rewrites if not name.startswith("Remove")],
            ["SwapExpandReshape", "FunctionHalfRotaryEmbedding"],
        )
        self.assertEqual(
            _node_types(optimized), ["Reshape", "Expand", "Unsqueeze", "HalfRotaryEmbedding"]
        )
        random = np.random.default_rng(1)
        feeds = {
            "X": random.random((2, 1, 1, 2), dtype=np.float32),
            "sin_cache": random.random((2, 1, 1, 2), dtype=np.float32),
            "weights": random.random((1, 2, 1), dtype=np.float32),
        }
        self._assert_equivalent(model, optimized, feeds)

    def _make_attention_model_4d(self):
        return _make_model(
            [
                oh.make_node("Transpose", ["query"], ["transposed_query"], perm=[0, 2, 1, 3]),
                oh.make_node("Transpose", ["keys"], ["transposed_keys"], perm=[0, 2, 1, 3]),
                oh.make_node("Transpose", ["values"], ["transposed_values"], perm=[0, 2, 1, 3]),
                oh.make_node(
                    "Concat", ["past_keys", "transposed_keys"], ["concatenated_keys"], axis=-2
                ),
                oh.make_node(
                    "Concat",
                    ["past_values", "transposed_values"],
                    ["concatenated_values"],
                    axis=-2,
                ),
                oh.make_node("Mul", ["transposed_query", "scale"], ["scaled_query"]),
                oh.make_node("Mul", ["concatenated_keys", "scale"], ["scaled_keys"]),
                oh.make_node(
                    "Transpose", ["scaled_keys"], ["transposed_scaled_keys"], perm=[0, 1, 3, 2]
                ),
                oh.make_node("MatMul", ["scaled_query", "transposed_scaled_keys"], ["scores"]),
                oh.make_node("Where", ["mask", "zero", "negative_infinity"], ["bias"]),
                oh.make_node("Add", ["scores", "bias"], ["masked_scores"]),
                oh.make_node("Softmax", ["masked_scores"], ["probabilities"], axis=-1),
                oh.make_node("IsNaN", ["probabilities"], ["nan_mask"]),
                oh.make_node(
                    "Where", ["nan_mask", "zero", "probabilities"], ["filtered_probabilities"]
                ),
                oh.make_node(
                    "MatMul", ["filtered_probabilities", "concatenated_values"], ["context"]
                ),
                oh.make_node("Transpose", ["context"], ["Y"], perm=[0, 2, 1, 3]),
            ],
            [
                oh.make_tensor_value_info("query", TensorProto.FLOAT, [2, 3, 2, 4]),
                oh.make_tensor_value_info("keys", TensorProto.FLOAT, [2, 3, 2, 4]),
                oh.make_tensor_value_info("values", TensorProto.FLOAT, [2, 3, 2, 4]),
                oh.make_tensor_value_info("past_keys", TensorProto.FLOAT, [2, 2, 3, 4]),
                oh.make_tensor_value_info("past_values", TensorProto.FLOAT, [2, 2, 3, 4]),
                oh.make_tensor_value_info("mask", TensorProto.BOOL, [2, 1, 3, 6]),
            ],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3, 2, 4])],
            [
                _initializer("zero", 0.0, np.float32),
                _initializer("negative_infinity", -np.inf, np.float32),
                _initializer("scale", 0.5, np.float32),
            ],
        )

    def test_attention_prepass_four_dimensional(self):
        model = self._make_attention_model_4d()
        optimized, rewrites = _optimize(model, ["FunctionAttention"])

        self.assertEqual(rewrites, ["FunctionAttention"])
        self.assertEqual(
            _node_types(optimized),
            [
                "Transpose",
                "Transpose",
                "Transpose",
                "Concat",
                "Concat",
                "LocalAttention_to1",
                "Transpose",
            ],
        )
        local_attention = optimized.graph.node[5]
        self.assertEqual(local_attention.domain, "intermediate")
        self.assertEqual(
            [(function.name, function.domain) for function in optimized.functions],
            [("LocalAttention_to1", "intermediate")],
        )
        random = np.random.default_rng(5)
        feeds = {
            "query": random.random((2, 3, 2, 4), dtype=np.float32),
            "keys": random.random((2, 3, 2, 4), dtype=np.float32),
            "values": random.random((2, 3, 2, 4), dtype=np.float32),
            "past_keys": random.random((2, 2, 3, 4), dtype=np.float32),
            "past_values": random.random((2, 2, 3, 4), dtype=np.float32),
            "mask": np.ones((2, 1, 3, 6), dtype=np.bool_),
        }
        self._assert_equivalent(model, optimized, feeds)

    def _make_attention_model_3d(self):
        random = np.random.default_rng(7)
        hidden = 8
        heads = 2
        head_size = 4
        return _make_model(
            [
                oh.make_node("MatMul", ["hidden", "query_weight"], ["query_3d"]),
                oh.make_node("Mul", ["query_3d", "scale"], ["scaled_query_3d"]),
                oh.make_node("Reshape", ["scaled_query_3d", "shape_4d"], ["query_4d"]),
                oh.make_node("Transpose", ["query_4d"], ["query"], perm=[0, 2, 1, 3]),
                oh.make_node("MatMul", ["hidden", "key_weight"], ["key_3d"]),
                oh.make_node("Mul", ["key_3d", "scale"], ["scaled_key_3d"]),
                oh.make_node("Reshape", ["scaled_key_3d", "shape_4d"], ["key_4d"]),
                oh.make_node("Transpose", ["key_4d"], ["key"], perm=[0, 2, 3, 1]),
                oh.make_node("MatMul", ["hidden", "value_weight"], ["value_3d"]),
                oh.make_node("Reshape", ["value_3d", "shape_4d"], ["value_4d"]),
                oh.make_node("Transpose", ["value_4d"], ["value"], perm=[0, 2, 1, 3]),
                oh.make_node("MatMul", ["query", "key"], ["scores"]),
                oh.make_node("Where", ["mask", "scores", "negative_infinity"], ["masked_scores"]),
                oh.make_node("Softmax", ["masked_scores"], ["probabilities"], axis=-1),
                oh.make_node("IsNaN", ["probabilities"], ["nan_mask"]),
                oh.make_node(
                    "Where", ["nan_mask", "zero", "probabilities"], ["filtered_probabilities"]
                ),
                oh.make_node("MatMul", ["filtered_probabilities", "value"], ["context"]),
                oh.make_node("Transpose", ["context"], ["transposed_context"], perm=[0, 2, 1, 3]),
                oh.make_node("Reshape", ["transposed_context", "shape_3d"], ["Y"]),
            ],
            [
                oh.make_tensor_value_info("hidden", TensorProto.FLOAT, [2, 3, hidden]),
                oh.make_tensor_value_info("mask", TensorProto.BOOL, [1, 1, 3, 3]),
            ],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3, hidden])],
            [
                _initializer(
                    "query_weight", random.random((hidden, hidden), dtype=np.float32), np.float32
                ),
                _initializer(
                    "key_weight", random.random((hidden, hidden), dtype=np.float32), np.float32
                ),
                _initializer(
                    "value_weight", random.random((hidden, hidden), dtype=np.float32), np.float32
                ),
                _initializer("scale", head_size**-0.5, np.float32),
                _initializer("shape_4d", [0, 0, heads, head_size], np.int64),
                _initializer("shape_3d", [0, 0, hidden], np.int64),
                _initializer("zero", 0.0, np.float32),
                _initializer("negative_infinity", -np.inf, np.float32),
            ],
            opset=22,
        )

    def test_attention_prepass_three_dimensional(self):
        model = self._make_attention_model_3d()
        optimized, rewrites = _optimize(model, ["FunctionAttention"])

        self.assertEqual(rewrites, ["FunctionAttention"])
        local_attention = [
            node
            for node in optimized.graph.node
            if node.op_type == "LocalAttention_to1" and node.domain == "intermediate"
        ]
        self.assertEqual(len(local_attention), 1)
        self.assertNotIn("Softmax", _node_types(optimized))
        random = np.random.default_rng(11)
        feeds = {
            "hidden": random.random((2, 3, 8), dtype=np.float32),
            "mask": np.ones((1, 1, 3, 3), dtype=np.bool_),
        }
        self._assert_equivalent(model, optimized, feeds)


if __name__ == "__main__":
    unittest.main(verbosity=2)
