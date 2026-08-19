# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Ports upstream xoptim canonicalization tests to isolated onnx-light patterns."""

from __future__ import annotations

import unittest

import numpy as np

from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx import TensorProto, helper, numpy_helper
from onnx_light.onnx.reference import ReferenceEvaluator
from onnx_light.onnx_core import optimization


def _value_info(name, dtype, shape):
    """Creates a tensor value-info."""
    return helper.make_tensor_value_info(name, dtype, shape)


def _model(nodes, inputs, outputs, initializers=(), opset=18, name="canonicalization"):
    """Creates a model with one default-domain opset."""
    graph = helper.make_graph(nodes, name, inputs, outputs, list(initializers))
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 10
    return model


def _array(values, dtype, name):
    """Creates an initializer from a NumPy array."""
    return numpy_helper.from_array(np.array(values, dtype=dtype), name=name)


def _op_types(model):
    """Returns the main-graph operator types."""
    return [node.op_type for node in model.graph.node]


def _node_attribute_ints(node, name):
    """Returns an integer-list node attribute."""
    return list(next(attribute for attribute in node.attribute if attribute.name == name).ints)


def _if_branches(model):
    """Returns the then and else graphs from the model's If node."""
    if_node = next(node for node in model.graph.node if node.op_type == "If")
    branches = {
        attribute.name: attribute.g for attribute in if_node.attribute if attribute.has_g()
    }
    return branches["then_branch"], branches["else_branch"]


def _recursive_op_types(graph):
    """Returns all operator types in a graph and its nested graphs."""
    result = []
    for node in graph.node:
        result.append(node.op_type)
        for attribute in node.attribute:
            if attribute.has_g():
                result.extend(_recursive_op_types(attribute.g))
    return result


def _cast_reshape_model(dynamic=False, keep_intermediate=False):
    """Creates the upstream reshape-matmul topology containing a redundant Cast."""
    x_shape = ["D2", "D3"] if dynamic else [2, 3]
    y_shape = ["batch", "channel", "D3", "D4"] if dynamic else [2, 2, 3, 4]
    z_shape = ["batch", "channel", "D2", "D4"] if dynamic else [2, 2, 2, 4]
    outputs = [_value_info("Z", TensorProto.FLOAT, z_shape)]
    if keep_intermediate:
        outputs.append(_value_info("xm1", TensorProto.FLOAT, [1, 2, 3]))
    return _model(
        [
            helper.make_node("Unsqueeze", ["X", "zero"], ["xu1"]),
            helper.make_node("Unsqueeze", ["xu1", "one"], ["xu2"]),
            helper.make_node("Reshape", ["xu2", "shape1"], ["xm1"]),
            helper.make_node("Reshape", ["Y", "shape2"], ["xm2c"]),
            helper.make_node("Cast", ["xm2c"], ["xm2"], to=TensorProto.FLOAT),
            helper.make_node("MatMul", ["xm1", "xm2"], ["xm"]),
            helper.make_node("Reshape", ["xm", "shape3"], ["Z"]),
        ],
        [
            _value_info("X", TensorProto.FLOAT, x_shape),
            _value_info("Y", TensorProto.FLOAT, y_shape),
        ],
        outputs,
        [
            _array([0], np.int64, "zero"),
            _array([1], np.int64, "one"),
            _array([1, 2, 3], np.int64, "shape1"),
            _array([4, 3, 4], np.int64, "shape2"),
            _array([2, 2, 2, 4], np.int64, "shape3"),
        ],
    )


def _layernorm_model(tail_nodes, output_name, extra_initializers=(), opset=20):
    """Creates a small layer-normalization fragment replacing the upstream artifact."""
    nodes = [
        helper.make_node(
            "LayerNormalization",
            ["X", "scale", "bias"],
            ["normalized", "mean", "inv_std"],
            axis=-1,
            epsilon=1e-5,
        ),
        *tail_nodes,
    ]
    return _model(
        nodes,
        [_value_info("X", TensorProto.FLOAT, [2, 4])],
        [_value_info(output_name, TensorProto.FLOAT, [2, 4])],
        [
            _array(np.ones(4), np.float32, "scale"),
            _array(np.zeros(4), np.float32, "bias"),
            *extra_initializers,
        ],
        opset=opset,
        name="minimal_layernorm",
    )


def _constant_capture_model(then_value, else_value):
    """Creates an If model whose branches capture X and shadow an outer initializer."""
    then_graph = helper.make_graph(
        [
            helper.make_node(
                "Constant",
                [],
                ["two"],
                value=numpy_helper.from_array(np.array([then_value], dtype=np.float32)),
            ),
            helper.make_node("Add", ["X", "two"], ["shifted"]),
            helper.make_node("Add", ["shifted", "two_cst2init"], ["branch_output"]),
        ],
        "then_branch",
        [],
        [_value_info("branch_output", TensorProto.FLOAT, [3])],
        [_array([0], np.float32, "two_cst2init")],
    )
    else_graph = helper.make_graph(
        [
            helper.make_node(
                "Constant",
                [],
                ["two"],
                value=numpy_helper.from_array(np.array([else_value], dtype=np.float32)),
            ),
            helper.make_node("Sub", ["X", "two"], ["shifted"]),
            helper.make_node("Add", ["shifted", "two_cst2init"], ["branch_output"]),
        ],
        "else_branch",
        [],
        [_value_info("branch_output", TensorProto.FLOAT, [3])],
        [_array([0], np.float32, "two_cst2init")],
    )
    return _model(
        [
            helper.make_node("ReduceSum", ["X"], ["sum"]),
            helper.make_node("Greater", ["sum", "threshold"], ["condition"]),
            helper.make_node(
                "If", ["condition"], ["Y"], then_branch=then_graph, else_branch=else_graph
            ),
        ],
        [_value_info("X", TensorProto.FLOAT, [3])],
        [_value_info("Y", TensorProto.FLOAT, [3])],
        [_array(0, np.float32, "threshold"), _array([99], np.float32, "outer_two")],
    )


class TestCanonicalizationPatterns(ExtTestCase):
    def _optimize(self, model, pattern):
        builder = optimization.GraphBuilder(model)
        graph = optimization.GraphGraph(builder, [pattern], use_global_patterns=False)
        all_rewrites, report = graph.optimize(report=True)
        self.assertEqual(report.rewrites, len(all_rewrites))
        rewrites = [rewrite for rewrite in all_rewrites if rewrite.pattern_name == pattern]
        return builder.to_onnx("model"), rewrites, report

    def _assert_equivalent(self, model, optimized, feeds, atol=0):
        expected = ReferenceEvaluator(model).run(None, feeds)
        got = ReferenceEvaluator(optimized).run(None, feeds)
        self.assertEqual(len(expected), len(got))
        for expected_value, got_value in zip(expected, got, strict=True):
            if np.issubdtype(expected_value.dtype, np.floating):
                np.testing.assert_allclose(expected_value, got_value, rtol=0, atol=atol)
            else:
                np.testing.assert_array_equal(expected_value, got_value)

    def test_cast_reshape_matmul_topologies(self):
        feeds = {
            "X": np.arange(6, dtype=np.float32).reshape(2, 3) / 7,
            "Y": np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4) / 11,
        }
        cases = (
            ("static", False, False),
            ("dynamic_1", True, False),
            ("dynamic_2", True, False),
            ("keep_intermediate", False, True),
        )
        for case, dynamic, keep_intermediate in cases:
            with self.subTest(case=case):
                model = _cast_reshape_model(dynamic, keep_intermediate)
                optimized, rewrites, _ = self._optimize(model, "Cast")
                self.assertEqual(
                    _op_types(optimized),
                    ["Unsqueeze", "Unsqueeze", "Reshape", "Reshape", "MatMul", "Reshape"],
                )
                self.assertEqual(len(rewrites), 1)
                self.assertEqual(
                    {initializer.name for initializer in optimized.graph.initializer},
                    {"zero", "one", "shape1", "shape2", "shape3"},
                )
                self._assert_equivalent(model, optimized, feeds)

    def test_cast_minimal_layernorm_artifact_replacement(self):
        model = _layernorm_model(
            [helper.make_node("Cast", ["normalized"], ["Y"], to=TensorProto.FLOAT)], "Y"
        )
        feeds = {"X": np.arange(8, dtype=np.float32).reshape(2, 4)}

        optimized, rewrites, _ = self._optimize(model, "Cast")

        self.assertEqual(_op_types(optimized), ["LayerNormalization", "Identity"])
        self.assertEqual(len(rewrites), 1)
        self.assertEqual(len(optimized.graph.initializer), 2)
        self._assert_equivalent(model, optimized, feeds, atol=1e-6)

    def test_cast_no_match_when_type_changes(self):
        model = _model(
            [helper.make_node("Cast", ["X"], ["Y"], to=TensorProto.FLOAT16)],
            [_value_info("X", TensorProto.FLOAT, [2, 3])],
            [_value_info("Y", TensorProto.FLOAT16, [2, 3])],
        )
        feeds = {"X": np.arange(6, dtype=np.float32).reshape(2, 3) / 7}

        optimized, rewrites, _ = self._optimize(model, "Cast")

        self.assertEqual(_op_types(optimized), ["Cast"])
        self.assertEqual(len(rewrites), 0)
        self._assert_equivalent(model, optimized, feeds)

    def test_cast_cast_identity(self):
        model = _model(
            [
                helper.make_node("Add", ["X", "one"], ["x1"]),
                helper.make_node("Cast", ["x1"], ["x2"], to=TensorProto.FLOAT),
                helper.make_node("Cast", ["x2"], ["Y"], to=TensorProto.FLOAT16),
            ],
            [_value_info("X", TensorProto.FLOAT16, [3, 4])],
            [_value_info("Y", TensorProto.FLOAT16, [3, 4])],
            [_array([1], np.float16, "one")],
        )
        feeds = {"X": (np.arange(12).reshape(3, 4) % 3).astype(np.float16)}

        optimized, rewrites, _ = self._optimize(model, "CastCast")

        self.assertEqual(_op_types(optimized), ["Add", "Identity"])
        self.assertEqual(len(rewrites), 1)
        self._assert_equivalent(model, optimized, feeds)

    def test_cast_cast_cast(self):
        model = _model(
            [
                helper.make_node("Add", ["X", "one"], ["x1"]),
                helper.make_node("Cast", ["x1"], ["x2"], to=TensorProto.FLOAT),
                helper.make_node("Cast", ["x2"], ["Y"], to=TensorProto.FLOAT),
            ],
            [_value_info("X", TensorProto.FLOAT16, [3, 4])],
            [_value_info("Y", TensorProto.FLOAT, [3, 4])],
            [_array([1], np.float16, "one")],
        )
        feeds = {"X": (np.arange(12).reshape(3, 4) % 3).astype(np.float16)}

        optimized, rewrites, _ = self._optimize(model, "CastCast")

        self.assertEqual(_op_types(optimized), ["Add", "Cast"])
        self.assertEqual(len(rewrites), 1)
        self._assert_equivalent(model, optimized, feeds)

    def test_cast_cast_no_match(self):
        model = _model(
            [
                helper.make_node("Cast", ["X"], ["middle"], to=TensorProto.FLOAT),
                helper.make_node("Cast", ["middle"], ["Y"], to=TensorProto.INT64),
            ],
            [_value_info("X", TensorProto.FLOAT16, [2, 3])],
            [_value_info("Y", TensorProto.INT64, [2, 3])],
        )
        feeds = {"X": np.arange(6, dtype=np.float16).reshape(2, 3)}

        optimized, rewrites, _ = self._optimize(model, "CastCast")

        self.assertEqual(_op_types(optimized), ["Cast", "Cast"])
        self.assertEqual(len(rewrites), 0)
        self._assert_equivalent(model, optimized, feeds)

    def test_cast_cast_keeps_used_intermediate_output(self):
        model = _model(
            [
                helper.make_node("Cast", ["X"], ["middle"], to=TensorProto.FLOAT),
                helper.make_node("Cast", ["middle"], ["Y"], to=TensorProto.FLOAT16),
            ],
            [_value_info("X", TensorProto.FLOAT16, [2, 3])],
            [
                _value_info("middle", TensorProto.FLOAT, [2, 3]),
                _value_info("Y", TensorProto.FLOAT16, [2, 3]),
            ],
        )
        feeds = {"X": np.arange(6, dtype=np.float16).reshape(2, 3)}

        optimized, rewrites, _ = self._optimize(model, "CastCast")

        self.assertEqual(_op_types(optimized), ["Cast", "Identity"])
        self.assertEqual(len(rewrites), 1)
        self._assert_equivalent(model, optimized, feeds)

    def test_cast_cast_binary(self):
        model = _model(
            [
                helper.make_node("Cast", ["X"], ["xc"], to=TensorProto.FLOAT16),
                helper.make_node("Cast", ["Y"], ["yc"], to=TensorProto.FLOAT16),
                helper.make_node("Add", ["xc", "yc"], ["Z"]),
            ],
            [
                _value_info("X", TensorProto.FLOAT, [3, 4]),
                _value_info("Y", TensorProto.FLOAT, [3, 4]),
            ],
            [_value_info("Z", TensorProto.FLOAT16, [3, 4])],
        )
        feeds = {
            "X": np.arange(12, dtype=np.float32).reshape(3, 4),
            "Y": np.arange(12, dtype=np.float32).reshape(3, 4),
        }

        optimized, rewrites, _ = self._optimize(model, "CastCastBinary")

        self.assertEqual(_op_types(optimized), ["Add", "Cast"])
        self.assertEqual(len(rewrites), 1)
        self.assertEqual(len(optimized.graph.initializer), 0)
        self._assert_equivalent(model, optimized, feeds)

    def test_cast_cast_binary_no_match_integer_sources(self):
        model = _model(
            [
                helper.make_node("Cast", ["X"], ["xc"], to=TensorProto.FLOAT16),
                helper.make_node("Cast", ["Y"], ["yc"], to=TensorProto.FLOAT16),
                helper.make_node("Add", ["xc", "yc"], ["Z"]),
            ],
            [
                _value_info("X", TensorProto.INT64, [3, 4]),
                _value_info("Y", TensorProto.INT64, [3, 4]),
            ],
            [_value_info("Z", TensorProto.FLOAT16, [3, 4])],
        )
        feeds = {
            "X": np.arange(12, dtype=np.int64).reshape(3, 4),
            "Y": np.arange(12, dtype=np.int64).reshape(3, 4),
        }

        optimized, rewrites, _ = self._optimize(model, "CastCastBinary")

        self.assertEqual(_op_types(optimized), ["Cast", "Cast", "Add"])
        self.assertEqual(len(rewrites), 0)
        self._assert_equivalent(model, optimized, feeds)

    def test_cast_cast_binary_no_match_when_cast_output_is_used(self):
        model = _model(
            [
                helper.make_node("Cast", ["X"], ["xc"], to=TensorProto.FLOAT16),
                helper.make_node("Cast", ["Y"], ["yc"], to=TensorProto.FLOAT16),
                helper.make_node("Add", ["xc", "yc"], ["Z"]),
            ],
            [
                _value_info("X", TensorProto.FLOAT, [2, 3]),
                _value_info("Y", TensorProto.FLOAT, [2, 3]),
            ],
            [
                _value_info("xc", TensorProto.FLOAT16, [2, 3]),
                _value_info("Z", TensorProto.FLOAT16, [2, 3]),
            ],
        )
        feeds = {
            "X": np.arange(6, dtype=np.float32).reshape(2, 3),
            "Y": np.arange(6, dtype=np.float32).reshape(2, 3),
        }

        optimized, rewrites, _ = self._optimize(model, "CastCastBinary")

        self.assertEqual(_op_types(optimized), ["Cast", "Cast", "Add"])
        self.assertEqual(len(rewrites), 0)
        self._assert_equivalent(model, optimized, feeds)

    def test_cast_op_cast_binary(self):
        model = _model(
            [
                helper.make_node("Cast", ["Y"], ["yc"], to=TensorProto.FLOAT),
                helper.make_node("Add", ["X", "yc"], ["computed"]),
                helper.make_node("Cast", ["computed"], ["Z"], to=TensorProto.FLOAT16),
            ],
            [
                _value_info("X", TensorProto.FLOAT, [2, 3]),
                _value_info("Y", TensorProto.FLOAT16, [2, 3]),
            ],
            [_value_info("Z", TensorProto.FLOAT16, [2, 3])],
        )
        feeds = {
            "X": np.arange(6, dtype=np.float32).reshape(2, 3),
            "Y": np.arange(6, dtype=np.float16).reshape(2, 3),
        }

        optimized, rewrites, _ = self._optimize(model, "CastOpCast")

        self.assertEqual(_op_types(optimized), ["Cast", "Add"])
        self.assertEqual(len(rewrites), 1)
        self.assertNotEqual(
            [tuple(node.input) for node in model.graph.node],
            [tuple(node.input) for node in optimized.graph.node],
        )
        self._assert_equivalent(model, optimized, feeds)

    def test_cast_op_cast_unary(self):
        model = _model(
            [
                helper.make_node("Cast", ["X"], ["xc"], to=TensorProto.FLOAT16),
                helper.make_node("Neg", ["xc"], ["computed"]),
                helper.make_node("Cast", ["computed"], ["Y"], to=TensorProto.FLOAT),
            ],
            [_value_info("X", TensorProto.FLOAT, [2, 3])],
            [_value_info("Y", TensorProto.FLOAT, [2, 3])],
        )
        feeds = {"X": np.arange(6, dtype=np.float32).reshape(2, 3)}

        optimized, rewrites, _ = self._optimize(model, "CastOpCast")

        self.assertEqual(_op_types(optimized), ["Neg"])
        self.assertEqual(len(rewrites), 1)
        self._assert_equivalent(model, optimized, feeds, atol=1e-3)

    def test_cast_op_cast_no_match_without_input_cast(self):
        model = _model(
            [
                helper.make_node("Add", ["X", "Y"], ["computed"]),
                helper.make_node("Cast", ["computed"], ["Z"], to=TensorProto.FLOAT16),
            ],
            [
                _value_info("X", TensorProto.FLOAT, [2, 3]),
                _value_info("Y", TensorProto.FLOAT, [2, 3]),
            ],
            [_value_info("Z", TensorProto.FLOAT16, [2, 3])],
        )
        feeds = {
            "X": np.arange(6, dtype=np.float32).reshape(2, 3),
            "Y": np.arange(6, dtype=np.float32).reshape(2, 3),
        }

        optimized, rewrites, _ = self._optimize(model, "CastOpCast")

        self.assertEqual(_op_types(optimized), ["Add", "Cast"])
        self.assertEqual(len(rewrites), 0)
        self._assert_equivalent(model, optimized, feeds)

    def test_cast_op_cast_preserves_used_computation_output(self):
        model = _model(
            [
                helper.make_node("Cast", ["Y"], ["yc"], to=TensorProto.FLOAT),
                helper.make_node("Add", ["X", "yc"], ["computed"]),
                helper.make_node("Cast", ["computed"], ["Z"], to=TensorProto.FLOAT16),
            ],
            [
                _value_info("X", TensorProto.FLOAT, [2, 3]),
                _value_info("Y", TensorProto.FLOAT16, [2, 3]),
            ],
            [
                _value_info("computed", TensorProto.FLOAT, [2, 3]),
                _value_info("Z", TensorProto.FLOAT16, [2, 3]),
            ],
        )
        feeds = {
            "X": np.arange(6, dtype=np.float32).reshape(2, 3),
            "Y": np.arange(6, dtype=np.float16).reshape(2, 3),
        }

        optimized, rewrites, _ = self._optimize(model, "CastOpCast")

        self.assertEqual(_op_types(optimized), ["Cast", "Add", "Cast"])
        self.assertEqual(len(rewrites), 1)
        self._assert_equivalent(model, optimized, feeds, atol=1e-3)

    def test_identity_reshape_all_zeros(self):
        for shape_values in ([0, 0, 0], [0], [0, 0]):
            with self.subTest(shape_values=shape_values):
                input_shape = list(range(2, 2 + len(shape_values)))
                model = _model(
                    [helper.make_node("Reshape", ["X", "shape"], ["Y"])],
                    [_value_info("X", TensorProto.FLOAT, input_shape)],
                    [_value_info("Y", TensorProto.FLOAT, input_shape)],
                    [_array(shape_values, np.int64, "shape")],
                )
                feeds = {
                    "X": np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape)
                }

                optimized, rewrites, _ = self._optimize(model, "Identity")

                self.assertEqual(_op_types(optimized), ["Identity"])
                self.assertEqual(len(rewrites), 1)
                self._assert_equivalent(model, optimized, feeds)

    def test_identity_transpose(self):
        model = _model(
            [
                helper.make_node("Add", ["X", "two"], ["x2"]),
                helper.make_node("Mul", ["x2", "two"], ["x3"]),
                helper.make_node("Transpose", ["x3"], ["Y"], perm=[0, 1, 2]),
            ],
            [_value_info("X", TensorProto.FLOAT, [2, 3, 4])],
            [_value_info("Y", TensorProto.FLOAT, [2, 3, 4])],
            [_array([2], np.float32, "two")],
        )
        feeds = {"X": np.arange(24, dtype=np.float32).reshape(2, 3, 4)}

        optimized, rewrites, _ = self._optimize(model, "Identity")

        self.assertEqual(_op_types(optimized), ["Add", "Mul", "Identity"])
        self.assertEqual(len(rewrites), 1)
        self._assert_equivalent(model, optimized, feeds)

    def test_identity_batch_normalization_float16(self):
        model = _model(
            [
                helper.make_node(
                    "BatchNormalization",
                    ["X", "scale", "bias", "bias", "scale"],
                    ["normalized", "saved_mean", "saved_variance"],
                ),
                helper.make_node("Neg", ["normalized"], ["Z"]),
            ],
            [_value_info("X", TensorProto.FLOAT16, [3, 2])],
            [_value_info("Z", TensorProto.FLOAT16, [3, 2])],
            [_array([0, 0], np.float32, "bias"), _array([1, 1], np.float32, "scale")],
            opset=20,
        )
        feeds = {"X": np.arange(6, dtype=np.float16).reshape(3, 2)}

        optimized, rewrites, _ = self._optimize(model, "Identity")

        self.assertEqual(_op_types(optimized), ["Neg"])
        self.assertEqual(len(rewrites), 1)
        got = ReferenceEvaluator(optimized).run(None, feeds)[0]
        np.testing.assert_array_equal(got, -feeds["X"])

    def test_identity_expand(self):
        model = _model(
            [
                helper.make_node("Add", ["X", "two"], ["x2"]),
                helper.make_node("Mul", ["x2", "two"], ["x3"]),
                helper.make_node("Expand", ["x3", "ones"], ["Y"]),
            ],
            [_value_info("X", TensorProto.FLOAT, [2, 3, 4])],
            [_value_info("Y", TensorProto.FLOAT, [2, 3, 4])],
            [_array([2], np.float32, "two"), _array([1, 1, 1], np.int64, "ones")],
        )
        feeds = {"X": np.arange(24, dtype=np.float32).reshape(2, 3, 4)}

        optimized, rewrites, _ = self._optimize(model, "Identity")

        self.assertEqual(_op_types(optimized), ["Add", "Mul", "Identity"])
        self.assertEqual(len(rewrites), 1)
        self._assert_equivalent(model, optimized, feeds)

    def test_identity_mul_constant_node(self):
        model = _model(
            [
                helper.make_node("Constant", [], ["one"], value_float=1.0),
                helper.make_node("Add", ["X", "two"], ["x2"]),
                helper.make_node("Mul", ["x2", "one"], ["Y"]),
            ],
            [_value_info("X", TensorProto.FLOAT, [2, 3, 4])],
            [_value_info("Y", TensorProto.FLOAT, [2, 3, 4])],
            [_array([2], np.float32, "two")],
        )
        feeds = {"X": np.arange(24, dtype=np.float32).reshape(2, 3, 4)}

        optimized, rewrites, _ = self._optimize(model, "Identity")

        self.assertEqual(_op_types(optimized), ["Add", "Identity"])
        self.assertEqual(len(rewrites), 1)
        self._assert_equivalent(model, optimized, feeds)

    def test_identity_add_mul_uniform_vectors(self):
        model = _model(
            [
                helper.make_node("Add", ["X", "zero"], ["x2"]),
                helper.make_node("Mul", ["x2", "one"], ["Y"]),
            ],
            [_value_info("X", TensorProto.FLOAT, [2, 3, 4])],
            [_value_info("Y", TensorProto.FLOAT, [2, 3, 4])],
            [_array([0, 0, 0, 0], np.float32, "zero"), _array([1, 1, 1, 1], np.float32, "one")],
        )
        feeds = {"X": np.arange(24, dtype=np.float32).reshape(2, 3, 4)}

        optimized, rewrites, _ = self._optimize(model, "Identity")

        self.assertEqual(_op_types(optimized), ["Identity"])
        self.assertEqual(len(rewrites), 2)
        self._assert_equivalent(model, optimized, feeds)

    def test_identity_add_constant_node(self):
        model = _model(
            [
                helper.make_node("Constant", [], ["zero"], value_float=0.0),
                helper.make_node("Add", ["X", "zero"], ["x2"]),
                helper.make_node("Mul", ["x2", "two"], ["Y"]),
            ],
            [_value_info("X", TensorProto.FLOAT, [2, 3, 4])],
            [_value_info("Y", TensorProto.FLOAT, [2, 3, 4])],
            [_array([2], np.float32, "two")],
        )
        feeds = {"X": np.arange(24, dtype=np.float32).reshape(2, 3, 4)}

        optimized, rewrites, _ = self._optimize(model, "Identity")

        self.assertEqual(_op_types(optimized), ["Mul"])
        self.assertEqual(len(rewrites), 1)
        self._assert_equivalent(model, optimized, feeds)

    def test_identity_no_match_non_identity_transpose(self):
        model = _model(
            [helper.make_node("Transpose", ["X"], ["Y"], perm=[1, 0])],
            [_value_info("X", TensorProto.FLOAT, [2, 3])],
            [_value_info("Y", TensorProto.FLOAT, [3, 2])],
        )
        feeds = {"X": np.arange(6, dtype=np.float32).reshape(2, 3)}

        optimized, rewrites, _ = self._optimize(model, "Identity")

        self.assertEqual(_op_types(optimized), ["Transpose"])
        self.assertEqual(len(rewrites), 0)
        self._assert_equivalent(model, optimized, feeds)

    def test_clip_clip(self):
        model = _model(
            [
                helper.make_node("Clip", ["X", "zero"], ["clipped_min"]),
                helper.make_node("Clip", ["clipped_min", "", "one"], ["Y"]),
            ],
            [_value_info("X", TensorProto.FLOAT, [2, 3])],
            [_value_info("Y", TensorProto.FLOAT, [2, 3])],
            [_array([0], np.float32, "zero"), _array([1], np.float32, "one")],
        )
        feeds = {"X": np.array([[-2, -0.5, 0.25], [0.75, 1.5, 3]], dtype=np.float32)}

        optimized, rewrites, _ = self._optimize(model, "ClipClip")

        self.assertEqual(_op_types(optimized), ["Clip"])
        self.assertEqual(list(optimized.graph.node[0].input), ["X", "zero", "one"])
        self.assertEqual(len(rewrites), 1)
        self.assertEqual(len(optimized.graph.initializer), 2)
        self._assert_equivalent(model, optimized, feeds)

    def test_clip_clip_no_match_when_first_output_is_used(self):
        model = _model(
            [
                helper.make_node("Clip", ["X", "zero"], ["clipped_min"]),
                helper.make_node("Clip", ["clipped_min", "", "one"], ["Y"]),
            ],
            [_value_info("X", TensorProto.FLOAT, [2, 3])],
            [
                _value_info("clipped_min", TensorProto.FLOAT, [2, 3]),
                _value_info("Y", TensorProto.FLOAT, [2, 3]),
            ],
            [_array([0], np.float32, "zero"), _array([1], np.float32, "one")],
        )
        feeds = {"X": np.array([[-2, -0.5, 0.25], [0.75, 1.5, 3]], dtype=np.float32)}

        optimized, rewrites, _ = self._optimize(model, "ClipClip")

        self.assertEqual(_op_types(optimized), ["Clip", "Clip"])
        self.assertEqual(len(rewrites), 0)
        self._assert_equivalent(model, optimized, feeds)

    def test_constant_to_initializer(self):
        model = _model(
            [
                helper.make_node(
                    "Constant",
                    [],
                    ["constant"],
                    value=numpy_helper.from_array(np.array([1, 2], dtype=np.float32)),
                ),
                helper.make_node("Add", ["X", "constant"], ["Y"]),
            ],
            [_value_info("X", TensorProto.FLOAT, [5, 2])],
            [_value_info("Y", TensorProto.FLOAT, [5, 2])],
        )
        feeds = {"X": np.arange(10, dtype=np.float32).reshape(5, 2)}

        optimized, rewrites, _ = self._optimize(model, "ConstantToInitializer")

        self.assertEqual(_op_types(optimized), ["Add"])
        self.assertGreaterEqual(len(rewrites), 1)
        self.assertEqual(len(optimized.graph.initializer), 1)
        initializer = optimized.graph.initializer[0]
        self.assertEqual(initializer.name, "constant_cst2init")
        np.testing.assert_array_equal(
            numpy_helper.to_array(initializer), np.array([1, 2], dtype=np.float32)
        )
        self._assert_equivalent(model, optimized, feeds)

    def test_constant_to_initializer_recursive_if(self):
        then_graph = helper.make_graph(
            [
                helper.make_node(
                    "Constant",
                    [],
                    ["then_output"],
                    value=numpy_helper.from_array(np.ones(5, dtype=np.float32)),
                )
            ],
            "then_branch",
            [],
            [_value_info("then_output", TensorProto.FLOAT, [5])],
        )
        else_graph = helper.make_graph(
            [
                helper.make_node(
                    "Constant",
                    [],
                    ["else_output"],
                    value=numpy_helper.from_array(-np.ones(5, dtype=np.float32)),
                )
            ],
            "else_branch",
            [],
            [_value_info("else_output", TensorProto.FLOAT, [5])],
        )
        model = _model(
            [
                helper.make_node("ReduceSum", ["X"], ["sum"]),
                helper.make_node("Greater", ["sum", "zero"], ["condition"]),
                helper.make_node(
                    "If", ["condition"], ["Y"], then_branch=then_graph, else_branch=else_graph
                ),
            ],
            [_value_info("X", TensorProto.FLOAT, [3, 2])],
            [_value_info("Y", TensorProto.FLOAT, [5])],
            [_array(0, np.float32, "zero")],
        )

        optimized, rewrites, report = self._optimize(model, "ConstantToInitializer")

        self.assertNotIn("Constant", _recursive_op_types(optimized.graph))
        self.assertGreaterEqual(len(rewrites), 2)
        self.assertTrue(report.subgraphs)
        then_optimized, else_optimized = _if_branches(optimized)
        self.assertEqual(len(then_optimized.initializer), 2)
        self.assertEqual(len(else_optimized.initializer), 2)
        for feeds in (
            {"X": np.ones((3, 2), dtype=np.float32)},
            {"X": -np.ones((3, 2), dtype=np.float32)},
        ):
            self._assert_equivalent(model, optimized, feeds)

    def test_constant_to_initializer_if_capture_conflicting_values(self):
        model = _constant_capture_model(2.1, 2.2)

        optimized, rewrites, _ = self._optimize(model, "ConstantToInitializer")

        self.assertNotIn("Constant", _recursive_op_types(optimized.graph))
        self.assertGreaterEqual(len(rewrites), 2)
        self.assertEqual(
            {initializer.name for initializer in optimized.graph.initializer},
            {"threshold", "outer_two"},
        )
        then_graph, else_graph = _if_branches(optimized)
        self.assertEqual(
            {initializer.name for initializer in then_graph.initializer},
            {"two_cst2init", "two_cst2init_0"},
        )
        self.assertEqual(
            {initializer.name for initializer in else_graph.initializer},
            {"two_cst2init", "two_cst2init_0"},
        )
        for feeds in (
            {"X": np.array([1, 2, 3], dtype=np.float32)},
            {"X": np.array([-1, -2, -3], dtype=np.float32)},
        ):
            self._assert_equivalent(model, optimized, feeds)

    def test_constant_to_initializer_if_capture_same_values(self):
        model = _constant_capture_model(2.0, 2.0)

        optimized, rewrites, _ = self._optimize(model, "ConstantToInitializer")

        self.assertNotIn("Constant", _recursive_op_types(optimized.graph))
        self.assertGreaterEqual(len(rewrites), 2)
        then_graph, else_graph = _if_branches(optimized)
        self.assertEqual(len(then_graph.initializer), 2)
        self.assertEqual(len(else_graph.initializer), 2)
        for feeds in (
            {"X": np.array([1, 2, 3], dtype=np.float32)},
            {"X": np.array([-1, -2, -3], dtype=np.float32)},
        ):
            self._assert_equivalent(model, optimized, feeds)

    def test_constant_to_initializer_graph_output(self):
        model = _model(
            [
                helper.make_node(
                    "Constant",
                    [],
                    ["Y"],
                    value=numpy_helper.from_array(np.array([3, 4], dtype=np.int64)),
                )
            ],
            [],
            [_value_info("Y", TensorProto.INT64, [2])],
        )

        optimized, rewrites, _ = self._optimize(model, "ConstantToInitializer")

        self.assertNotIn("Constant", _op_types(optimized))
        self.assertGreaterEqual(len(rewrites), 1)
        self.assertEqual(
            {initializer.name for initializer in optimized.graph.initializer}, {"Y", "Y_cst2init"}
        )
        self._assert_equivalent(model, optimized, {})

    def test_dropout_minimal_layernorm_artifact_replacement(self):
        model = _layernorm_model(
            [
                helper.make_node("Dropout", ["normalized", "ratio"], ["dropped", "mask"]),
                helper.make_node("Add", ["dropped", "residual"], ["Y"]),
            ],
            "Y",
            [_array(0.1, np.float32, "ratio"), _array(np.ones((1, 4)), np.float32, "residual")],
            opset=18,
        )
        feeds = {"X": np.arange(8, dtype=np.float32).reshape(2, 4)}

        optimized, rewrites, _ = self._optimize(model, "Dropout")

        self.assertEqual(_op_types(optimized), ["LayerNormalization", "Add"])
        self.assertGreaterEqual(len(rewrites), 1)
        self.assertEqual(len(optimized.graph.initializer), 4)
        self._assert_equivalent(model, optimized, feeds, atol=1e-6)

    def test_dropout_no_match_when_mask_is_used(self):
        model = _model(
            [
                helper.make_node("Dropout", ["X", "ratio"], ["Y", "mask"]),
                helper.make_node("Cast", ["mask"], ["mask_float"], to=TensorProto.FLOAT),
                helper.make_node("Mul", ["Y", "mask_float"], ["Z"]),
            ],
            [_value_info("X", TensorProto.FLOAT, [2, 3])],
            [_value_info("Z", TensorProto.FLOAT, [2, 3])],
            [_array(0.5, np.float32, "ratio")],
            opset=13,
        )
        feeds = {"X": np.arange(6, dtype=np.float32).reshape(2, 3)}

        optimized, rewrites, _ = self._optimize(model, "Dropout")

        self.assertEqual(_op_types(optimized), ["Dropout", "Cast", "Mul"])
        self.assertEqual(len(rewrites), 0)
        self._assert_equivalent(model, optimized, feeds)

    def test_dropout_no_match_when_training_is_enabled(self):
        model = _model(
            [helper.make_node("Dropout", ["X", "ratio", "training"], ["Y", "mask"])],
            [_value_info("X", TensorProto.FLOAT, [2, 3])],
            [
                _value_info("Y", TensorProto.FLOAT, [2, 3]),
                _value_info("mask", TensorProto.BOOL, [2, 3]),
            ],
            [_array(0.5, np.float32, "ratio"), _array(True, np.bool_, "training")],
            opset=13,
        )

        optimized, rewrites, _ = self._optimize(model, "Dropout")

        self.assertEqual(_op_types(optimized), ["Dropout"])
        self.assertEqual(len(rewrites), 0)

    def test_conv_bias_null(self):
        model = _model(
            [
                helper.make_node(
                    "Conv", ["X", "W", "bias"], ["Y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]
                )
            ],
            [
                _value_info("X", TensorProto.FLOAT, [1, 1, 5, 5]),
                _value_info("W", TensorProto.FLOAT, [2, 1, 3, 3]),
            ],
            [_value_info("Y", TensorProto.FLOAT, [1, 2, 5, 5])],
            [_array(np.zeros(2), np.float32, "bias")],
        )
        feeds = {
            "X": np.arange(25, dtype=np.float32).reshape(1, 1, 5, 5) / 10,
            "W": np.arange(18, dtype=np.float32).reshape(2, 1, 3, 3) / 20,
        }

        optimized, rewrites, _ = self._optimize(model, "ConvBiasNull")

        self.assertEqual(_op_types(optimized), ["Conv"])
        self.assertEqual(list(optimized.graph.node[0].input), ["X", "W"])
        self.assertEqual(len(rewrites), 1)
        self.assertEqual(
            {initializer.name for initializer in optimized.graph.initializer}, {"bias"}
        )
        self._assert_equivalent(model, optimized, feeds, atol=1e-6)

    def test_conv_bias_null_materialized_constant(self):
        model = _model(
            [
                helper.make_node(
                    "Constant",
                    [],
                    ["bias"],
                    value=numpy_helper.from_array(np.zeros(2, dtype=np.float32)),
                ),
                helper.make_node(
                    "Conv", ["X", "W", "bias"], ["Y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]
                ),
            ],
            [
                _value_info("X", TensorProto.FLOAT, [1, 1, 5, 5]),
                _value_info("W", TensorProto.FLOAT, [2, 1, 3, 3]),
            ],
            [_value_info("Y", TensorProto.FLOAT, [1, 2, 5, 5])],
        )
        feeds = {
            "X": np.arange(25, dtype=np.float32).reshape(1, 1, 5, 5) / 10,
            "W": np.arange(18, dtype=np.float32).reshape(2, 1, 3, 3) / 20,
        }

        optimized, rewrites, _ = self._optimize(model, "ConvBiasNull")

        self.assertEqual(_op_types(optimized), ["Conv"])
        self.assertEqual(list(optimized.graph.node[-1].input), ["X", "W"])
        self.assertEqual(len(rewrites), 1)
        self._assert_equivalent(model, optimized, feeds, atol=1e-6)

    def test_conv_bias_null_shape_expand(self):
        model = _model(
            [
                helper.make_node("Shape", ["bias_shape_source"], ["bias_shape"], start=0, end=1),
                helper.make_node("Expand", ["zero", "bias_shape"], ["bias"]),
                helper.make_node(
                    "Conv", ["X", "W", "bias"], ["Y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]
                ),
            ],
            [
                _value_info("X", TensorProto.FLOAT, [1, 1, 5, 5]),
                _value_info("W", TensorProto.FLOAT, [2, 1, 3, 3]),
            ],
            [_value_info("Y", TensorProto.FLOAT, [1, 2, 5, 5])],
            [
                _array(np.zeros(2), np.float32, "bias_shape_source"),
                _array(np.zeros(1), np.float32, "zero"),
            ],
        )
        feeds = {
            "X": np.arange(25, dtype=np.float32).reshape(1, 1, 5, 5) / 10,
            "W": np.arange(18, dtype=np.float32).reshape(2, 1, 3, 3) / 20,
        }

        optimized, rewrites, _ = self._optimize(model, "ConvBiasNull")

        self.assertEqual(_op_types(optimized), ["Conv"])
        self.assertEqual(list(optimized.graph.node[0].input), ["X", "W"])
        self.assertEqual(len(rewrites), 1)
        self._assert_equivalent(model, optimized, feeds, atol=1e-6)

    def test_conv_bias_null_no_match_nonzero_bias(self):
        model = _model(
            [
                helper.make_node(
                    "Conv", ["X", "W", "bias"], ["Y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]
                )
            ],
            [
                _value_info("X", TensorProto.FLOAT, [1, 1, 5, 5]),
                _value_info("W", TensorProto.FLOAT, [2, 1, 3, 3]),
            ],
            [_value_info("Y", TensorProto.FLOAT, [1, 2, 5, 5])],
            [_array([0, 1], np.float32, "bias")],
        )
        feeds = {
            "X": np.arange(25, dtype=np.float32).reshape(1, 1, 5, 5) / 10,
            "W": np.arange(18, dtype=np.float32).reshape(2, 1, 3, 3) / 20,
        }

        optimized, rewrites, _ = self._optimize(model, "ConvBiasNull")

        self.assertEqual(list(optimized.graph.node[0].input), ["X", "W", "bias"])
        self.assertEqual(len(rewrites), 0)
        self._assert_equivalent(model, optimized, feeds, atol=1e-6)

    def test_pad_conv_fusion(self):
        model = _model(
            [
                helper.make_node("Pad", ["X", "pads"], ["padded"]),
                helper.make_node("Conv", ["padded", "W"], ["Y"], kernel_shape=[3, 3]),
            ],
            [
                _value_info("X", TensorProto.FLOAT, [1, 1, 6, 6]),
                _value_info("W", TensorProto.FLOAT, [2, 1, 3, 3]),
            ],
            [_value_info("Y", TensorProto.FLOAT, [1, 2, 6, 6])],
            [_array([0, 0, 1, 1, 0, 0, 1, 1], np.int64, "pads")],
        )
        feeds = {
            "X": np.arange(36, dtype=np.float32).reshape(1, 1, 6, 6) / 10,
            "W": np.arange(18, dtype=np.float32).reshape(2, 1, 3, 3) / 20,
        }

        optimized, rewrites, _ = self._optimize(model, "PadConv")

        self.assertEqual(_op_types(optimized), ["Conv"])
        self.assertEqual(list(optimized.graph.node[0].input), ["X", "W"])
        self.assertEqual(_node_attribute_ints(optimized.graph.node[0], "pads"), [1, 1, 1, 1])
        self.assertEqual(len(rewrites), 1)
        self._assert_equivalent(model, optimized, feeds, atol=1e-6)

    def test_pad_conv_fusion_with_existing_pads(self):
        model = _model(
            [
                helper.make_node("Pad", ["X", "pads"], ["padded"]),
                helper.make_node(
                    "Conv", ["padded", "W"], ["Y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]
                ),
            ],
            [
                _value_info("X", TensorProto.FLOAT, [1, 1, 6, 6]),
                _value_info("W", TensorProto.FLOAT, [2, 1, 3, 3]),
            ],
            [_value_info("Y", TensorProto.FLOAT, [1, 2, 8, 8])],
            [_array([0, 0, 1, 1, 0, 0, 1, 1], np.int64, "pads")],
        )
        feeds = {
            "X": np.arange(36, dtype=np.float32).reshape(1, 1, 6, 6) / 10,
            "W": np.arange(18, dtype=np.float32).reshape(2, 1, 3, 3) / 20,
        }

        optimized, rewrites, _ = self._optimize(model, "PadConv")

        self.assertEqual(_op_types(optimized), ["Conv"])
        self.assertEqual(_node_attribute_ints(optimized.graph.node[0], "pads"), [2, 2, 2, 2])
        self.assertEqual(len(rewrites), 1)
        self._assert_equivalent(model, optimized, feeds, atol=1e-6)

    def test_pad_conv_no_fusion_batch_channel_padded(self):
        model = _model(
            [
                helper.make_node("Pad", ["X", "pads"], ["padded"]),
                helper.make_node("Conv", ["padded", "W"], ["Y"], kernel_shape=[1, 1]),
            ],
            [
                _value_info("X", TensorProto.FLOAT, [1, 2, 4, 4]),
                _value_info("W", TensorProto.FLOAT, [2, 2, 1, 1]),
            ],
            [_value_info("Y", TensorProto.FLOAT, None)],
            [_array([1, 0, 0, 0, 1, 0, 0, 0], np.int64, "pads")],
        )
        feeds = {
            "X": np.arange(32, dtype=np.float32).reshape(1, 2, 4, 4) / 10,
            "W": np.arange(4, dtype=np.float32).reshape(2, 2, 1, 1) / 10,
        }

        optimized, rewrites, _ = self._optimize(model, "PadConv")

        self.assertEqual(_op_types(optimized), ["Pad", "Conv"])
        self.assertEqual(len(rewrites), 0)
        self._assert_equivalent(model, optimized, feeds)


if __name__ == "__main__":
    unittest.main(verbosity=2)
