# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests collection-oriented graph optimization patterns."""

from __future__ import annotations

import unittest

import numpy

import onnx_light.onnx.helper as oh

import onnx_light.onnx.numpy_helper as onh

from onnx_light.onnx import TensorProto
from onnx_light.ext_test_case import import_or_skip

from onnx_light.onnx_core import optimization

ReferenceEvaluator = import_or_skip("onnx_light.onnx.reference", "ReferenceEvaluator")


def make_model(nodes, inputs, outputs, initializers=()):
    """Builds an opset-18 model."""
    graph = oh.make_graph(nodes, "test", inputs, outputs, list(initializers))
    return oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)], ir_version=10)


def make_value_info(name, data_type, shape):
    """Builds tensor value information."""
    return oh.make_tensor_value_info(name, data_type, shape)


def make_initializer(name, value, dtype=numpy.int64):
    """Builds an initializer from a NumPy value."""
    return onh.from_array(numpy.asarray(value, dtype=dtype), name=name)


def make_range(*shape, dtype=numpy.float32):
    """Builds a deterministic tensor with the requested shape."""
    return numpy.arange(numpy.prod(shape), dtype=dtype).reshape(shape)


def node_types(model):
    """Returns the model node types in topological order."""
    return [node.op_type for node in model.graph.node]


def initializer_values(model):
    """Returns every initializer as a NumPy array."""
    return {
        initializer.name: onh.to_array(initializer) for initializer in model.graph.initializer
    }


class TestCollectionPatterns(unittest.TestCase):
    """Tests collection, slicing, gathering, splitting, and sequence patterns."""

    def optimize(self, model, pattern_name):
        """Optimizes a model with one isolated GraphGraph pattern."""
        builder = optimization.GraphBuilder(model)
        graph = optimization.GraphGraph(builder, [pattern_name], use_global_patterns=False)
        rewrites = graph.optimize()
        return builder.to_onnx("model"), rewrites

    def assert_equivalent(self, original, optimized, feeds):
        """Checks numerical equivalence with the onnx-light evaluator."""
        expected = ReferenceEvaluator(original).run(None, feeds)
        got = ReferenceEvaluator(optimized).run(None, feeds)
        self.assertEqual(len(expected), len(got))
        for expected_value, got_value in zip(expected, got):
            self.assertEqual(expected_value.dtype, got_value.dtype)
            self.assertEqual(expected_value.shape, got_value.shape)
            numpy.testing.assert_allclose(expected_value, got_value, rtol=1e-6, atol=1e-6)

    def test_slices_split(self):
        model = make_model(
            [
                oh.make_node("Slice", ["X", "zero", "seven", "one"], ["x1"]),
                oh.make_node("Slice", ["X", "seven", "eight", "one"], ["x2"]),
                oh.make_node("Add", ["x1", "x2"], ["Y"]),
            ],
            [make_value_info("X", TensorProto.FLOAT, ["a", 8])],
            [make_value_info("Y", TensorProto.FLOAT, ["a", 7])],
            [
                make_initializer("zero", [0]),
                make_initializer("one", [1]),
                make_initializer("seven", [7]),
                make_initializer("eight", [8]),
            ],
        )
        feeds = {"X": make_range(11, 8)}

        optimized, rewrites = self.optimize(model, "SlicesSplit")

        self.assertEqual(len(rewrites), 1)
        self.assertEqual(node_types(optimized), ["Split", "Add"])
        split = optimized.graph.node[0]
        self.assertEqual(list(split.output), ["x1", "x2"])
        values = initializer_values(optimized)
        self.assertIn(split.input[1], values)
        numpy.testing.assert_array_equal(values[split.input[1]], numpy.array([7, 1]))
        self.assert_equivalent(model, optimized, feeds)

    def test_gathers_split_rank1(self):
        model = make_model(
            [
                oh.make_node("Gather", ["X", "zero"], ["x1"], axis=1),
                oh.make_node("Gather", ["X", "one"], ["x2"], axis=1),
                oh.make_node("Add", ["x1", "x2"], ["Y"]),
            ],
            [make_value_info("X", TensorProto.FLOAT, ["a", 2])],
            [make_value_info("Y", TensorProto.FLOAT, ["a", 1])],
            [make_initializer("zero", [0]), make_initializer("one", [1])],
        )
        feeds = {"X": make_range(11, 2)}

        optimized, rewrites = self.optimize(model, "GathersSplit")

        self.assertEqual(len(rewrites), 1)
        self.assertEqual(node_types(optimized), ["Split", "Add"])
        self.assertEqual(list(optimized.graph.node[0].input), ["X"])
        self.assertEqual(list(optimized.graph.node[0].output), ["x1", "x2"])
        self.assert_equivalent(model, optimized, feeds)

    def test_gathers_split_rank0(self):
        model = make_model(
            [
                oh.make_node("Gather", ["X", "zero"], ["x1"], axis=1),
                oh.make_node("Gather", ["X", "one"], ["x2"], axis=1),
                oh.make_node("Add", ["x1", "x2"], ["Y"]),
            ],
            [make_value_info("X", TensorProto.FLOAT, ["a", 2])],
            [make_value_info("Y", TensorProto.FLOAT, ["a"])],
            [make_initializer("zero", 0), make_initializer("one", 1)],
        )
        feeds = {"X": make_range(11, 2)}

        optimized, rewrites = self.optimize(model, "GathersSplit")

        self.assertEqual(len(rewrites), 1)
        self.assertEqual(node_types(optimized), ["Split", "Squeeze", "Squeeze", "Add"])
        values = initializer_values(optimized)
        squeeze_axes = {
            tuple(values[node.input[1]].tolist())
            for node in optimized.graph.node
            if node.op_type == "Squeeze"
        }
        self.assertEqual(squeeze_axes, {(1,)})
        self.assert_equivalent(model, optimized, feeds)

    def make_slice_slice_model(self, first_step, second_step):
        """Builds two Slice nodes operating on distinct axes."""
        first_inputs = ["X", "zero", "one", "zero"]
        second_inputs = ["x1", "zero", "one", "one"]
        if first_step:
            first_inputs.append("one")
        if second_step:
            second_inputs.append("one")
        return make_model(
            [
                oh.make_node("Slice", first_inputs, ["x1"]),
                oh.make_node("Slice", second_inputs, ["Y"]),
            ],
            [make_value_info("X", TensorProto.FLOAT, ["a", "b"])],
            [make_value_info("Y", TensorProto.FLOAT, ["c", "d"])],
            [make_initializer("zero", [0]), make_initializer("one", [1])],
        )

    def check_slice_slice(self, first_step, second_step):
        """Checks one combination of optional Slice steps."""
        model = self.make_slice_slice_model(first_step, second_step)
        feeds = {"X": make_range(2, 3)}

        optimized, rewrites = self.optimize(model, "SliceSlice")

        self.assertEqual(sum(rewrite.pattern_name == "SliceSlice" for rewrite in rewrites), 1)
        self.assertEqual(node_types(optimized), ["Slice"])
        slice_node = optimized.graph.node[0]
        self.assertEqual(slice_node.input[0], "X")
        self.assertEqual(len(slice_node.input), 5 if first_step or second_step else 4)
        values = initializer_values(optimized)
        numpy.testing.assert_array_equal(values[slice_node.input[1]], numpy.array([0, 0]))
        numpy.testing.assert_array_equal(values[slice_node.input[2]], numpy.array([1, 1]))
        numpy.testing.assert_array_equal(values[slice_node.input[3]], numpy.array([0, 1]))
        if first_step or second_step:
            numpy.testing.assert_array_equal(values[slice_node.input[4]], numpy.array([1, 1]))
        if first_step != second_step:
            generated_ones = [
                value
                for name, value in values.items()
                if name not in {"zero", "one"}
                and value.shape == (1,)
                and numpy.array_equal(value, numpy.array([1]))
            ]
            self.assertEqual(len(generated_ones), 1)
            numpy.testing.assert_array_equal(generated_ones[0], numpy.array([1]))
        self.assert_equivalent(model, optimized, feeds)

    def test_slice_slice_nostep(self):
        self.check_slice_slice(False, False)

    def test_slice_slice_steps1(self):
        self.check_slice_slice(True, False)

    def test_slice_slice_steps2(self):
        self.check_slice_slice(False, True)

    def test_slice_slice_steps3(self):
        self.check_slice_slice(True, True)

    def test_sequence_construct(self):
        model = make_model(
            [
                oh.make_node("SequenceConstruct", ["X1", "X2"], ["seq"]),
                oh.make_node("SequenceAt", ["seq", "i0"], ["Y1"]),
                oh.make_node("SequenceAt", ["seq", "i1"], ["Y2"]),
            ],
            [
                make_value_info("X1", TensorProto.FLOAT, ["a", "b"]),
                make_value_info("X2", TensorProto.FLOAT, ["c", "d"]),
            ],
            [
                make_value_info("Y1", TensorProto.FLOAT, ["a", "b"]),
                make_value_info("Y2", TensorProto.FLOAT, ["c", "d"]),
            ],
            [make_initializer("i0", 0), make_initializer("i1", 1)],
        )
        feeds = {"X1": make_range(2, 3), "X2": make_range(3, 4)}

        optimized, rewrites = self.optimize(model, "SequenceConstructAt")

        self.assertEqual(len(rewrites), 1)
        self.assertEqual(node_types(optimized), ["Identity", "Identity"])
        self.assertEqual([list(node.input) for node in optimized.graph.node], [["X1"], ["X2"]])
        self.assert_equivalent(model, optimized, feeds)

    def test_split_to_sequence_sequence_at(self):
        model = make_model(
            [
                oh.make_node("SplitToSequence", ["X", "split"], ["seq"], axis=1),
                oh.make_node("SequenceAt", ["seq", "i0"], ["Y1"]),
                oh.make_node("SequenceAt", ["seq", "i1"], ["Y2"]),
                oh.make_node("SequenceAt", ["seq", "i2"], ["Y3"]),
            ],
            [make_value_info("X", TensorProto.FLOAT, ["a", 6])],
            [
                make_value_info("Y1", TensorProto.FLOAT, ["a", 2]),
                make_value_info("Y2", TensorProto.FLOAT, ["a", 2]),
                make_value_info("Y3", TensorProto.FLOAT, ["a", 2]),
            ],
            [
                make_initializer("split", [2, 2, 2]),
                make_initializer("i0", 0),
                make_initializer("i1", 1),
                make_initializer("i2", 2),
            ],
        )
        feeds = {"X": make_range(3, 6)}

        optimized, rewrites = self.optimize(model, "SplitToSequenceSequenceAt")

        self.assertEqual(len(rewrites), 1)
        self.assertEqual(node_types(optimized), ["Split"])
        self.assertEqual(list(optimized.graph.node[0].output), ["Y1", "Y2", "Y3"])
        self.assert_equivalent(model, optimized, feeds)

    def test_split_concat(self):
        model = make_model(
            [
                oh.make_node("Split", ["X"], ["s1", "s2"], num_outputs=2, axis=-1),
                oh.make_node("Concat", ["s1", "s2"], ["Y"], axis=-1),
            ],
            [make_value_info("X", TensorProto.FLOAT, ["a", "b"])],
            [make_value_info("Y", TensorProto.FLOAT, ["a", "b"])],
        )
        feeds = {"X": make_range(2, 3)}

        optimized, rewrites = self.optimize(model, "SplitConcat")

        self.assertEqual(len(rewrites), 1)
        self.assertEqual(node_types(optimized), ["Identity"])
        self.assertEqual(list(optimized.graph.node[0].input), ["X"])
        self.assert_equivalent(model, optimized, feeds)

    def make_concat_gather_model(self, first_shape, second_shape, index):
        """Builds a Concat followed by a single-index Gather."""
        return make_model(
            [
                oh.make_node("Concat", ["D1", "D2"], ["d"], axis=0),
                oh.make_node("Gather", ["d", "index"], ["Y"]),
            ],
            [
                make_value_info("D1", TensorProto.INT64, first_shape),
                make_value_info("D2", TensorProto.INT64, second_shape),
            ],
            [make_value_info("Y", TensorProto.INT64, [1])],
            [make_initializer("index", [index])],
        )

    def test_concat_gather(self):
        model = self.make_concat_gather_model([1], [1], 1)
        feeds = {
            "D1": numpy.array([5], dtype=numpy.int64),
            "D2": numpy.array([7], dtype=numpy.int64),
        }

        optimized, rewrites = self.optimize(model, "ConcatGather")

        self.assertEqual(len(rewrites), 1)
        self.assertEqual(node_types(optimized), ["Identity"])
        self.assertEqual(list(optimized.graph.node[0].input), ["D2"])
        self.assert_equivalent(model, optimized, feeds)

    def test_concat_gather_multi_element_identity(self):
        model = self.make_concat_gather_model([2], [1], 2)
        feeds = {
            "D1": numpy.array([5, 6], dtype=numpy.int64),
            "D2": numpy.array([7], dtype=numpy.int64),
        }

        optimized, rewrites = self.optimize(model, "ConcatGather")

        self.assertEqual(len(rewrites), 1)
        self.assertEqual(node_types(optimized), ["Identity"])
        self.assertEqual(list(optimized.graph.node[0].input), ["D2"])
        self.assert_equivalent(model, optimized, feeds)

    def test_concat_gather_multi_element_gather(self):
        model = self.make_concat_gather_model([2], [1], 1)
        feeds = {
            "D1": numpy.array([10, 20], dtype=numpy.int64),
            "D2": numpy.array([30], dtype=numpy.int64),
        }

        optimized, rewrites = self.optimize(model, "ConcatGather")

        self.assertEqual(len(rewrites), 1)
        self.assertEqual(node_types(optimized), ["Gather"])
        gather = optimized.graph.node[0]
        self.assertEqual(gather.input[0], "D1")
        values = initializer_values(optimized)
        numpy.testing.assert_array_equal(values[gather.input[1]], numpy.array([1]))
        self.assert_equivalent(model, optimized, feeds)

    def make_concat_reshape_model(self, use_abs):
        """Builds a reshape shape assembled from constants and dynamic dimensions."""
        nodes = [
            oh.make_node("Shape", ["X"], ["D2"], start=2, end=3),
            oh.make_node("Shape", ["X"], ["D1"], start=3, end=4),
        ]
        shape_input = "D1"
        if use_abs:
            nodes.append(oh.make_node("Abs", ["D1"], ["aD1"]))
            shape_input = "aD1"
        nodes.extend(
            [
                oh.make_node("Concat", ["I1", "I2", shape_input, "D2"], ["shape"], axis=0),
                oh.make_node("Reshape", ["X", "shape"], ["Y"]),
            ]
        )
        return make_model(
            nodes,
            [make_value_info("X", TensorProto.FLOAT, ["a", "b", "c", "d"])],
            [make_value_info("Y", TensorProto.FLOAT, ["a", "b", "d", "c"])],
            [make_initializer("I1", [2]), make_initializer("I2", [1])],
        )

    def check_concat_reshape(self, use_abs):
        """Checks ConcatReshape with Shape and arbitrary dynamic producers."""
        model = self.make_concat_reshape_model(use_abs)
        feeds = {"X": make_range(2, 1, 3, 5)}

        optimized, rewrites = self.optimize(model, "ConcatReshape")

        self.assertEqual(sum(rewrite.pattern_name == "ConcatReshape" for rewrite in rewrites), 1)
        self.assertEqual(node_types(optimized), ["Shape", "Concat", "Reshape"])
        concat = optimized.graph.node[1]
        values = initializer_values(optimized)
        replacement_names = [
            name
            for name in concat.input
            if name in values and numpy.array_equal(values[name], numpy.array([-1]))
        ]
        self.assertEqual(len(replacement_names), 1)
        self.assert_equivalent(model, optimized, feeds)

    def test_concat_reshape(self):
        self.check_concat_reshape(False)

    def test_concat_reshape_any(self):
        self.check_concat_reshape(True)

    def test_concat_empty(self):
        model = make_model(
            [oh.make_node("Concat", ["X", "Y", "empty"], ["Z"], axis=0)],
            [
                make_value_info("X", TensorProto.INT64, ["a"]),
                make_value_info("Y", TensorProto.INT64, ["b"]),
            ],
            [make_value_info("Z", TensorProto.INT64, ["c"])],
            [make_initializer("empty", [])],
        )
        feeds = {"X": numpy.arange(2, dtype=numpy.int64), "Y": numpy.arange(2, dtype=numpy.int64)}

        optimized, rewrites = self.optimize(model, "ConcatEmpty")

        self.assertEqual(len(rewrites), 1)
        self.assertEqual(node_types(optimized), ["Concat"])
        self.assertEqual(list(optimized.graph.node[0].input), ["X", "Y"])
        self.assertNotIn("empty", optimized.graph.node[0].input)
        self.assert_equivalent(model, optimized, feeds)

    def test_concat_twice(self):
        model = make_model(
            [
                oh.make_node("Concat", ["X", "X"], ["xx"], axis=0),
                oh.make_node("Sin", ["xx"], ["xsin"]),
                oh.make_node("Cos", ["xsin"], ["xsc"]),
                oh.make_node("Cast", ["xsc"], ["Y"], to=TensorProto.FLOAT),
            ],
            [make_value_info("X", TensorProto.FLOAT, ["b", "c"])],
            [make_value_info("Y", TensorProto.FLOAT, ["d", "c"])],
        )
        feeds = {"X": (make_range(3, 4) % 3).astype(numpy.float32)}

        optimized, rewrites = self.optimize(model, "ConcatTwiceUnary")

        self.assertEqual(len(rewrites), 3)
        self.assertEqual(node_types(optimized), ["Sin", "Cos", "Cast", "Concat"])
        self.assertEqual(
            list(optimized.graph.node[-1].input), [optimized.graph.node[-2].output[0]] * 2
        )
        self.assert_equivalent(model, optimized, feeds)

    def make_gather_shape_model(
        self, indices, output_shape, *, shape_start=None, shape_end=None, shared=False
    ):
        """Builds Gather(Shape(X), indices) with optional Shape bounds."""
        attributes = {}
        if shape_start is not None:
            attributes["start"] = shape_start
        if shape_end is not None:
            attributes["end"] = shape_end
        nodes = [
            oh.make_node("Shape", ["X"], ["shape"], **attributes),
            oh.make_node("Gather", ["shape", "indices"], ["Y"]),
        ]
        outputs = [make_value_info("Y", TensorProto.INT64, output_shape)]
        if shared:
            nodes.append(oh.make_node("Identity", ["shape"], ["shape_copy"]))
            outputs.append(make_value_info("shape_copy", TensorProto.INT64, [4]))
        return make_model(
            nodes,
            [make_value_info("X", TensorProto.FLOAT, ["a", "b", "c", "d"])],
            outputs,
            [make_initializer("indices", indices)],
        )

    def test_gather_shape_basic(self):
        for gather_start, gather_length in [(0, 3), (1, 2), (0, 4), (2, 2)]:
            with self.subTest(gather_start=gather_start, gather_length=gather_length):
                indices = numpy.arange(
                    gather_start, gather_start + gather_length, dtype=numpy.int64
                )
                model = self.make_gather_shape_model(indices, [gather_length])
                feeds = {"X": make_range(2, 3, 5, 7)}

                optimized, rewrites = self.optimize(model, "GatherShape")

                self.assertEqual(len(rewrites), 1)
                self.assertEqual(node_types(optimized), ["Shape"])
                shape = optimized.graph.node[0]
                attributes = {
                    attribute.name: oh.get_attribute_value(attribute)
                    for attribute in shape.attribute
                }
                self.assertEqual(attributes["start"], gather_start)
                self.assertEqual(attributes["end"], gather_start + gather_length)
                self.assert_equivalent(model, optimized, feeds)

    def test_gather_shape_with_shape_start(self):
        model = self.make_gather_shape_model([0, 1], [2], shape_start=2)
        feeds = {"X": make_range(2, 3, 5, 7)}

        optimized, rewrites = self.optimize(model, "GatherShape")

        self.assertEqual(len(rewrites), 1)
        self.assertEqual(node_types(optimized), ["Shape"])
        attributes = {
            attribute.name: oh.get_attribute_value(attribute)
            for attribute in optimized.graph.node[0].attribute
        }
        self.assertEqual(attributes, {"start": 2, "end": 4})
        self.assert_equivalent(model, optimized, feeds)

    def test_gather_shape_multi_consumer(self):
        model = self.make_gather_shape_model([0, 1], [2], shared=True)
        feeds = {"X": make_range(2, 3, 5, 7)}

        optimized, rewrites = self.optimize(model, "GatherShape")

        self.assertEqual(len(rewrites), 1)
        self.assertEqual(node_types(optimized), ["Shape", "Shape", "Identity"])
        self.assertEqual(sum(node.op_type == "Shape" for node in optimized.graph.node), 2)
        self.assert_equivalent(model, optimized, feeds)

    def test_gather_shape_single_element(self):
        model = self.make_gather_shape_model([2], [1])
        feeds = {"X": make_range(2, 3, 5, 7)}

        optimized, rewrites = self.optimize(model, "GatherShape")

        self.assertEqual(len(rewrites), 1)
        self.assertEqual(node_types(optimized), ["Shape"])
        self.assert_equivalent(model, optimized, feeds)

    def test_gather_shape_non_contiguous_no_match(self):
        model = self.make_gather_shape_model([0, 2], [2])
        feeds = {"X": make_range(2, 3, 5, 7)}

        optimized, rewrites = self.optimize(model, "GatherShape")

        self.assertEqual(len(rewrites), 0)
        self.assertEqual(node_types(optimized), ["Shape", "Gather"])
        self.assert_equivalent(model, optimized, feeds)

    def test_gather_shape_scalar_index(self):
        for index in range(4):
            with self.subTest(index=index):
                model = self.make_gather_shape_model(index, [])
                feeds = {"X": make_range(2, 3, 5, 7)}

                optimized, rewrites = self.optimize(model, "GatherShape")

                self.assertEqual(len(rewrites), 1)
                self.assertEqual(node_types(optimized), ["Shape", "Squeeze"])
                squeeze = optimized.graph.node[1]
                values = initializer_values(optimized)
                numpy.testing.assert_array_equal(values[squeeze.input[1]], numpy.array([0]))
                self.assert_equivalent(model, optimized, feeds)

    def make_gather_gather_model(self, outer_indices, output_shape, shared=False):
        """Builds two consecutive axis-zero Gather nodes."""
        nodes = [
            oh.make_node("Gather", ["X", "inner_indices"], ["Y"], axis=0),
            oh.make_node("Gather", ["Y", "outer_indices"], ["Z"], axis=0),
        ]
        outputs = [make_value_info("Z", TensorProto.FLOAT, output_shape)]
        if shared:
            nodes.append(oh.make_node("Add", ["Y", "Y"], ["W"]))
            outputs.append(make_value_info("W", TensorProto.FLOAT, [5, 4]))
        return make_model(
            nodes,
            [make_value_info("X", TensorProto.FLOAT, [5, 4])],
            outputs,
            [
                make_initializer("inner_indices", [1, 3, 0, 2, 4]),
                make_initializer("outer_indices", outer_indices),
            ],
        )

    def test_gather_gather_scalar_index(self):
        model = self.make_gather_gather_model(2, [4])
        feeds = {"X": make_range(5, 4)}

        optimized, rewrites = self.optimize(model, "GatherGather")

        self.assertEqual(len(rewrites), 1)
        self.assertEqual(node_types(optimized), ["Gather"])
        gather = optimized.graph.node[0]
        values = initializer_values(optimized)
        self.assertEqual(values[gather.input[1]].shape, ())
        self.assertEqual(values[gather.input[1]].item(), 0)
        self.assert_equivalent(model, optimized, feeds)

    def test_gather_gather_1d_index(self):
        model = self.make_gather_gather_model([2, 0, 1], [3, 4])
        feeds = {"X": make_range(5, 4)}

        optimized, rewrites = self.optimize(model, "GatherGather")

        self.assertEqual(len(rewrites), 1)
        self.assertEqual(node_types(optimized), ["Gather"])
        gather = optimized.graph.node[0]
        values = initializer_values(optimized)
        numpy.testing.assert_array_equal(values[gather.input[1]], numpy.array([0, 1, 3]))
        self.assert_equivalent(model, optimized, feeds)

    def test_gather_gather_inner_used_twice(self):
        model = self.make_gather_gather_model(2, [4], shared=True)
        feeds = {"X": make_range(5, 4)}

        optimized, rewrites = self.optimize(model, "GatherGather")

        self.assertEqual(len(rewrites), 1)
        self.assertEqual(node_types(optimized), ["Gather", "Gather", "Add"])
        self.assertEqual(optimized.graph.node[0].output[0], "Y")
        self.assert_equivalent(model, optimized, feeds)

    def make_gather_concat_model(
        self, concat_inputs, constants, indices, output_shape, *, shared=False
    ):
        """Builds Gather(Concat(...), indices) with one dynamic input."""
        nodes = [
            oh.make_node("Concat", concat_inputs, ["Z"], axis=0),
            oh.make_node("Gather", ["Z", "indices"], ["Y"]),
        ]
        outputs = [make_value_info("Y", TensorProto.INT64, output_shape)]
        if shared:
            nodes.append(oh.make_node("Identity", ["Z"], ["W"]))
            total = 5 + sum(len(value) for value in constants.values())
            outputs.append(make_value_info("W", TensorProto.INT64, [total]))
        initializers = [make_initializer(name, value) for name, value in constants.items()]
        initializers.append(make_initializer("indices", indices))
        return make_model(
            nodes, [make_value_info("X", TensorProto.INT64, [5])], outputs, initializers
        )

    def check_gather_concat(
        self, concat_inputs, constants, indices, output_shape, expected_indices
    ):
        """Checks a successful GatherConcat rewrite."""
        model = self.make_gather_concat_model(concat_inputs, constants, indices, output_shape)
        feeds = {"X": numpy.arange(5, dtype=numpy.int64)}

        optimized, rewrites = self.optimize(model, "GatherConcat")

        self.assertEqual(len(rewrites), 1)
        self.assertEqual(node_types(optimized), ["Gather"])
        gather = optimized.graph.node[0]
        self.assertEqual(gather.input[0], "X")
        values = initializer_values(optimized)
        numpy.testing.assert_array_equal(
            values[gather.input[1]], numpy.asarray(expected_indices, dtype=numpy.int64)
        )
        self.assert_equivalent(model, optimized, feeds)

    def test_gather_concat_scalar_no_pre(self):
        self.check_gather_concat(["X", "C"], {"C": [10, 20]}, 3, [], 3)

    def test_gather_concat_scalar_with_pre(self):
        self.check_gather_concat(["C", "X"], {"C": [10, 20, 30]}, 4, [], 1)

    def test_gather_concat_1d_index_with_pre(self):
        self.check_gather_concat(["C", "X"], {"C": [10, 20, 30]}, [3, 5, 4], [3], [0, 2, 1])

    def test_gather_concat_x_middle_with_post(self):
        self.check_gather_concat(
            ["C1", "X", "C2"], {"C1": [10, 20], "C2": [30, 40, 50]}, 4, [], 2
        )

    def test_gather_concat_no_match_index_outside_x(self):
        model = self.make_gather_concat_model(
            ["C1", "X", "C2"], {"C1": [10, 20], "C2": [30, 40, 50]}, 8, []
        )
        feeds = {"X": numpy.arange(5, dtype=numpy.int64)}

        optimized, rewrites = self.optimize(model, "GatherConcat")

        self.assertEqual(len(rewrites), 0)
        self.assertEqual(node_types(optimized), ["Concat", "Gather"])
        self.assert_equivalent(model, optimized, feeds)

    def test_gather_concat_concat_used_twice(self):
        model = self.make_gather_concat_model(["C", "X"], {"C": [10, 20, 30]}, 4, [], shared=True)
        feeds = {"X": numpy.arange(5, dtype=numpy.int64)}

        optimized, rewrites = self.optimize(model, "GatherConcat")

        self.assertEqual(len(rewrites), 1)
        self.assertEqual(node_types(optimized), ["Concat", "Gather", "Identity"])
        self.assertEqual(optimized.graph.node[0].output[0], "Z")
        self.assert_equivalent(model, optimized, feeds)


if __name__ == "__main__":
    unittest.main(verbosity=2)
