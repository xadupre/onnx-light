"""Tests transfer-aware proposals for local function bodies."""

import itertools
import unittest

import numpy

import onnx_light.onnx.checker as checker
import onnx_light.onnx.helper as oh
from onnx_light.onnx import TensorProto
from onnx_light.ext_test_case import ExtTestCase, import_or_skip
from onnx_light.onnx_core.graph_builder import GraphBuilder
from onnx_light.onnx_core.graph_partition import propose_operator_groups
from onnx_light.onnx_core.optimization import GraphGraph, PartitionGroupPattern
from onnx_light.onnx_py._onnxpycore import graph_partition as native_partition


class TestGraphPartition(ExtTestCase):
    @staticmethod
    def chain(shapes, dtype=TensorProto.FLOAT):
        names = ["x"] + [f"v{i}" for i in range(len(shapes) - 2)] + ["y"]
        values = [
            oh.make_tensor_value_info(name, dtype, shape)
            for name, shape in zip(names, shapes, strict=True)
        ]
        return oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Identity", [before], [after])
                    for before, after in itertools.pairwise(names)
                ],
                "chain",
                values[:1],
                values[-1:],
                value_info=values[1:-1],
            )
        )

    def test_tiny_interface_large_internal_values(self):
        graph = oh.make_graph(
            [
                oh.make_node("Expand", ["x", "shape"], ["large"]),
                oh.make_node("Relu", ["large"], ["positive"]),
                oh.make_node("ReduceSum", ["positive"], ["y"], keepdims=0),
            ],
            "expand_reduce",
            [oh.make_tensor_value_info("x", TensorProto.FLOAT, [])],
            [oh.make_tensor_value_info("y", TensorProto.FLOAT, [])],
            initializer=[oh.make_tensor("shape", TensorProto.INT64, [1], [4096])],
            value_info=[
                oh.make_tensor_value_info(name, TensorProto.FLOAT, [4096])
                for name in ("large", "positive")
            ],
        )
        model = oh.make_model(graph)
        before = model.SerializeToString()
        groups = propose_operator_groups(model, max_boundary_bytes=8)
        self.assertEqual(len(groups), 1)
        group = groups[0]
        self.assertEqual(group.node_indices, (0, 1, 2))
        self.assertEqual(group.inputs, ("x", "shape"))
        self.assertEqual(group.outputs, ("y",))
        self.assertEqual(group.transfer_bytes, 8)
        self.assertEqual(model.SerializeToString(), before)
        self.assertEqual(propose_operator_groups(model, max_boundary_bytes=7), [])
        # The proposed interface can directly describe a valid local function.
        checker.check_model(model)
        function = oh.make_function(
            "local",
            "TinyInterface",
            group.inputs,
            group.outputs,
            [model.graph.node[i] for i in group.node_indices],
            list(model.opset_import),
        )
        model.functions.append(function)
        model.opset_import.append(oh.make_opsetid("local", 1))
        model.graph.node.clear()
        model.graph.node.append(
            oh.make_node("TinyInterface", group.inputs, group.outputs, domain="local")
        )
        checker.check_model(model)

    def test_unknown_internal_sizes_are_allowed(self):
        model = self.chain([[], ["N"], [], None, []])
        groups = propose_operator_groups(model, max_boundary_bytes=8)
        self.assertEqual([g.node_indices for g in groups], [(0, 1, 2, 3)])
        self.assertEqual(groups[0].transfer_bytes, 8)

    def test_extracts_tiny_island_after_large_input(self):
        graph = oh.make_graph(
            [
                oh.make_node("Shape", ["x"], ["shape"]),
                oh.make_node("Gather", ["shape", "index"], ["dimension"]),
                oh.make_node("Add", ["dimension", "one"], ["y"]),
            ],
            "shape_island",
            [oh.make_tensor_value_info("x", TensorProto.FLOAT, [4096])],
            [oh.make_tensor_value_info("y", TensorProto.INT64, [])],
            initializer=[
                oh.make_tensor(name, TensorProto.INT64, [], [value])
                for name, value in (("index", 0), ("one", 1))
            ],
            value_info=[
                oh.make_tensor_value_info("shape", TensorProto.INT64, [1]),
                oh.make_tensor_value_info("dimension", TensorProto.INT64, []),
            ],
        )
        group = propose_operator_groups(graph, max_boundary_bytes=16)[0]
        self.assertEqual(group.node_indices, (1, 2))
        self.assertEqual(group.inputs, ("shape", "index", "one"))
        self.assertEqual(group.outputs, ("y",))
        self.assertEqual(group.transfer_bytes, 16)

    def test_unknown_boundaries_are_not_free(self):
        for shape in (None, ["N"], [None]):
            with self.subTest(shape=shape):
                model = self.chain([shape, [1024], shape])
                self.assertEqual(propose_operator_groups(model), [])
                group = propose_operator_groups(model, value_sizes={"x": 4, "y": 4})[0]
                self.assertEqual(group.transfer_bytes, 8)
        model = self.chain([[], [], []], dtype=TensorProto.STRING)
        self.assertEqual(propose_operator_groups(model), [])

    def test_packed_types_scalars_and_empty_tensors(self):
        for dtype, shape, cost in (
            (TensorProto.FLOAT, [], 8),
            (TensorProto.INT64, [], 16),
            (TensorProto.BOOL, [3], 6),
            (TensorProto.INT4, [3], 4),
            (TensorProto.UINT2, [5], 4),
            (TensorProto.FLOAT, [0, 1024], 0),
        ):
            with self.subTest(dtype=dtype, shape=shape):
                model = self.chain([shape, [10000], shape], dtype=dtype)
                group = propose_operator_groups(model, max_boundary_bytes=cost)[0]
                self.assertEqual(group.transfer_bytes, cost)

    def test_constants_keep_explicit_inputs_but_have_no_cost(self):
        model = self.chain([[], [], []])
        graph = model.graph
        graph.node[0:0] = [
            oh.make_node(
                "Constant",
                [],
                ["weight"],
                value=oh.make_tensor("weight", TensorProto.FLOAT, [1], [1]),
            )
        ]
        graph.node[1].input.append("weight")
        # Duplicate uses must not duplicate the function interface or the charge.
        graph.node[1].input.append("x")
        graph.node[2].input.append("weight")
        group = propose_operator_groups(model, max_boundary_bytes=8)[0]
        self.assertEqual(group.node_indices, (1, 2))
        self.assertEqual(group.inputs, ("x", "weight"))
        self.assertEqual(group.transfer_bytes, 8)
        graph.node[0].domain = "custom"
        self.assertEqual(
            propose_operator_groups(model, max_boundary_bytes=8, max_group_nodes=2), []
        )

    def test_sparse_initializer_and_overridable_initializer(self):
        model = self.chain([[], [], []])
        sparse = oh.make_sparse_tensor(
            oh.make_tensor("weight", TensorProto.FLOAT, [1], [1]),
            oh.make_tensor("indices", TensorProto.INT64, [1], [0]),
            [10000],
        )
        model.graph.sparse_initializer.append(sparse)
        model.graph.node[0].input.append("weight")
        self.assertEqual(propose_operator_groups(model)[0].transfer_bytes, 8)
        model.graph.input.append(oh.make_tensor_value_info("weight", TensorProto.FLOAT, [10000]))
        self.assertEqual(propose_operator_groups(model), [])
        model.graph.sparse_initializer.clear()
        model.graph.initializer.append(oh.make_tensor("weight", TensorProto.FLOAT, [1], [1]))
        self.assertEqual(propose_operator_groups(model), [])

    def test_fanout_and_graph_outputs_are_boundary_outputs(self):
        model = self.chain([[], [4096], [], []])
        model.graph.node[2].input.append("v0")
        self.assertEqual(
            propose_operator_groups(model, max_group_nodes=2, max_boundary_bytes=8), []
        )
        model.graph.node[2].input.clear()
        model.graph.node[2].input.append("v1")
        group = propose_operator_groups(model, max_group_nodes=2, max_boundary_bytes=8)[0]
        self.assertEqual(group.node_indices, (0, 1))
        model.graph.output.append(oh.make_tensor_value_info("v0", TensorProto.FLOAT, [4096]))
        self.assertEqual(propose_operator_groups(model, max_boundary_bytes=8), [])

    def test_multiple_outputs_preserve_port_order(self):
        graph = oh.make_graph(
            [
                oh.make_node("Split", ["x"], ["a", "b"], num_outputs=2),
                oh.make_node("Add", ["a", "b"], ["sum"]),
            ],
            "multiple_outputs",
            [oh.make_tensor_value_info("x", TensorProto.FLOAT, [4])],
            [
                oh.make_tensor_value_info("sum", TensorProto.FLOAT, [2]),
                oh.make_tensor_value_info("a", TensorProto.FLOAT, [2]),
            ],
            value_info=[oh.make_tensor_value_info("b", TensorProto.FLOAT, [2])],
        )
        group = propose_operator_groups(graph, max_boundary_bytes=32)[0]
        self.assertEqual(group.outputs, ("a", "sum"))
        self.assertEqual(group.transfer_bytes, 32)
        self.assertEqual(propose_operator_groups(graph, max_boundary_bytes=31), [])

    def test_multiple_disjoint_proposals(self):
        model = self.chain([[], [4096], [], [4096], []])
        groups = propose_operator_groups(model, max_group_nodes=2, max_boundary_bytes=8)
        self.assertEqual([g.node_indices for g in groups], [(0, 1), (2, 3)])
        self.assertEqual([g.transfer_bytes for g in groups], [8, 8])

    def test_native_binding_matches_python_facade(self):
        model = self.chain([[], [4096], [], [4096], []])
        options = {"max_group_nodes": 2, "max_boundary_bytes": 8}
        native = native_partition.propose_operator_groups(model.graph, **options)
        python = propose_operator_groups(model, **options)
        self.assertEqual(len(native), len(python))
        for actual, expected in zip(native, python, strict=True):
            self.assertEqual(tuple(actual.node_indices), expected.node_indices)
            self.assertEqual(tuple(actual.inputs), expected.inputs)
            self.assertEqual(tuple(actual.outputs), expected.outputs)
            self.assertEqual(actual.transfer_bytes, expected.transfer_bytes)

    def test_native_binding_validates_empty_graph(self):
        graph = oh.make_graph([], "empty", [], [])
        for kwargs in (
            {"max_boundary_bytes": -1},
            {"max_boundary_bytes": True},
            {"min_group_nodes": 1},
            {"max_group_nodes": 1},
            {"value_sizes": {"x": -1}},
            {"value_sizes": {"x": True}},
            {"value_sizes": {"x": 1 << 64}},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                native_partition.propose_operator_groups(graph, **kwargs)

    def test_native_sizes_do_not_wrap(self):
        model = self.chain([[1 << 62, 16], [], [1 << 62, 16]])
        self.assertEqual(propose_operator_groups(model, max_boundary_bytes=(1 << 64) - 1), [])
        model = self.chain([[], [], []])
        # Two individually representable boundary sizes overflow their sum.
        with self.assertRaises((ValueError, OverflowError)):
            propose_operator_groups(
                model, max_boundary_bytes=(1 << 64) - 1, value_sizes={"x": 1 << 63, "y": 1 << 63}
            )

    def test_nested_captures_are_charged_and_exported(self):
        model = self.chain([[], [4096], [], []])
        branch = oh.make_graph(
            [oh.make_node("Identity", ["v0"], ["branch_y"])],
            "branch",
            [],
            [oh.make_tensor_value_info("branch_y", TensorProto.FLOAT, [4096])],
        )
        model.graph.node[2].attribute.append(oh.make_attribute("body", branch))
        self.assertEqual(
            propose_operator_groups(model, max_group_nodes=2, max_boundary_bytes=8), []
        )
        # A capture originating outside the group must also become an input.
        model.graph.node[2].attribute.clear()
        model.graph.node[1].attribute.append(oh.make_attribute("body", branch))
        group = propose_operator_groups(model, max_boundary_bytes=8)[0]
        self.assertEqual(group.inputs, ("x",))
        self.assertEqual(group.outputs, ("y",))
        model.graph.input.append(oh.make_tensor_value_info("external", TensorProto.FLOAT, [4096]))
        capture = model.graph.node[1].attribute[0].g.node[0]
        capture.input.clear()
        capture.input.append("external")
        self.assertEqual(propose_operator_groups(model, max_boundary_bytes=8), [])

    def test_minimum_cost_matches_exhaustive_partitions(self):
        # Arbitrary descriptors exercise costing independently of operator semantics.
        for elements in ([1, 100, 1, 100, 1], [1, 1, 100, 1, 1, 100, 1]):
            with self.subTest(elements=elements):
                model = self.chain([[n] for n in elements])
                count = len(elements) - 1
                groups = propose_operator_groups(model, max_boundary_bytes=16, max_group_nodes=3)
                grouped = {i for group in groups for i in group.node_indices}
                actual = sum(g.transfer_bytes for g in groups) + sum(
                    4 * (elements[i] + elements[i + 1]) for i in range(count) if i not in grouped
                )
                costs = []
                for cuts in itertools.product((False, True), repeat=count - 1):
                    boundaries = [0] + [i + 1 for i, cut in enumerate(cuts) if cut] + [count]
                    total = 0
                    for start, end in itertools.pairwise(boundaries):
                        cost = 4 * (elements[start] + elements[end])
                        if end - start > 3 or (end - start > 1 and cost > 16):
                            break
                        total += cost
                    else:
                        costs.append(total)
                self.assertEqual(actual, min(costs))
                self.assertEqual(
                    groups,
                    propose_operator_groups(model, max_boundary_bytes=16, max_group_nodes=3),
                )

    def test_empty_constant_only_and_optional_inputs(self):
        graph = oh.make_graph([], "empty", [], [])
        self.assertEqual(propose_operator_groups(graph), [])
        graph.node.append(oh.make_node("Constant", [], ["c"], value_int=1))
        self.assertEqual(propose_operator_groups(graph), [])
        model = self.chain([[], [], []])
        model.graph.node[0].input.append("")
        self.assertEqual(propose_operator_groups(model)[0].inputs, ("x",))
        self.assertEqual(propose_operator_groups(model, min_group_nodes=3), [])

    def test_invalid_options_and_graphs(self):
        graph = oh.make_graph([], "empty", [], [])
        for kwargs in (
            {"max_boundary_bytes": -1},
            {"max_boundary_bytes": True},
            {"min_group_nodes": 1},
            {"max_group_nodes": 1},
            {"value_sizes": {"x": -1}},
            {"value_sizes": {"x": True}},
            {"value_sizes": {"x": 1.5}},
            {"max_boundary_bytes": 1 << 64},
            {"max_group_nodes": 1 << 64},
            {"min_group_nodes": 1 << 64, "max_group_nodes": 1 << 64},
            {"value_sizes": {"x": 1 << 64}},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                propose_operator_groups(graph, **kwargs)
        for replacement in ("undefined", "y"):
            model = self.chain([[], [], []])
            model.graph.node[0].input.clear()
            model.graph.node[0].input.append(replacement)
            with self.assertRaisesRegex(ValueError, "undefined or non-topological"):
                propose_operator_groups(model)
        model = self.chain([[], [], []])
        model.graph.node[1].output.clear()
        model.graph.node[1].output.append("x")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            propose_operator_groups(model)
        model = self.chain([[], [], []])
        model.graph.output[0].name = "undefined"
        with self.assertRaisesRegex(ValueError, "Undefined graph output"):
            propose_operator_groups(model)


class TestPartitionGroupPattern(ExtTestCase):
    def test_extracts_and_executes_large_intermediates_with_tiny_boundaries(self):
        reference = import_or_skip("onnx_light.onnx.reference", "ReferenceEvaluator")
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Expand", ["x", "shape"], ["large"]),
                    oh.make_node("Relu", ["large"], ["positive"]),
                    oh.make_node("ReduceSum", ["positive"], ["y"], keepdims=0),
                ],
                "expand_reduce",
                [oh.make_tensor_value_info("x", TensorProto.FLOAT, [])],
                [oh.make_tensor_value_info("y", TensorProto.FLOAT, [])],
                initializer=[oh.make_tensor("shape", TensorProto.INT64, [1], [4096])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
        )
        builder = GraphBuilder(model)
        graph = GraphGraph(builder, [PartitionGroupPattern(max_boundary_bytes=8)])
        plan = graph.optimize()
        grouped = builder.to_model()
        self.assertEqual(len(plan.rewrites), 1)
        self.assertEqual(len(grouped.graph.node), 1)
        self.assertEqual(list(grouped.graph.node[0].input), ["x", "shape"])
        self.assertEqual(list(grouped.graph.node[0].output), ["y"])
        self.assertEqual(
            [node.op_type for node in grouped.functions[0].node], ["Expand", "Relu", "ReduceSum"]
        )
        checker.check_model(grouped)
        replayed = GraphBuilder(model).apply_rewritings(plan).to_model()
        checker.check_model(replayed)
        for value in (-2, 0, 3):
            feeds = {"x": numpy.array(value, dtype=numpy.float32)}
            self.assertEqualArray(
                reference(model).run(None, feeds)[0], reference(grouped).run(None, feeds)[0]
            )
            self.assertEqualArray(
                reference(model).run(None, feeds)[0], reference(replayed).run(None, feeds)[0]
            )
        self.assertEqual(len(graph.optimize().rewrites), 0)
        self.assertEqual(len(builder.to_model().functions), 1)

    def test_limits_unknown_boundaries_and_registration(self):
        for boundary, maximum, shape, functions in (
            (7, 4, [], 0),
            (8, 4, [], 1),
            (8, 2, [], 2),
            (1024, 4, ["N"], 0),
        ):
            with self.subTest(boundary=boundary, maximum=maximum, shape=shape):
                model = oh.make_model(
                    oh.make_graph(
                        [
                            oh.make_node(op, [before], [after])
                            for op, before, after in (
                                ("Neg", "x", "v0"),
                                ("Abs", "v0", "v1"),
                                ("Neg", "v1", "v2"),
                                ("Abs", "v2", "y"),
                            )
                        ],
                        "chain",
                        [oh.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
                        [oh.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
                    ),
                    opset_imports=[oh.make_opsetid("", 18)],
                )
                builder = GraphBuilder(model)
                GraphGraph(
                    builder,
                    [PartitionGroupPattern(max_boundary_bytes=boundary, max_group_nodes=maximum)],
                ).optimize()
                self.assertEqual(len(builder.to_model().functions), functions)
        builder = GraphBuilder(model)
        graph = GraphGraph(builder, ["PartitionGroup"])
        self.assertEqual(graph.patterns[0].name, "PartitionGroup")
        self.assertNotIn(
            "PartitionGroup", [pattern.name for pattern in GraphGraph(builder).patterns]
        )
        with self.assertRaises(ValueError):
            PartitionGroupPattern(min_group_nodes=1)

    def test_device_configurations_prevent_extraction(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Neg", ["x"], ["middle"]),
                    oh.make_node("Constant", [], ["weight"], value_float=2),
                    oh.make_node("Add", ["middle", "weight"], ["y"]),
                ],
                "configured",
                [oh.make_tensor_value_info("x", TensorProto.FLOAT, [])],
                [oh.make_tensor_value_info("y", TensorProto.FLOAT, [])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
        )
        for configured in (0, 1):
            with self.subTest(configured=configured):
                configuration = model.graph.node[configured].device_configurations.add()
                configuration.configuration_id = "sharded"
                configuration.pipeline_stage = 2
                builder = GraphBuilder(model)
                self.assertEqual(
                    len(GraphGraph(builder, ["PartitionGroup"]).optimize().rewrites), 0
                )
                self.assertEqual(len(builder.to_model().functions), 0)
                model.graph.node[configured].device_configurations.clear()

    def test_extracts_capturing_subgraphs_without_recursive_functions(self):
        reference = import_or_skip("onnx_light.onnx.reference", "ReferenceEvaluator")
        branch = oh.make_graph(
            [
                oh.make_node("Neg", ["x"], ["middle"]),
                oh.make_node("Abs", ["middle"], ["branch_y"]),
            ],
            "branch",
            [],
            [oh.make_tensor_value_info("branch_y", TensorProto.FLOAT, [])],
        )
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("If", ["cond"], ["y"], then_branch=branch, else_branch=branch)],
                "conditional",
                [
                    oh.make_tensor_value_info("cond", TensorProto.BOOL, []),
                    oh.make_tensor_value_info("x", TensorProto.FLOAT, []),
                ],
                [oh.make_tensor_value_info("y", TensorProto.FLOAT, [])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
        )
        builder = GraphBuilder(model)
        GraphGraph(builder, ["PartitionGroup"]).optimize()
        grouped = builder.to_model()
        self.assertEqual(len(grouped.functions), 2)
        checker.check_model(grouped)
        for condition in (False, True):
            feeds = {"cond": numpy.array(condition), "x": numpy.array(-3, dtype=numpy.float32)}
            self.assertEqualArray(
                reference(model).run(None, feeds)[0], reference(grouped).run(None, feeds)[0]
            )


if __name__ == "__main__":
    unittest.main()
