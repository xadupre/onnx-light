# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests native model deltas using the existing rewriting representation."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy

from onnx_light import onnx
from onnx_light.ext_test_case import ExtTestCase
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh
from onnx_light.onnx import ModelProto, RewritingProto, TensorProto
from onnx_light.onnx.reference import ReferenceEvaluator
from onnx_light.onnx_core.optimization import GraphBuilder, apply_rewritings_to_model, diff_models
from onnx_light.onnx_lib.external_data_helper import load_external_data_for_model


def _model(op="Add", weight=(1, 2)):
    """Creates an independent model with a named weight."""
    return oh.make_model(
        oh.make_graph(
            [oh.make_node(op, ["x", "w"], ["y"])],
            "model",
            [oh.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
            [oh.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
            initializer=[onh.from_array(numpy.array(weight, dtype=numpy.float32), "w")],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
    )


class TestModelDiff(ExtTestCase):
    def assertEquivalent(self, expected, actual, feeds=None):
        if feeds is None:
            feeds = {"x": numpy.array([-2, 3], dtype=numpy.float32)}
        self.assertEqual(
            [(n.op_type, list(n.input), list(n.output)) for n in expected.graph.node],
            [(n.op_type, list(n.input), list(n.output)) for n in actual.graph.node],
        )
        self.assertEqual(
            [t.name for t in expected.graph.initializer],
            [t.name for t in actual.graph.initializer],
        )
        self.assertEqual([f.name for f in expected.functions], [f.name for f in actual.functions])
        for left, right in zip(expected.graph.initializer, actual.graph.initializer):
            self.assertEqualArray(onh.to_array(left), onh.to_array(right))
        for left, right in zip(
            ReferenceEvaluator(expected).run(None, feeds),
            ReferenceEvaluator(actual).run(None, feeds),
        ):
            self.assertEqualArray(left, right)

    def test_diff_rewritings_accepts_models_and_builders_without_mutation(self):
        canonical, expected = _model(), _model("Mul", (7, 8))
        source, target = GraphBuilder(canonical), GraphBuilder(expected)
        source_before = source.to_model().SerializeToString()
        target_before = target.to_model().SerializeToString()
        for destination in (expected, target):
            plan = source.diff_rewritings(destination)
            self.assertEquivalent(expected, apply_rewritings_to_model(canonical, plan))
        self.assertEqual(source.to_model().SerializeToString(), source_before)
        self.assertEqual(target.to_model().SerializeToString(), target_before)
        with self.assertRaisesRegex(ValueError, "root builders"):
            source.diff_rewritings(source.make_subgraph("nested"))

    def test_named_plans_serialize_with_one_canonical_graph(self):
        canonical = _model()
        targets = {"cpu": _model("Mul"), "cuda": _model("Sub", (7, 8))}
        for name, target in targets.items():
            canonical.add_rewriting(name, diff_models(canonical, target))
        with TemporaryDirectory() as folder:
            path = Path(folder) / "model.onnx"
            onnx.save(canonical, path)
            loaded = onnx.load(path)
            self.assertEqual(len(loaded.graph.node), 1)
            self.assertEqual(set(loaded.rewritings), set(targets))
            self.assertFalse(loaded.rewritings["cpu"].rewrites[0].added_initializers)
            for name, target in targets.items():
                self.assertEquivalent(
                    target, apply_rewritings_to_model(loaded, loaded.rewritings[name])
                )

    def test_initializer_replacement_order_and_node_moves(self):
        base = _model()
        base.graph.initializer.append(onh.from_array(numpy.array([3, 4], numpy.float32), "b"))
        base.graph.node.clear()
        base.graph.node.extend(
            [
                oh.make_node("Add", ["x", "w"], ["a"]),
                oh.make_node("Mul", ["x", "b"], ["b_out"]),
                oh.make_node("Add", ["a", "b_out"], ["y"]),
            ]
        )
        target = ModelProto()
        target.CopyFrom(base)
        target.graph.initializer.clear()
        target.graph.initializer.extend(
            [
                onh.from_array(numpy.array([3, 4], numpy.float32), "b"),
                onh.from_array(numpy.array([5, 6], numpy.float32), "w"),
            ]
        )
        target.graph.node.clear()
        target.graph.node.extend([base.graph.node[1], base.graph.node[0], base.graph.node[2]])
        plan = diff_models(base, target)
        self.assertEqual([t.name for t in plan.rewrites[0].added_initializers], ["w"])
        self.assertTrue(plan.moved_nodes)
        serialized = RewritingProto()
        serialized.ParseFromString(plan.SerializeToString())
        self.assertTrue(plan.Equals(serialized))
        self.assertEquivalent(target, apply_rewritings_to_model(base, serialized))

    def test_initializer_permutation_uses_local_function_calls(self):
        base = _model()
        base.graph.initializer.append(onh.from_array(numpy.array([3, 4], numpy.float32), "b"))
        target = ModelProto()
        target.CopyFrom(base)
        target.graph.initializer.clear()
        target.graph.initializer.extend([base.graph.initializer[1], base.graph.initializer[0]])
        plan = diff_models(base, target)
        self.assertEqual(len(plan.added_functions), 1)
        self.assertFalse(plan.added_functions[0].node)
        self.assertEqual(len(plan.rewrites), 1)
        self.assertFalse(plan.rewrites[0].added_initializers)
        self.assertEqual(len(plan.rewrites[0].initializer_calls), 2)
        self.assertEqual(list(plan.rewrites[0].computed_initializer_positions), [0, 1])
        self.assertEquivalent(target, apply_rewritings_to_model(base, plan))
        invalid = RewritingProto()
        invalid.CopyFrom(plan)
        invalid.added_functions[0].output.clear()
        invalid.added_functions[0].output.append("undefined")
        with self.assertRaisesRegex(ValueError, "function output is undefined"):
            apply_rewritings_to_model(base, invalid)
        invalid.CopyFrom(plan)
        invalid.rewrites[0].initializer_calls[0].input.clear()
        invalid.rewrites[0].initializer_calls[0].input.append("x")
        with self.assertRaisesRegex(ValueError, "not an available constant"):
            apply_rewritings_to_model(base, invalid)
        base.graph.input.append(oh.make_tensor_value_info("w", TensorProto.FLOAT, [2]))
        target.graph.input.append(oh.make_tensor_value_info("w", TensorProto.FLOAT, [2]))
        with self.assertRaisesRegex(ValueError, "graph input defaults"):
            diff_models(base, target)

    def test_external_references_compare_contents_not_filenames(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            for name, model in [("base", _model()), ("target", _model("Mul"))]:
                directory = root / name
                directory.mkdir()
                onnx.save(
                    model,
                    directory / "model.onnx",
                    save_as_external_data=True,
                    location="weights.bin",
                    size_threshold=0,
                )
            base = onnx.load(root / "base/model.onnx")
            target = onnx.load(root / "target/model.onnx")
            plan = diff_models(
                base, target, base_dir=str(root / "base"), target_dir=str(root / "target")
            )
            self.assertFalse(plan.rewrites[0].added_initializers)
            result = apply_rewritings_to_model(base, plan)
            load_external_data_for_model(result, str(root / "base"))
            self.assertEquivalent(_model("Mul"), result)
            (root / "target/weights.bin").write_bytes(
                numpy.array([7, 8], numpy.float32).tobytes()
            )
            plan = diff_models(
                base, target, base_dir=str(root / "base"), target_dir=str(root / "target")
            )
            self.assertEqual(len(plan.rewrites[0].added_initializers), 1)
            serialized = RewritingProto()
            serialized.ParseFromString(plan.SerializeToString())
            result = apply_rewritings_to_model(base, serialized)
            load_external_data_for_model(result, str(root / "base"))
            self.assertEquivalent(_model("Mul", (7, 8)), result)

    def test_forwarding_recipes_chain_and_retain_runtime_dependencies(self):
        base = _model()
        base.graph.initializer.append(onh.from_array(numpy.array([3, 4], numpy.float32), "b"))
        for runtime_call in (False, True):
            with self.subTest(runtime_call=runtime_call):
                plan = RewritingProto()
                plan.added_functions.append(
                    oh.make_function("local", "SwapInputs", ["a", "b"], ["b", "a"], [], [])
                )
                edit = plan.rewrites.add()
                edit.removed_initializers.extend([0, 1])
                edit.initializer_calls.extend(
                    [
                        oh.make_node("SwapInputs", ["w", "b"], ["w", "b"], domain="local"),
                        oh.make_node(
                            "SwapInputs", ["w", "b"], ["copy_w", "copy_b"], domain="local"
                        ),
                    ]
                )
                edit.computed_initializer_positions.extend([0, 1, 2, 3])
                if runtime_call:
                    plan.required_opset_imports.append(oh.make_opsetid("local", 1))
                    edit.matched_nodes.append(0)
                    edit.added_nodes.append(
                        oh.make_node("SwapInputs", ["w", "x"], ["y", "unused"], domain="local")
                    )
                    edit.added_nodes_positions.append(0)
                captured = GraphBuilder(base).capture_rewritings(plan)
                result = apply_rewritings_to_model(base, captured)
                self.assertEqual(len(result.functions), int(runtime_call))
                for tensor, expected in zip(
                    result.graph.initializer, ([3, 4], [1, 2], [1, 2], [3, 4])
                ):
                    self.assertEqualArray(
                        onh.to_array(tensor), numpy.array(expected, numpy.float32)
                    )
                feeds = {"x": numpy.array([-2, 3], numpy.float32)}
                self.assertEqualArray(
                    ReferenceEvaluator(result).run(None, feeds)[0],
                    (
                        feeds["x"]
                        if runtime_call
                        else feeds["x"] + numpy.array([3, 4], numpy.float32)
                    ),
                )

    def test_nested_graph_edits_and_new_outer_captures(self):
        def make(op, weight=None, nested=False):
            branches = {
                name: oh.make_graph(
                    [oh.make_node(op, ["x"] if weight is None else ["x", weight], ["z"])],
                    name,
                    [],
                    [oh.make_tensor_value_info("z", TensorProto.FLOAT, [2])],
                )
                for name in ("then_branch", "else_branch")
            }
            if nested:
                branches = {
                    name: oh.make_graph(
                        [oh.make_node("If", ["condition"], ["z"], then_branch=g, else_branch=g)],
                        name,
                        [],
                        [oh.make_tensor_value_info("z", TensorProto.FLOAT, [2])],
                    )
                    for name, g in branches.items()
                }
            return oh.make_model(
                oh.make_graph(
                    [oh.make_node("If", ["condition"], ["y"], **branches)],
                    "outer",
                    [
                        oh.make_tensor_value_info("condition", TensorProto.BOOL, []),
                        oh.make_tensor_value_info("x", TensorProto.FLOAT, [2]),
                    ],
                    [oh.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
                    initializer=(
                        []
                        if weight is None
                        else [onh.from_array(numpy.array([1, 2], numpy.float32), weight)]
                    ),
                ),
                opset_imports=[oh.make_opsetid("", 18)],
            )

        for nested in (False, True):
            for base, target in [
                (make("Neg", nested=nested), make("Relu", nested=nested)),
                (make("Add", "old_w", nested), make("Mul", "new_w", nested)),
            ]:
                result = apply_rewritings_to_model(base, diff_models(base, target))
                for condition in (False, True):
                    self.assertEquivalent(
                        target,
                        result,
                        {
                            "condition": numpy.array(condition),
                            "x": numpy.array([-2, 3], numpy.float32),
                        },
                    )
        target = _model()
        target.graph.node.clear()
        target.graph.node.append(oh.make_node("Dropout", ["x"], ["y", ""]))
        result = apply_rewritings_to_model(_model(), diff_models(_model(), target))
        self.assertEqual(list(result.graph.node[0].output), ["y", ""])

    def test_local_functions_additions_and_conflicts(self):
        def make(op):
            model = _model()
            model.graph.node.clear()
            model.graph.node.append(oh.make_node("F", ["x"], ["y"], domain="local"))
            model.opset_import.append(oh.make_opsetid("local", 1))
            model.functions.append(
                oh.make_function(
                    "local",
                    "F",
                    ["a"],
                    ["b"],
                    [oh.make_node(op, ["a"], ["b"])],
                    [oh.make_opsetid("", 18)],
                )
            )
            return model

        base, target = make("Neg"), make("Relu")
        with self.assertRaisesRegex(ValueError, "conflicting local function"):
            diff_models(base, target)
        self.assertEquivalent(base, apply_rewritings_to_model(base, diff_models(base, base)))
        no_functions = _model()
        no_functions.opset_import.append(oh.make_opsetid("local", 1))
        plan = diff_models(no_functions, target)
        self.assertFalse(plan.replaced_function_ids)
        self.assertEquivalent(target, apply_rewritings_to_model(no_functions, plan))
        with self.assertRaisesRegex(ValueError, "removing local functions"):
            diff_models(base, no_functions)

    def test_rejects_incompatible_interfaces_and_corrupt_hashes(self):
        base = _model()
        for field in ("input", "output"):
            target = _model("Mul")
            getattr(target.graph, field)[0].type.tensor_type.elem_type = TensorProto.FLOAT16
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "incompatible"):
                diff_models(base, target)
        plan = diff_models(base, _model("Mul"))
        plan.hash_after ^= 1
        with self.assertRaisesRegex(ValueError, "final structural hash mismatch"):
            apply_rewritings_to_model(base, plan)
        plan.ClearField("hash_before")
        with self.assertRaisesRegex(ValueError, "hash_before and hash_after"):
            apply_rewritings_to_model(base, plan)

    def test_multiple_shards_keep_nonzero_ranges(self):
        expected = _model("Mul")
        expected.graph.initializer.append(
            onh.from_array(numpy.array([3, 4], numpy.float32), "bias")
        )
        with TemporaryDirectory() as folder:
            root = Path(folder)
            base = ModelProto()
            base.CopyFrom(expected)
            base.graph.node[0].op_type = "Add"
            for index, tensor in enumerate(base.graph.initializer):
                offset, location = 8 * (index + 1), f"shard{index}.bin"
                (root / location).write_bytes(bytes(offset) + tensor.raw_data)
                tensor.ClearField("raw_data")
                tensor.data_location = TensorProto.EXTERNAL
                for key, value in [
                    ("location", location),
                    ("offset", str(offset)),
                    ("length", "8"),
                ]:
                    tensor.external_data.add(key=key, value=value)
            plan = diff_models(base, expected, base_dir=str(root))
            self.assertFalse(plan.rewrites[0].added_initializers)
            result = apply_rewritings_to_model(base, plan)
            for index, tensor in enumerate(result.graph.initializer):
                values = {e.key: e.value for e in tensor.external_data}
                self.assertEqual(values["offset"], str(8 * (index + 1)))
                self.assertEqual(values["length"], "8")
            load_external_data_for_model(result, str(root))
            self.assertEquivalent(expected, result)

    def test_large_metadata_only_delta_and_initializer_permutation(self):
        base = _model()
        base.graph.initializer.clear()
        for name in ("huge_a", "huge_b"):
            weight = TensorProto(
                name=name,
                data_type=TensorProto.FLOAT,
                dims=[2**34],
                data_location=TensorProto.EXTERNAL,
            )
            for key, value in [
                ("location", "does-not-exist.bin"),
                ("offset", "4096"),
                ("length", str(2**36)),
            ]:
                weight.external_data.add(key=key, value=value)
            base.graph.initializer.append(weight)
        base.graph.node.clear()
        base.graph.node.extend(
            [
                oh.make_node(
                    "Neg", ["x" if i == 0 else f"v{i}"], ["y" if i == 2999 else f"v{i + 1}"]
                )
                for i in range(3000)
            ]
        )
        target = ModelProto()
        target.CopyFrom(base)
        target.graph.node[-1].op_type = "Relu"
        target.graph.initializer.clear()
        target.graph.initializer.extend([base.graph.initializer[1], base.graph.initializer[0]])
        plan = diff_models(base, target)
        self.assertEqual(len(plan.rewrites), 2)
        self.assertEqual(list(plan.rewrites[0].matched_nodes), [2999])
        self.assertTrue(all(not edit.added_initializers for edit in plan.rewrites))
        self.assertLess(len(plan.SerializeToString()), 1024)
        result = apply_rewritings_to_model(base, plan)
        self.assertEqual(result.graph.node[-1].op_type, "Relu")
        self.assertEqual([t.name for t in result.graph.initializer], ["huge_b", "huge_a"])
        self.assertFalse(result.functions)
        self.assertTrue(all(not t.raw_data for t in result.graph.initializer))


if __name__ == "__main__":
    unittest.main()
