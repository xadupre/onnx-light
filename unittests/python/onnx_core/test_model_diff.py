# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests native model deltas using the existing rewriting representation."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy

from onnx_light import onnx
from onnx_light.ext_test_case import ExtTestCase, import_or_skip
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh
from onnx_light.onnx import ModelProto, RewritingProto, TensorProto
from onnx_light.onnx_core.optimization import GraphBuilder, apply_rewritings_to_model, diff_models
from onnx_light.onnx_lib.external_data_helper import load_external_data_for_model

ReferenceEvaluator = import_or_skip("onnx_light.onnx.reference", "ReferenceEvaluator")


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


def _external_tensor(directory, values, name="literal"):
    """Creates an external tensor with identical metadata across storage directories."""
    directory.mkdir(exist_ok=True)
    tensor = onh.from_array(numpy.array(values, numpy.float32), name)
    (directory / "weights.bin").write_bytes(tensor.raw_data)
    tensor.ClearField("raw_data")
    tensor.data_location = TensorProto.EXTERNAL
    tensor.external_data.add(key="location", value="weights.bin")
    tensor.external_data.add(key="length", value=str(4 * len(values)))
    return tensor


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

    def test_constant_to_initializer_replacement_is_atomic(self):
        target = _model()
        base = _model()
        weight = base.graph.initializer[0]
        base.graph.initializer.clear()
        nodes = list(base.graph.node)
        base.graph.node.clear()
        base.graph.node.extend([oh.make_node("Constant", [], ["w"], value=weight), *nodes])
        plan = diff_models(base, target)
        self.assertEqual(len(plan.rewrites), 1)
        self.assertEqual(list(plan.rewrites[0].matched_nodes), [0])
        self.assertEquivalent(target, apply_rewritings_to_model(base, plan))

    def test_moves_use_dependency_repaired_node_positions(self):
        base, target = _model(), _model()
        base.graph.node.clear()
        base.graph.node.extend(
            [
                oh.make_node("Neg", ["x"], ["a"]),
                oh.make_node("Relu", ["x"], ["b"]),
                oh.make_node("Add", ["a", "b"], ["y"]),
            ]
        )
        target.graph.node.clear()
        target.graph.node.extend(
            [
                base.graph.node[1],
                oh.make_node("Neg", ["b"], ["n"]),
                base.graph.node[0],
                oh.make_node("Add", ["a", "n"], ["y"]),
            ]
        )
        plan = diff_models(base, target)
        self.assertEquivalent(target, apply_rewritings_to_model(base, plan))

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

    def test_deferred_weight_sources_and_serializable_new_payloads(self):
        for external in (False, True):
            with self.subTest(external=external), TemporaryDirectory() as folder:
                root = Path(folder)
                for name, shift in (("base", 0), ("target", 10)):
                    directory = root / name
                    directory.mkdir()
                    model = _model()
                    model.graph.initializer.append(
                        onh.from_array(numpy.arange(1024, dtype=numpy.float32) + shift, "unused")
                    )
                    if name == "target":
                        model.graph.initializer.append(
                            onh.from_array(numpy.arange(1024, dtype=numpy.float32), "new")
                        )
                    onnx.save(
                        model,
                        directory / "model.onnx",
                        save_as_external_data=external,
                        size_threshold=0,
                    )
                base = onnx.load(root / "base/model.onnx", defer_weights=True)
                target = onnx.load(root / "target/model.onnx", defer_weights=True)
                plan = diff_models(base, target)
                serialized = RewritingProto()
                serialized.ParseFromString(plan.SerializeToString())
                result = apply_rewritings_to_model(base, serialized)
                result.load_weights()
                expected = onnx.load(root / "target/model.onnx", load_external_data=True)
                self.assertEquivalent(expected, result)
                self.assertFalse(base.is_weight_loaded("unused"))
                self.assertFalse(target.is_weight_loaded("unused"))
                self.assertFalse(target.is_weight_loaded("new"))

    def test_independent_deferred_loads_share_unchanged_sources_without_io(self):
        for external in (False, True):
            with self.subTest(external=external), TemporaryDirectory() as folder:
                root = Path(folder)
                model = _model()
                model.graph.initializer.clear()
                model.graph.initializer.append(
                    onh.from_array(numpy.arange(1024, dtype=numpy.float32), "unused")
                )
                model.graph.sparse_initializer.append(
                    oh.make_sparse_tensor(
                        onh.from_array(numpy.arange(1024, dtype=numpy.float32), "sparse_values"),
                        onh.from_array(numpy.arange(1024, dtype=numpy.int64), "sparse_indices"),
                        [2048],
                    )
                )
                model.graph.node.clear()
                model.graph.node.append(oh.make_node("Identity", ["x"], ["y"]))
                path = root / "model.onnx"
                onnx.save(model, path, save_as_external_data=external, size_threshold=0)
                base = onnx.load(path, defer_weights=True)
                target = onnx.load(path, defer_weights=True)
                target.graph.node[0].op_type = "Neg"
                source, destination = GraphBuilder(base), GraphBuilder(target)
                for file in root.iterdir():
                    file.unlink()
                plan = source.diff_rewritings(destination)
                serialized = RewritingProto()
                serialized.ParseFromString(plan.SerializeToString())
                result = apply_rewritings_to_model(base, serialized)
                self.assertEqual(result.graph.node[0].op_type, "Neg")
                for model in (base, target, result):
                    for name in ("unused", "sparse_values", "sparse_indices"):
                        self.assertFalse(model.is_weight_loaded(name))
                self.assertTrue(all(not r.added_initializers for r in plan.rewrites))

    def test_deferred_inline_ranges_in_the_same_file_are_distinct(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "model.onnx"
            model = _model()
            for name, shift in (("first", 0), ("second", 10)):
                model.graph.initializer.append(
                    onh.from_array(numpy.arange(1024, dtype=numpy.float32) + shift, name)
                )
            onnx.save(model, path)
            base = onnx.load(path, defer_weights=True)
            target = onnx.load(path, defer_weights=True)
            target.graph.initializer[1].CopyFrom(target.graph.initializer[2])
            target.graph.initializer[1].name = "first"
            plan = diff_models(base, target)
            serialized = RewritingProto()
            serialized.ParseFromString(plan.SerializeToString())
            result = apply_rewritings_to_model(base, serialized)
            self.assertEqualArray(
                onh.to_array(result.graph.initializer[1]),
                numpy.arange(1024, dtype=numpy.float32) + 10,
            )
            self.assertFalse(base.is_weight_loaded("first"))
            self.assertFalse(target.is_weight_loaded("first"))

    def test_deferred_sparse_changes_are_rejected(self):
        for external in (False, True):
            for changed in ("values", "indices"):
                with (
                    self.subTest(external=external, changed=changed),
                    TemporaryDirectory() as folder,
                ):
                    root = Path(folder)
                    for name, shift in (("base", 0), ("target", 1)):
                        model = _model()
                        model.graph.sparse_initializer.append(
                            oh.make_sparse_tensor(
                                onh.from_array(
                                    numpy.arange(1024, dtype=numpy.float32)
                                    + (shift if changed == "values" else 0),
                                    "sparse_values",
                                ),
                                onh.from_array(
                                    numpy.arange(1024, dtype=numpy.int64)
                                    + (shift if changed == "indices" else 0),
                                    "sparse_indices",
                                ),
                                [2048],
                            )
                        )
                        directory = root / name
                        directory.mkdir()
                        onnx.save(
                            model,
                            directory / "model.onnx",
                            save_as_external_data=external,
                            size_threshold=0,
                        )
                    base = onnx.load(root / "base/model.onnx", defer_weights=True)
                    target = onnx.load(root / "target/model.onnx", defer_weights=True)
                    with self.assertRaisesRegex(ValueError, "persistent declarations"):
                        diff_models(base, target)
                    self.assertFalse(base.is_weight_loaded("sparse_values"))
                    self.assertFalse(target.is_weight_loaded("sparse_indices"))

    def test_nested_edits_use_replaced_parent_initializer_values(self):
        base, target = _model(), _model()
        for model in (base, target):
            model.graph.initializer.clear()
            model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 3
            model.graph.input.append(oh.make_tensor_value_info("condition", TensorProto.BOOL, []))
            model.graph.output[0].type.CopyFrom(oh.make_tensor_type_proto(TensorProto.FLOAT, [3]))
            shape = [2, -1] if model is base else [3, -1]
            model.graph.initializer.append(
                onh.from_array(numpy.array(shape, numpy.int64), "shape")
            )
            nodes = (
                [oh.make_node("Neg", ["x"], ["z"])]
                if model is base
                else [
                    oh.make_node("Reshape", ["x", "shape"], ["reshaped"]),
                    oh.make_node("Flatten", ["reshaped"], ["flat"], axis=0),
                    oh.make_node("Squeeze", ["flat"], ["z"]),
                ]
            )
            branch = oh.make_graph(
                nodes, "branch", [], [oh.make_tensor_value_info("z", TensorProto.FLOAT, [3])]
            )
            model.graph.node.clear()
            model.graph.node.append(
                oh.make_node("If", ["condition"], ["y"], then_branch=branch, else_branch=branch)
            )
        plan = diff_models(base, target)
        serialized = RewritingProto()
        serialized.ParseFromString(plan.SerializeToString())
        result = apply_rewritings_to_model(base, serialized)
        for condition in (False, True):
            self.assertEquivalent(
                target,
                result,
                {
                    "condition": numpy.array(condition),
                    "x": numpy.array([-2, 3, 5], numpy.float32),
                },
            )

    def test_external_tensor_attributes_and_local_function_definitions(self):
        for context in ("node", "body", "default", "repeated", "sparse"):
            with self.subTest(context=context), TemporaryDirectory() as folder:
                root = Path(folder)
                models = []
                for name, values in (("base", [1, 2]), ("target", [7, 8])):
                    tensor = _external_tensor(root / name, values)
                    model = _model()
                    model.graph.initializer.clear()
                    body = [
                        oh.make_node("Constant", [], ["w"], value=tensor),
                        oh.make_node("Add", ["x", "w"], ["y"]),
                    ]
                    if context == "node":
                        model.graph.node.clear()
                        model.graph.node.extend(body)
                    else:
                        function = oh.make_function(
                            "local",
                            "F",
                            ["x"],
                            ["y"],
                            body if context == "body" else [oh.make_node("Neg", ["x"], ["y"])],
                            [oh.make_opsetid("", 18)],
                        )
                        if context in ("default", "repeated", "sparse"):
                            function.attribute.append("extra")
                            if context == "repeated":
                                value = [tensor]
                            elif context == "sparse":
                                value = oh.make_sparse_tensor(
                                    tensor, onh.from_array(numpy.array([0, 1], numpy.int64)), [2]
                                )
                            else:
                                value = tensor
                            function.attribute_proto.append(oh.make_attribute("extra", value))
                        model.functions.append(function)
                        model.opset_import.append(oh.make_opsetid("local", 1))
                        model.graph.node.clear()
                        model.graph.node.append(oh.make_node("F", ["x"], ["y"], domain="local"))
                    models.append(model)
                base, target = models
                directories = {"base_dir": str(root / "base"), "target_dir": str(root / "target")}
                if context != "node":
                    with self.assertRaisesRegex(ValueError, "conflicting local function"):
                        diff_models(base, target, **directories)
                    base = _model()
                plan = diff_models(base, target, **directories)
                serialized = RewritingProto()
                serialized.ParseFromString(plan.SerializeToString())
                result = apply_rewritings_to_model(base, serialized)
                load_external_data_for_model(target, str(root / "target"))
                self.assertEquivalent(target, result)
                if context in ("default", "repeated", "sparse"):
                    attribute = result.functions[0].attribute_proto[0]
                    actual = (
                        attribute.tensors[0]
                        if context == "repeated"
                        else (
                            attribute.sparse_tensor.values if context == "sparse" else attribute.t
                        )
                    )
                    self.assertEqualArray(
                        onh.to_array(actual), numpy.array([7, 8], numpy.float32)
                    )

    def test_added_subgraphs_embed_target_weights_and_new_parent_captures(self):
        for external in (False, True):
            with self.subTest(external=external), TemporaryDirectory() as folder:
                root = Path(folder)
                base = _model()
                base.graph.input.append(
                    oh.make_tensor_value_info("condition", TensorProto.BOOL, [])
                )
                base.graph.node.clear()
                branch = oh.make_graph(
                    [oh.make_node("Neg", ["x"], ["z"])],
                    "branch",
                    [],
                    [oh.make_tensor_value_info("z", TensorProto.FLOAT, [2])],
                )
                base.graph.node.append(
                    oh.make_node(
                        "If", ["condition"], ["y"], then_branch=branch, else_branch=branch
                    )
                )
                target = ModelProto()
                target.CopyFrom(base)
                target.graph.node.clear()
                if external:
                    weight = _external_tensor(root / "target", [7, 8], "inner")
                    branch = oh.make_graph(
                        [oh.make_node("Add", ["x", "inner"], ["z"])],
                        "weighted",
                        [],
                        [oh.make_tensor_value_info("z", TensorProto.FLOAT, [2])],
                        initializer=[weight],
                    )
                    target.graph.node.append(oh.make_node("Not", ["condition"], ["inverted"]))
                    condition = "inverted"
                else:
                    branch.node[0].input.clear()
                    branch.node[0].input.append("v")
                    target.graph.node.append(oh.make_node("Neg", ["x"], ["v"]))
                    condition = "condition"
                target.graph.node.append(
                    oh.make_node("If", [condition], ["y"], then_branch=branch, else_branch=branch)
                )
                plan = diff_models(
                    base, target, base_dir=str(root), target_dir=str(root / "target")
                )
                serialized = RewritingProto()
                serialized.ParseFromString(plan.SerializeToString())
                result = apply_rewritings_to_model(base, serialized)
                if external:
                    load_external_data_for_model(target, str(root / "target"))
                for condition in (False, True):
                    self.assertEquivalent(
                        target,
                        result,
                        {
                            "condition": numpy.array(condition),
                            "x": numpy.array([-2, 3], numpy.float32),
                        },
                    )

    def test_added_subgraphs_reject_cross_directory_external_extensions(self):
        base, target = _model(), _model()
        for model in (base, target):
            model.graph.input.append(oh.make_tensor_value_info("condition", TensorProto.BOOL, []))
        branch = oh.make_graph(
            [oh.make_node("Neg", ["x"], ["z"])],
            "branch",
            [],
            [oh.make_tensor_value_info("z", TensorProto.FLOAT, [2])],
        )
        encoded = onnx.EncodedValueProto(
            name="unused",
            affine=onnx.AffineLayoutProto(
                storage_type=TensorProto.UINT8,
                scale=onh.from_array(numpy.array(1, numpy.float32)),
            ),
            logical_type=oh.make_tensor_type_proto(TensorProto.FLOAT, [2]),
            data_location=TensorProto.EXTERNAL,
            external_data=[
                onnx.StringStringEntryProto(key="location", value="weights.bin"),
                onnx.StringStringEntryProto(key="length", value="2"),
            ],
        )
        branch.extended_initializer.append(onnx.ExtendedInitializerProto(encoded_value=encoded))
        target.graph.node.clear()
        target.graph.node.append(
            oh.make_node("If", ["condition"], ["y"], then_branch=branch, else_branch=branch)
        )
        with self.assertRaisesRegex(ValueError, "external encoded declarations"):
            diff_models(base, target, base_dir="base", target_dir="target")

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
