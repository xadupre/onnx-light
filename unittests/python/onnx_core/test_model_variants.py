# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests native model deltas and graph-free file-backed variants."""

import hashlib
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


class TestModelVariants(ExtTestCase):
    def assertEquivalent(self, expected, actual, feeds=None):
        if feeds is None:
            feeds = {"x": numpy.array([-2, 3], dtype=numpy.float32)}
        self.assertEqual(
            [(node.op_type, list(node.input), list(node.output)) for node in expected.graph.node],
            [(node.op_type, list(node.input), list(node.output)) for node in actual.graph.node],
        )
        self.assertEqual(
            [tensor.name for tensor in expected.graph.initializer],
            [tensor.name for tensor in actual.graph.initializer],
        )
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

    def test_cpu_cuda_files_relocate_without_target_exports(self):
        canonical = _model()
        cpu = _model()
        cpu.graph.node[0].output.clear()
        cpu.graph.node[0].output.append("cpu_result")
        cpu.graph.node.append(oh.make_node("Identity", ["cpu_result"], ["y"]))
        cuda = _model()
        cuda.graph.node[0].input.clear()
        cuda.graph.node[0].input.extend(["w", "x"])
        with TemporaryDirectory() as folder:
            root = Path(folder)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            base = artifacts / "model.onnx"
            onnx.save(canonical, base)
            for name, target in [("cpu", cpu), ("cuda", cuda)]:
                target_file = artifacts / f"target.{name}.onnx"
                onnx.save(target, target_file)
                envelope = onnx.save_model_variant(
                    base, target_file, artifacts / f"model.{name}.onnx", name=name
                )
                self.assertFalse(envelope.HasField("graph"))
                self.assertEqual(list(envelope.rewritings), [name])
                self.assertTrue(envelope.rewritings[name].HasField("hash_before"))
                self.assertTrue(envelope.rewritings[name].HasField("hash_after"))
                if name == "cpu":
                    self.assertFalse(envelope.rewritings[name].rewrites[0].added_initializers)
                target_file.unlink()
            moved = root / "moved"
            artifacts.rename(moved)
            self.assertEquivalent(cpu, onnx.load(moved / "model.cpu.onnx"))
            self.assertEquivalent(cuda, onnx.load(moved / "model.cuda.onnx"))
            feeds = {"x": numpy.array([-2, 3], numpy.float32)}
            canonical_output = ReferenceEvaluator(canonical).run(None, feeds)[0]
            for name in ("cpu", "cuda"):
                resolved = onnx.load(moved / f"model.{name}.onnx")
                self.assertEqualArray(
                    canonical_output, ReferenceEvaluator(resolved).run(None, feeds)[0]
                )
            raw = onnx.load(moved / "model.cpu.onnx", resolve_references=False)
            self.assertEqual(raw.base_model, "model.onnx")
            self.assertFalse(raw.HasField("graph"))

    def test_initializer_replacement_order_and_node_moves(self):
        def make(order, values, nodes):
            return oh.make_model(
                oh.make_graph(
                    nodes,
                    "g",
                    [oh.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
                    [oh.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
                    initializer=[
                        onh.from_array(numpy.array(values[name], dtype=numpy.float32), name)
                        for name in order
                    ],
                ),
                opset_imports=[oh.make_opsetid("", 18)],
            )

        base = make(
            ["w", "b"],
            {"w": [1, 2], "b": [3, 4]},
            [
                oh.make_node("Add", ["x", "w"], ["a"]),
                oh.make_node("Mul", ["x", "b"], ["b_out"]),
                oh.make_node("Add", ["a", "b_out"], ["y"]),
            ],
        )
        target = make(
            ["b", "w"],
            {"w": [5, 6], "b": [3, 4]},
            [
                oh.make_node("Mul", ["x", "b"], ["b_out"]),
                oh.make_node("Add", ["x", "w"], ["a"]),
                oh.make_node("Add", ["a", "b_out"], ["y"]),
            ],
        )
        plan = diff_models(base, target)
        self.assertEqual([t.name for t in plan.rewrites[0].added_initializers], ["w"])
        self.assertTrue(plan.moved_nodes)
        serialized = RewritingProto()
        serialized.ParseFromString(plan.SerializeToString())
        self.assertTrue(plan.Equals(serialized))
        self.assertEquivalent(target, apply_rewritings_to_model(base, serialized))
        self.assertEqual([t.name for t in base.graph.initializer], ["w", "b"])
        invalid = RewritingProto()
        invalid.CopyFrom(plan)
        invalid.rewrites[0].initializer_order.clear()
        invalid.rewrites[0].initializer_order.extend(["w", "w"])
        with self.assertRaisesRegex(ValueError, "duplicate initializer order"):
            apply_rewritings_to_model(base, invalid)

    def test_external_shards_content_comparison_and_subdirectory_base(self):
        with TemporaryDirectory() as folder:
            root = Path(folder) / "artifacts"
            root.mkdir()
            weights = root / "base"
            weights.mkdir()
            canonical = _model()
            onnx.save(
                canonical,
                weights / "model.onnx",
                save_as_external_data=True,
                location="weights.bin",
                size_threshold=0,
            )
            target = _model("Mul")
            onnx.save(
                target,
                root / "target.onnx",
                save_as_external_data=True,
                location="target.bin",
                size_threshold=0,
            )
            envelope = onnx.save_model_variant(
                weights / "model.onnx", root / "target.onnx", root / "cuda.onnx", name="cuda"
            )
            self.assertFalse(envelope.rewritings["cuda"].rewrites[0].added_initializers)
            (root / "target.onnx").unlink()
            (root / "target.bin").unlink()
            raw = onnx.load(root / "cuda.onnx")
            entries = {entry.key: entry.value for entry in raw.graph.initializer[0].external_data}
            self.assertEqual(entries["location"], "base/weights.bin")
            self.assertEqual(entries["offset"], "0")
            self.assertEqual(entries["length"], "8")
            self.assertEquivalent(target, onnx.load(root / "cuda.onnx", load_external_data=True))
            changed = _model("Mul", (7, 8))
            onnx.save(
                changed,
                root / "changed.onnx",
                save_as_external_data=True,
                location="weights.bin",
                size_threshold=0,
            )
            envelope = onnx.save_model_variant(
                weights / "model.onnx",
                root / "changed.onnx",
                root / "changed.variant.onnx",
                name="changed",
            )
            self.assertEqual(
                len(envelope.rewritings["changed"].rewrites[0].added_initializers), 1
            )
            (root / "changed.onnx").unlink()
            self.assertEquivalent(
                changed, onnx.load(root / "changed.variant.onnx", load_external_data=True)
            )
            moved = root.parent / "relocated"
            root.rename(moved)
            self.assertEquivalent(
                changed, onnx.load(moved / "changed.variant.onnx", load_external_data=True)
            )

    def test_nested_graph_edits_and_optional_output_slots(self):
        def make(op, weight=None, nested=False):
            branches = {
                name: oh.make_graph(
                    [oh.make_node(op, ["x"] if weight is None else ["x", weight], ["branch_y"])],
                    name,
                    [],
                    [oh.make_tensor_value_info("branch_y", TensorProto.FLOAT, [2])],
                )
                for name in ("then_branch", "else_branch")
            }
            if nested:
                branches = {
                    name: oh.make_graph(
                        [
                            oh.make_node(
                                "If",
                                ["condition"],
                                ["nested_y"],
                                then_branch=branch,
                                else_branch=branch,
                            )
                        ],
                        name,
                        [],
                        [oh.make_tensor_value_info("nested_y", TensorProto.FLOAT, [2])],
                    )
                    for name, branch in branches.items()
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

        base, target = make("Neg"), make("Relu")
        plan = diff_models(base, target)
        self.assertEqual(len(plan.rewrites), 2)
        self.assertTrue(all(edit.graph_path for edit in plan.rewrites))
        feeds = {"condition": numpy.array(True), "x": numpy.array([-2, 3], numpy.float32)}
        self.assertEquivalent(target, apply_rewritings_to_model(base, plan), feeds)

        for nested in (False, True):
            base, target = make("Add", "old_w", nested), make("Mul", "new_w", nested)
            plan = diff_models(base, target)
            self.assertEqual([t.name for t in plan.rewrites[0].added_initializers], ["new_w"])
            replayed = apply_rewritings_to_model(base, plan)
            for condition in (False, True):
                feeds["condition"] = numpy.array(condition)
                self.assertEquivalent(target, replayed, feeds)

        base = _model()
        target = _model()
        target.graph.node.clear()
        target.graph.node.append(oh.make_node("Dropout", ["x"], ["y", ""]))
        replayed = apply_rewritings_to_model(base, diff_models(base, target))
        self.assertEqual(list(replayed.graph.node[0].output), ["y", ""])

    def test_function_replacement_addition_and_removal(self):
        def make(op, unused=False):
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
            if unused:
                model.functions.append(
                    oh.make_function(
                        "local",
                        "Unused",
                        ["a"],
                        ["b"],
                        [oh.make_node("Neg", ["a"], ["b"])],
                        [oh.make_opsetid("", 18)],
                    )
                )
            return model

        base, target = make("Neg", unused=True), make("Relu")
        plan = diff_models(base, target)
        self.assertEqual(list(plan.replaced_function_ids), ["local:F"])
        self.assertEqual(list(plan.removed_function_ids), ["local:Unused"])
        result = apply_rewritings_to_model(base, plan)
        self.assertEqual([f.name for f in result.functions], ["F"])
        self.assertEquivalent(target, result)
        extra = make("Relu", unused=True)
        self.assertEquivalent(
            extra, apply_rewritings_to_model(target, diff_models(target, extra))
        )
        no_functions = _model()
        no_functions.opset_import.append(oh.make_opsetid("local", 1))
        removed = apply_rewritings_to_model(base, diff_models(base, no_functions))
        self.assertFalse(removed.functions)
        self.assertEquivalent(no_functions, removed)

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

    def test_missing_modified_cyclic_and_unsafe_bases(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            base = root / "model.onnx"
            variant = root / "cpu.onnx"
            onnx.save(_model(), base)
            envelope = onnx.save_model_variant(base, _model("Sub"), variant, name="cpu")
            original = base.read_bytes()
            onnx.save(_model("Mul"), base)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                onnx.load(variant)
            base.unlink()
            with self.assertRaises(FileNotFoundError):
                onnx.load(variant)
            base.write_bytes(original)
            for location in ("../model.onnx", "/model.onnx", "C:\\model.onnx"):
                envelope.base_model = location
                onnx.save(envelope, variant)
                with self.subTest(location=location), self.assertRaises(ValueError):
                    onnx.load(variant)
            envelope.base_model = "link.onnx"
            (root / "link.onnx").symlink_to(base)
            onnx.save(envelope, variant)
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                onnx.load(variant)
            envelope.base_model = "cpu.onnx"
            onnx.save(envelope, variant)
            with self.assertRaisesRegex(ValueError, "cycle"):
                onnx.load(variant)
            envelope.base_model = "other.onnx"
            onnx.save(envelope, variant)
            envelope.base_model = "cpu.onnx"
            onnx.save(envelope, root / "other.onnx")
            with self.assertRaisesRegex(ValueError, "cycle"):
                onnx.load(variant)
            envelope.base_model = "model.onnx"
            envelope.graph.CopyFrom(_model().graph)
            onnx.save(envelope, variant)
            with self.assertRaisesRegex(ValueError, "must not contain a graph"):
                onnx.load(variant)

    def test_reference_chain_and_bytes_require_source_path(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            base, first, second = [
                root / name for name in ("model.onnx", "cpu.onnx", "cuda.onnx")
            ]
            onnx.save(_model(), base)
            onnx.save_model_variant(base, _model("Mul"), first, name="cpu")
            envelope = onnx.save_model_variant(first, _model("Sub"), second, name="cuda")
            self.assertEqual(
                envelope.base_model_sha256, hashlib.sha256(first.read_bytes()).hexdigest()
            )
            self.assertEquivalent(_model("Sub"), onnx.load(second))
            with self.assertRaisesRegex(ValueError, "requires its file path"):
                onnx.load(second.read_bytes())
            with self.assertRaisesRegex(ValueError, "binary protobuf"):
                onnx.save(envelope, root / "variant.textproto")

    def test_reference_depth_limit(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            previous = root / "model.onnx"
            onnx.save(_model(), previous)
            plan = diff_models(_model(), _model())
            for index in range(33):
                envelope = ModelProto()
                envelope.base_model = previous.name
                envelope.base_model_sha256 = hashlib.sha256(previous.read_bytes()).hexdigest()
                envelope.add_rewriting("cpu", plan)
                previous = root / f"variant{index}.onnx"
                onnx.save(envelope, previous)
            with self.assertRaisesRegex(ValueError, "depth exceeds 32"):
                onnx.load(previous)

    def test_variant_edits_cannot_bypass_path_and_range_validation(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            base, variant = root / "model.onnx", root / "cpu.onnx"
            onnx.save(_model(), base)
            envelope = onnx.save_model_variant(base, _model("Mul", (7, 8)), variant, name="cpu")
            plan = envelope.rewritings["cpu"]
            weight = plan.rewrites[0].added_initializers[0]
            weight.ClearField("raw_data")
            weight.data_location = TensorProto.EXTERNAL
            for location, offset in [
                ("../outside.bin", "0"),
                ("weights.bin", "-1"),
                ("weights.bin", str(2**63 - 1)),
            ]:
                weight.external_data.clear()
                weight.external_data.add(key="location", value=location)
                weight.external_data.add(key="offset", value=offset)
                weight.external_data.add(key="length", value="8")
                envelope.clear_rewritings()
                envelope.add_rewriting("cpu", plan)
                onnx.save(envelope, variant)
                with (
                    self.subTest(location=location, offset=offset),
                    self.assertRaises(ValueError),
                ):
                    onnx.load(variant)

    def test_unchanged_encoded_external_payload_rebases_without_copying(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            storage = root / "base"
            storage.mkdir()
            (storage / "packed.bin").write_bytes(b"padding" + b"\x21")
            encoded = onnx.EncodedValueProto(
                name="packed",
                logical_type=oh.make_tensor_type_proto(TensorProto.FLOAT, [2]),
                affine=onnx.AffineLayoutProto(
                    storage_type=TensorProto.INT4,
                    scale=onh.from_array(numpy.array(1, numpy.float32)),
                ),
                data_location=TensorProto.EXTERNAL,
            )
            encoded.external_data.add(key="location", value="packed.bin")
            encoded.external_data.add(key="offset", value="7")
            encoded.external_data.add(key="length", value="1")
            base, target = _model(), _model("Mul")
            for model in (base, target):
                model.graph.extended_initializer.add().encoded_value.CopyFrom(encoded)
            onnx.save(base, storage / "model.onnx")
            onnx.save_model_variant(
                storage / "model.onnx", target, root / "cpu.onnx", name="cpu", target_dir=storage
            )
            loaded = onnx.load(root / "cpu.onnx")
            payload = loaded.graph.extended_initializer[0].encoded_value
            entries = {entry.key: entry.value for entry in payload.external_data}
            self.assertEqual(
                entries, {"location": "base/packed.bin", "offset": "7", "length": "1"}
            )
            self.assertFalse(payload.raw_data)

    def test_multiple_shards_keep_nonzero_ranges_and_unchanged_payloads(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            expected = _model()
            expected.graph.initializer.append(
                onh.from_array(numpy.array([3, 4], numpy.float32), "bias")
            )
            expected.graph.node[0].output.clear()
            expected.graph.node[0].output.append("middle")
            expected.graph.node.append(oh.make_node("Add", ["middle", "bias"], ["y"]))
            base = ModelProto()
            base.CopyFrom(expected)
            for index, tensor in enumerate(base.graph.initializer):
                offset = 8 * (index + 1)
                location = f"shard{index}.bin"
                (root / location).write_bytes(bytes(offset) + tensor.raw_data)
                tensor.ClearField("raw_data")
                tensor.data_location = TensorProto.EXTERNAL
                tensor.external_data.add(key="location", value=location)
                tensor.external_data.add(key="offset", value=str(offset))
                tensor.external_data.add(key="length", value="8")
            onnx.save(base, root / "model.onnx")
            expected.graph.node[0].op_type = "Mul"
            envelope = onnx.save_model_variant(
                root / "model.onnx", expected, root / "cuda.onnx", name="cuda"
            )
            self.assertFalse(envelope.rewritings["cuda"].rewrites[0].added_initializers)
            metadata = onnx.load(root / "cuda.onnx")
            self.assertEqual(len(metadata.external_tensors()), 2)
            for index, tensor in enumerate(metadata.graph.initializer):
                values = {entry.key: entry.value for entry in tensor.external_data}
                self.assertEqual(values["offset"], str(8 * (index + 1)))
                self.assertEqual(values["length"], "8")
            self.assertEquivalent(
                expected, onnx.load(root / "cuda.onnx", load_external_data=True)
            )

    def test_large_metadata_only_delta_does_not_open_payloads(self):
        def make(count, last):
            weight = TensorProto(
                name="huge",
                data_type=TensorProto.FLOAT,
                dims=[2**34],
                data_location=TensorProto.EXTERNAL,
            )
            weight.external_data.add(key="location", value="does-not-exist.bin")
            weight.external_data.add(key="offset", value="4096")
            weight.external_data.add(key="length", value=str(2**36))
            nodes = [
                oh.make_node(
                    last if i == count - 1 else "Neg",
                    ["x" if i == 0 else f"v{i}"],
                    ["y" if i == count - 1 else f"v{i + 1}"],
                )
                for i in range(count)
            ]
            return oh.make_model(
                oh.make_graph(
                    nodes,
                    "large",
                    [oh.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
                    [oh.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
                    initializer=[weight],
                ),
                opset_imports=[oh.make_opsetid("", 18)],
            )

        base, target = make(3000, "Neg"), make(3000, "Relu")
        plan = diff_models(base, target)
        self.assertEqual(len(plan.rewrites), 1)
        self.assertEqual(list(plan.rewrites[0].matched_nodes), [2999])
        self.assertFalse(plan.rewrites[0].added_initializers)
        self.assertLess(len(plan.SerializeToString()), 1024)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "model.onnx"
            onnx.save(base, path)
            onnx.save_model_variant(path, target, Path(folder) / "cpu.onnx", name="cpu")
            result = onnx.load(Path(folder) / "cpu.onnx")
            self.assertEqual(len(result.graph.node), 3000)
            self.assertEqual(result.graph.node[-1].op_type, "Relu")
            self.assertFalse(result.graph.initializer[0].raw_data)


if __name__ == "__main__":
    unittest.main()
