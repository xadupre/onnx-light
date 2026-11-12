"""Tests optimization of explicitly on-demand model copies."""

import gc
import tempfile
import unittest
from pathlib import Path

import numpy
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh
from onnx_light.onnx import ModelProto, RewritingProto, TensorProto, ValueInfoProto, load, save
from onnx_light.onnx.reference import ReferenceEvaluator
from onnx_light.onnx_core.graph_builder import ConstantFoldingOptions
from onnx_light.onnx_core.optimization import GraphBuilder, GraphGraph, PartitionGroupPattern


def _value(name, shape, dtype=TensorProto.FLOAT):
    return oh.make_tensor_value_info(name, dtype, shape)


def _model(nodes, inputs, outputs, initializers=(), functions=(), opsets=None):
    return oh.make_model(
        oh.make_graph(nodes, "deferred_optimization", inputs, outputs, list(initializers)),
        functions=list(functions),
        opset_imports=opsets or [oh.make_opsetid("", 18)],
        ir_version=10,
    )


def _sub_mul_model(value=1.0, count=1):
    inputs, outputs, nodes = [], [], []
    for i in range(count):
        x, y, intermediate, output = (f"{prefix}{i}" for prefix in ("x", "y", "s", "z"))
        inputs.extend([_value(x, [2]), _value(y, [2])])
        outputs.append(_value(output, [2]))
        nodes.extend(
            [
                oh.make_node("Sub", ["one", x], [intermediate]),
                oh.make_node("Mul", [intermediate, y], [output]),
            ]
        )
    return _model(
        nodes, inputs, outputs, [onh.from_array(numpy.array([value], numpy.float32), "one")]
    )


def _matmul_model(size, count=1, value=1.0, transpose=False):
    nodes = []
    previous = "x"
    if transpose:
        nodes.extend(
            [
                oh.make_node("Transpose", ["x"], ["t"], perm=[1, 0]),
                oh.make_node("Transpose", ["t"], ["tt"], perm=[1, 0]),
            ]
        )
        previous = "tt"
    for i in range(count):
        output = "y" if i == count - 1 else f"r{i}"
        nodes.extend(
            [
                oh.make_node("MatMul", [previous, "w"], [f"m{i}"]),
                oh.make_node("Relu", [f"m{i}"], [output]),
            ]
        )
        previous = output
    return _model(
        nodes,
        [_value("x", [1, size])],
        [_value("y", [1, size])],
        [onh.from_array(numpy.full((size, size), value, dtype=numpy.float32), "w")],
    )


def _ops(model):
    return [node.op_type for node in model.graph.node]


def _metadata(info):
    return [
        (entry.key, entry.value)
        for entry in info.metadata_props
        if entry.key != "onnx_light.value_tag"
    ]


class TestDeferredOptimization(unittest.TestCase):
    def test_value_dependent_rewrite_capture_replay_undo_and_snapshot(self):
        feeds = {
            "x0": numpy.array([0.25, 0.75], numpy.float32),
            "y0": numpy.array([2, 4], numpy.float32),
        }
        for external in (False,):
            for value in (1.0, 2.0):
                with (
                    self.subTest(external=external, value=value),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    path = Path(directory) / "model.onnx"
                    save(_sub_mul_model(value), path)
                    eager = load(path)
                    eager.load_weights()
                    deferred = load(path, defer_weights=True)
                    lazy = deferred.copy_with_on_demand_weights()
                    manual_builder = GraphBuilder(deferred)
                    manual_plan = GraphGraph(manual_builder, ["Sub1Mul"]).optimize()
                    self.assertEqual(len(manual_plan.rewrites), 0)
                    self.assertFalse(deferred.is_weight_loaded("one"))

                    eager_builder = GraphBuilder(eager)
                    eager_plan = GraphGraph(eager_builder, ["Sub1Mul"]).optimize()
                    builder = GraphBuilder(lazy)
                    plan = GraphGraph(builder, ["Sub1Mul"]).optimize()
                    self.assertEqual(
                        [edit.pattern_name for edit in plan.rewrites],
                        [edit.pattern_name for edit in eager_plan.rewrites],
                    )
                    optimized = builder.to_model()
                    self.assertEqual(_ops(optimized), _ops(eager_builder.to_model()))
                    self.assertFalse(deferred.is_weight_loaded("one"))
                    self.assertFalse(lazy.is_weight_loaded("one"))

                    if value == 1.0:
                        self.assertEqual(_ops(optimized), ["Mul", "Sub"])
                        direct_replay = GraphBuilder(lazy).apply_rewritings(plan).to_model()
                        eager_replay = GraphBuilder(eager).apply_rewritings(plan).to_model()
                        lazy_replay = GraphBuilder(lazy).apply_rewritings(eager_plan).to_model()
                        self.assertEqual(_ops(direct_replay), ["Mul", "Sub"])
                        self.assertEqual(_ops(eager_replay), ["Mul", "Sub"])
                        self.assertEqual(_ops(lazy_replay), ["Mul", "Sub"])
                        self.assertEqual(
                            _ops(builder.undo_rewritings(plan).to_model()), ["Sub", "Mul"]
                        )
                        uncaptured = RewritingProto()
                        uncaptured.ParseFromString(plan.SerializeToString())
                        uncaptured.hash_before = None
                        uncaptured.hash_after = None
                        captured = GraphBuilder(lazy).capture_rewritings(uncaptured)
                        loaded_plan = RewritingProto()
                        loaded_plan.ParseFromString(captured.SerializeToString())
                        replayed = GraphBuilder(lazy).apply_rewritings(loaded_plan).to_model()
                        self.assertEqual(_ops(replayed), ["Mul", "Sub"])
                        expected = ReferenceEvaluator(eager).run(None, feeds)[0]
                        for result in (
                            optimized,
                            direct_replay,
                            eager_replay,
                            lazy_replay,
                            replayed,
                        ):
                            numpy.testing.assert_allclose(
                                ReferenceEvaluator(result).run(None, feeds)[0], expected
                            )

                        backing = Path(str(path) + ".data") if external else path
                        backing.unlink()
                        del lazy, deferred
                        gc.collect()
                        restored = builder.undo_rewritings(loaded_plan).to_model()
                        self.assertEqual(_ops(restored), ["Sub", "Mul"])
                        restored.load_weights()
                        numpy.testing.assert_array_equal(
                            onh.to_array(restored.graph.initializer[0]),
                            numpy.array([1], numpy.float32),
                        )
                    else:
                        self.assertEqual(_ops(optimized), ["Sub", "Mul"])

    def test_parallel_matching_uses_shared_lazy_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            save(_sub_mul_model(count=32), path)
            results = []
            for model in (
                load(path),
                load(path, defer_weights=True).copy_with_on_demand_weights(),
            ):
                builder = GraphBuilder(model)
                plan = GraphGraph(builder, ["Sub1Mul"]).optimize(parallel=True)
                results.append(
                    ([edit.pattern_name for edit in plan.rewrites], _ops(builder.to_model()))
                )
            self.assertEqual(results[0], results[1])
            self.assertEqual(results[1][0].count("Sub1Mul"), 32)

    def test_small_integer_shapes_and_numeric_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shape.onnx"
            model = _model(
                [oh.make_node("Reshape", ["x", "shape"], ["y"])],
                [_value("x", [6])],
                [_value("y", None)],
                [onh.from_array(numpy.array([2, 3], numpy.int64), "shape")],
            )
            save(model, path)
            lazy = GraphBuilder(load(path, defer_weights=True).copy_with_on_demand_weights())
            self.assertEqual(list(lazy.get_shape("y").shape), [2, 3])
            self.assertTrue(lazy.shapes.get("shape").has_value_as_shape())
            self.assertTrue(lazy.to_model().is_weight_loaded("shape"))

            for values in (
                numpy.array([-2, 3, 1], numpy.float32),
                numpy.array([numpy.nan, 1], numpy.float32),
                numpy.array([], numpy.float32),
                numpy.array([-0.0, 0.0], numpy.float32),
            ):
                with self.subTest(values=values.tolist()):
                    model = _model(
                        [], [], [_value("w", [len(values)])], [onh.from_array(values, "w")]
                    )
                    save(model, path)
                    builders = [
                        GraphBuilder(source)
                        for source in (
                            load(path),
                            load(path, defer_weights=True).copy_with_on_demand_weights(),
                        )
                    ]
                    descriptors = [builder.shapes.get("w") for builder in builders]
                    self.assertEqual(descriptors[0].has_min(), descriptors[1].has_min())
                    self.assertEqual(descriptors[0].has_max(), descriptors[1].has_max())
                    if descriptors[0].has_min():
                        self.assertEqual(descriptors[0].min(), descriptors[1].min())
                        self.assertEqual(descriptors[0].max(), descriptors[1].max())
                        self.assertEqual(
                            numpy.signbit(descriptors[0].min()),
                            numpy.signbit(descriptors[1].min()),
                        )
                        self.assertEqual(
                            numpy.signbit(descriptors[0].max()),
                            numpy.signbit(descriptors[1].max()),
                        )

    def test_constant_folding_and_canonicalization_match_eager(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            data = numpy.arange(6, dtype=numpy.float32).reshape(2, 3)
            save(
                _model(
                    [oh.make_node("Transpose", ["w"], ["y"], perm=[1, 0])],
                    [],
                    [_value("y", [3, 2])],
                    [onh.from_array(data, "w")],
                ),
                path,
            )
            for source in (
                load(path),
                load(path, defer_weights=True).copy_with_on_demand_weights(),
            ):
                builder = GraphBuilder(source)
                self.assertEqual(builder.constant_fold(), 1)
                result = builder.to_model()
                folded = next(t for t in result.graph.initializer if t.name == "y")
                numpy.testing.assert_array_equal(onh.to_array(folded), data.T)

            for equal in (False, True):
                values = [
                    numpy.array([2, 3], numpy.float32),
                    numpy.array([2, 3 if equal else 9], numpy.float32),
                ]
                for attributes in (False, True):
                    with self.subTest(equal=equal, attributes=attributes):
                        if attributes:
                            constants = [
                                oh.make_node("Constant", [], [name], value=onh.from_array(value))
                                for name, value in zip(("a", "b"), values, strict=True)
                            ]
                            initializers = []
                        else:
                            constants = []
                            initializers = [
                                onh.from_array(value, name)
                                for name, value in zip(("a", "b"), values, strict=True)
                            ]
                        nodes = [
                            *constants,
                            oh.make_node("Add", ["x", "a"], ["u"]),
                            oh.make_node("Add", ["x", "b"], ["v"]),
                            oh.make_node("Add", ["u", "v"], ["y"]),
                        ]
                        save(
                            _model(nodes, [_value("x", [2])], [_value("y", [2])], initializers),
                            path,
                        )
                        outcomes = []
                        for source in (
                            load(path),
                            load(path, defer_weights=True).copy_with_on_demand_weights(),
                        ):
                            builder = GraphBuilder(source)
                            folding = ConstantFoldingOptions()
                            folding.enabled = False
                            GraphGraph(builder, False).optimize(
                                canonicalize=True, folding_options=folding
                            )
                            result = builder.to_model()
                            outcomes.append(
                                (
                                    [
                                        (node.op_type, list(node.input))
                                        for node in result.graph.node
                                    ],
                                    [tensor.name for tensor in result.graph.initializer],
                                )
                            )
                        self.assertEqual(outcomes[0], outcomes[1])
                        removed = (2 if attributes else 1) if equal else 0
                        self.assertEqual(len(outcomes[1][0]), len(nodes) - removed)
                        if equal and attributes:
                            mixed = load(path, defer_weights=True).copy_with_on_demand_weights()
                            mixed.graph.node[0].attribute[0].t.raw_data = values[0].tobytes()
                            builder = GraphBuilder(mixed)
                            folding = ConstantFoldingOptions()
                            folding.enabled = False
                            GraphGraph(builder, False).optimize(
                                canonicalize=True, folding_options=folding
                            )
                            result = builder.to_model()
                            self.assertEqual(
                                (
                                    [
                                        (node.op_type, list(node.input))
                                        for node in result.graph.node
                                    ],
                                    [tensor.name for tensor in result.graph.initializer],
                                ),
                                outcomes[0],
                            )

    def test_nested_graph_and_function_tensor_attributes(self):
        constant_nodes = [
            oh.make_node(
                "Constant",
                [],
                ["one"],
                value=onh.from_array(numpy.ones(1, numpy.float32), "nested"),
            ),
            oh.make_node("Sub", ["one", "x"], ["s"]),
            oh.make_node("Mul", ["s", "y"], ["z"]),
        ]
        then_branch = oh.make_graph(constant_nodes, "then", [], [_value("z", [2])])
        else_branch = oh.make_graph(
            [oh.make_node("Identity", ["x"], ["z"])], "else", [], [_value("z", [2])]
        )
        function = oh.make_function(
            "local", "F", ["x", "y"], ["z"], constant_nodes, [oh.make_opsetid("", 18)]
        )
        function.value_info.extend(_value(name, [2]) for name in ("x", "y", "z"))
        model = _model(
            [
                oh.make_node(
                    "If", ["cond"], ["v"], then_branch=then_branch, else_branch=else_branch
                ),
                oh.make_node("F", ["v", "y"], ["out"], domain="local"),
            ],
            [_value("cond", [], TensorProto.BOOL), _value("x", [2]), _value("y", [2])],
            [_value("out", [2])],
            functions=[function],
            opsets=[oh.make_opsetid("", 18), oh.make_opsetid("local", 1)],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            save(model, path)
            outcomes = []
            for source in (
                load(path),
                load(path, defer_weights=True).copy_with_on_demand_weights(),
            ):
                builder = GraphBuilder(source)
                GraphGraph(builder, ["Sub1Mul"]).optimize()
                result = builder.to_model()
                then = next(
                    a.g for a in result.graph.node[0].attribute if a.name == "then_branch"
                )
                outcomes.append(
                    (
                        [node.op_type for node in then.node],
                        [node.op_type for node in result.functions[0].node],
                    )
                )
            self.assertEqual(outcomes[0], outcomes[1])
            self.assertEqual(outcomes[1], (["Mul", "Sub"], ["Mul", "Sub"]))

    def test_partition_range_metadata_and_function_reuse_match_eager(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            for value in (1.0, float("nan")):
                with self.subTest(value=value):
                    save(_matmul_model(64, count=3, value=value), path)
                    results = []
                    for source in (
                        load(path),
                        load(path, defer_weights=True).copy_with_on_demand_weights(),
                    ):
                        builder = GraphBuilder(source)
                        plan = GraphGraph(
                            builder, [PartitionGroupPattern(max_group_nodes=2)]
                        ).optimize()
                        result = builder.to_model()
                        replayed = GraphBuilder(source).apply_rewritings(plan).to_model()
                        restored = builder.undo_rewritings(plan).to_model()
                        self.assertEqual(_ops(replayed), _ops(result))
                        self.assertEqual(
                            _ops(restored), ["MatMul", "Relu", "MatMul", "Relu", "MatMul", "Relu"]
                        )
                        self.assertEqual(len(restored.functions), 0)
                        results.append(
                            (
                                [edit.pattern_name for edit in plan.rewrites],
                                _ops(result),
                                [
                                    [(info.name, _metadata(info)) for info in function.value_info]
                                    for function in result.functions
                                ],
                            )
                        )
                    self.assertEqual(results[0], results[1])
                    self.assertEqual(len(results[1][2]), 1 if numpy.isnan(value) else 3)

    def test_pending_bounds_guard_source_before_all_exports(self):
        for external in (False, True):
            with self.subTest(external=external), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.onnx"
                save(_matmul_model(64), path, save_as_external_data=external, size_threshold=0)
                backing = Path(str(path) + ".data") if external else path
                before = backing.read_bytes()
                builder = GraphBuilder(
                    load(path, defer_weights=True).copy_with_on_demand_weights()
                )
                GraphGraph(builder, [PartitionGroupPattern(max_group_nodes=2)]).optimize()
                model = builder.to_model()
                info = next(value for value in model.functions[0].value_info if value.name == "w")
                copied = ValueInfoProto()
                copied.CopyFrom(info)
                model.graph.initializer.clear()
                self.assertEqual(model.weight_names(), [])
                self.assertTrue(model.has_deferred_weights())
                nested = ModelProto()
                nested.graph.node.add().attribute.add().g.value_info.add().CopyFrom(copied)
                self.assertTrue(nested.has_deferred_weights())
                plan = RewritingProto()
                plan.added_functions.add().value_info.add().CopyFrom(copied)
                with self.assertRaises(RuntimeError):
                    save(model, backing)
                for part in (model, nested, model.functions[0], info, copied, plan):
                    with self.subTest(proto=type(part).__name__), self.assertRaises(RuntimeError):
                        part.SerializeToFile(str(backing))
                self.assertEqual(backing.read_bytes(), before)
                with backing.open("r+b") as output:
                    with self.assertRaises(RuntimeError):
                        save(model, output)
                    for part in (model, nested, model.functions[0], info, copied, plan):
                        with self.assertRaises(RuntimeError):
                            part.SerializeToFileDescriptor(output.fileno())
                        with self.assertRaises(RuntimeError):
                            part.SerializeToOstream(output)
                self.assertEqual(backing.read_bytes(), before)
                self.assertEqual(model.load_weights(), 0)
                self.assertFalse(model.has_deferred_weights())
                self.assertEqual(dict(_metadata(info))["min"], "1.000000")
                self.assertEqual(nested.load_weights(), 0)
                self.assertEqual(dict(_metadata(copied))["max"], "1.000000")
                backing.unlink()
                self.assertTrue(model.SerializeToString())
                self.assertTrue(nested.SerializeToString())

    def test_missing_and_changed_sources_fail_when_values_are_needed(self):
        for external in (False, True):
            for missing in (False, True):
                with (
                    self.subTest(external=external, missing=missing),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    path = Path(directory) / "model.onnx"
                    save(_sub_mul_model(), path, save_as_external_data=external, size_threshold=0)
                    builder = GraphBuilder(
                        load(path, defer_weights=True).copy_with_on_demand_weights()
                    )
                    backing = Path(str(path) + ".data") if external else path
                    if missing:
                        backing.unlink()
                    else:
                        backing.write_bytes(backing.read_bytes() + b"x")
                    with self.assertRaises(RuntimeError):
                        GraphGraph(builder, ["Sub1Mul"]).optimize()

    @unittest.skipUnless(Path("/proc/self/io").exists(), "Requires Linux process I/O accounting")
    def test_untouched_large_weights_are_not_read_or_allocated(self):
        def metric(path, name):
            with open(path, encoding="ascii") as stream:
                return int(
                    next(line.split()[1] for line in stream if line.startswith(name + ":"))
                )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            size = 2048
            payload_bytes = size * size * 4
            for name in ("RepeatedLayer", "PartitionGroup", "TransposeTranspose"):
                with self.subTest(pattern=name):
                    model = _matmul_model(
                        size,
                        count=3 if name == "RepeatedLayer" else 1,
                        transpose=name == "TransposeTranspose",
                    )
                    save(model, path)
                    del model
                    gc.collect()
                    source = load(path, defer_weights=True).copy_with_on_demand_weights()
                    before_io = metric("/proc/self/io", "rchar")
                    before_rss = metric("/proc/self/status", "VmRSS") * 1024
                    builder = GraphBuilder(source)
                    pattern = (
                        PartitionGroupPattern(max_group_nodes=2)
                        if name == "PartitionGroup"
                        else name
                    )
                    plan = GraphGraph(builder, [pattern]).optimize()
                    result = builder.to_model()
                    io_bytes = metric("/proc/self/io", "rchar") - before_io
                    rss_bytes = metric("/proc/self/status", "VmRSS") * 1024 - before_rss
                    self.assertIn(name, [edit.pattern_name for edit in plan.rewrites])
                    self.assertFalse(result.is_weight_loaded("w"))
                    self.assertLess(
                        io_bytes, payload_bytes // 8, f"{name}: read {io_bytes} bytes"
                    )
                    self.assertLess(
                        rss_bytes,
                        3 * payload_bytes // 4,
                        f"{name}: allocated {rss_bytes} resident bytes",
                    )


if __name__ == "__main__":
    unittest.main()
