"""Tests graph optimization with deferred weights."""

import gc
import importlib.util
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh
from onnx_light.ext_test_case import import_or_skip
from onnx_light.onnx import TensorProto, load, save
from onnx_light.onnx_core.graph_builder import ConstantFoldingOptions

optim = import_or_skip("onnx_light.onnx_core.optimization")
ReferenceEvaluator = import_or_skip("onnx_light.onnx.reference", "ReferenceEvaluator")
HAS_KERNELS = importlib.util.find_spec("onnx_light.onnx_py._onnxpykernels") is not None


def _value(name, shape):
    return oh.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _model(nodes, inputs, outputs, initializers):
    return oh.make_model(
        oh.make_graph(nodes, "deferred_optimization", inputs, outputs, initializers),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=10,
    )


def _fold_model(op_type="Transpose"):
    data = numpy.arange(4096, dtype=numpy.float32).reshape(64, 64)
    model = _model(
        [oh.make_node(op_type, ["w"], ["y"])],
        [],
        [_value("y", [64, 64])],
        [onh.from_array(data, "w"), onh.from_array(numpy.full_like(data, 7), "untouched")],
    )
    return model, data


def _matmul_model(size, count, transpose):
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
        [onh.from_array(numpy.ones((size, size), dtype=numpy.float32), "w")],
    )


class TestDeferredOptimization(unittest.TestCase):
    def test_small_constants_enable_value_dependent_rewrites(self):
        feeds = {
            "x": numpy.array([0.25, 0.75], dtype=numpy.float32),
            "y": numpy.array([2, 4], dtype=numpy.float32),
        }
        for external in (False, True):
            for value in (1.0, 2.0):
                with (
                    self.subTest(external=external, value=value),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    path = Path(directory) / "model.onnx"
                    original = _model(
                        [
                            oh.make_node("Sub", ["one", "x"], ["s"]),
                            oh.make_node("Mul", ["s", "y"], ["z"]),
                        ],
                        [_value("x", [2]), _value("y", [2])],
                        [_value("z", [2])],
                        [onh.from_array(numpy.array([value], dtype=numpy.float32), "one")],
                    )
                    save(original, path, save_as_external_data=external, size_threshold=0)
                    eager = load(path, load_external_data=external)
                    deferred = load(path, defer_weights=True)
                    self.assertTrue(deferred.is_weight_loaded("one"))
                    results = []
                    for source in (eager, deferred):
                        builder = optim.GraphBuilder(source)
                        plan = optim.GraphGraph(builder, ["Sub1Mul"]).optimize()
                        result = builder.to_model()
                        results.append(
                            (
                                [edit.pattern_name for edit in plan.rewrites],
                                [node.op_type for node in result.graph.node],
                            )
                        )
                        numpy.testing.assert_allclose(
                            ReferenceEvaluator(result).run(None, feeds)[0],
                            ReferenceEvaluator(eager).run(None, feeds)[0],
                        )
                        if source is deferred and value == 1.0:
                            replayed = (
                                optim.GraphBuilder(deferred).apply_rewritings(plan).to_model()
                            )
                            restored = builder.undo_rewritings(plan).to_model()
                            self.assertEqual(
                                [node.op_type for node in replayed.graph.node], results[-1][1]
                            )
                            self.assertEqual(
                                [node.op_type for node in restored.graph.node], ["Sub", "Mul"]
                            )
                            for replay in (replayed, restored):
                                numpy.testing.assert_allclose(
                                    ReferenceEvaluator(replay).run(None, feeds)[0],
                                    ReferenceEvaluator(eager).run(None, feeds)[0],
                                )
                    self.assertEqual(results[0], results[1])
                    self.assertEqual("Sub1Mul" in results[1][0], value == 1.0)

    def test_small_shape_initializer_is_available_to_shape_inference(self):
        model = _model(
            [oh.make_node("Reshape", ["x", "shape"], ["y"])],
            [_value("x", [6])],
            [_value("y", None)],
            [onh.from_array(numpy.array([2, 3], dtype=numpy.int64), "shape")],
        )
        for external in (False, True):
            with self.subTest(external=external), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.onnx"
                save(model, path, save_as_external_data=external, size_threshold=0)
                source = load(path, defer_weights=True)
                self.assertTrue(source.is_weight_loaded("shape"))
                builder = optim.GraphBuilder(source)
                self.assertEqual(list(builder.get_shape("y").shape), [2, 3])

    @unittest.skipUnless(HAS_KERNELS, "Requires the compiled kernels extension.")
    def test_canonicalization_skips_unloaded_inline_weights(self):
        weight = onh.from_array(numpy.arange(2048, dtype=numpy.float32), "w")
        for pattern in ("ConstantToInitializer", "InitializerUnsqueezeCast"):
            with self.subTest(pattern=pattern), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.onnx"
                if pattern == "ConstantToInitializer":
                    model = _model(
                        [oh.make_node("Constant", [], ["y"], value=weight)],
                        [],
                        [_value("y", [2048])],
                        [],
                    )
                else:
                    model = _model(
                        [
                            oh.make_node("Unsqueeze", ["w", "axes"], ["u"]),
                            oh.make_node("Cast", ["u"], ["c"], to=TensorProto.FLOAT),
                            oh.make_node("Add", ["x", "c"], ["y"]),
                        ],
                        [_value("x", [1, 2048])],
                        [_value("y", [1, 2048])],
                        [weight, onh.from_array(numpy.array([0], dtype=numpy.int64), "axes")],
                    )
                save(model, path)
                options = ConstantFoldingOptions()
                options.enabled = False
                eager_builder = optim.GraphBuilder(load(path))
                eager_plan = optim.GraphGraph(eager_builder, [pattern]).optimize(
                    folding_options=options
                )
                self.assertIn(pattern, [edit.pattern_name for edit in eager_plan.rewrites])
                source = load(path, defer_weights=True)
                self.assertFalse(source.is_weight_loaded("w"))
                builder = optim.GraphBuilder(source)
                plan = optim.GraphGraph(builder, [pattern]).optimize(folding_options=options)
                self.assertNotIn(pattern, [edit.pattern_name for edit in plan.rewrites])
                self.assertFalse(builder.to_model().is_weight_loaded("w"))

    @unittest.skipUnless(HAS_KERNELS, "Requires the compiled kernels extension.")
    def test_constant_folding_loads_only_inputs_and_preserves_sources(self):
        for external in (False, True):
            for op_type in ("Transpose", "Identity"):
                with (
                    self.subTest(external=external, op_type=op_type),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    path = Path(directory) / "model.onnx"
                    model, data = _fold_model(op_type)
                    save(model, path, save_as_external_data=external, size_threshold=0)
                    source = load(path, defer_weights=True)
                    builder = optim.GraphBuilder(source)
                    self.assertFalse(source.is_weight_loaded("w"))
                    self.assertFalse(source.is_weight_loaded("untouched"))
                    self.assertFalse(builder.shapes.get("w").has_min())
                    options = ConstantFoldingOptions()
                    options.max_element_count = 1
                    self.assertEqual(builder.constant_fold(options), 0)
                    self.assertFalse(builder.to_model().is_weight_loaded("w"))
                    self.assertFalse(builder.shapes.get("w").has_min())
                    self.assertEqual(builder.constant_fold(), 1)
                    result = builder.to_model()
                    self.assertFalse(source.is_weight_loaded("w"))
                    self.assertFalse(result.is_weight_loaded("w"))
                    self.assertFalse(result.is_weight_loaded("untouched"))
                    folded = next(t for t in result.graph.initializer if t.name == "y")
                    expected = data.T if op_type == "Transpose" else data
                    backing = Path(str(path) + ".data") if external else path
                    backing.unlink()
                    numpy.testing.assert_array_equal(onh.to_array(folded), expected)

    @unittest.skipUnless(
        HAS_KERNELS and sys.platform == "linux", "Requires Linux and the compiled kernels."
    )
    def test_rejected_folds_do_not_accumulate_deferred_inputs(self):
        size = 8 * 2**20 // numpy.dtype(numpy.float32).itemsize
        names = [f"w{i}" for i in range(6)]
        values = numpy.zeros(size, dtype=numpy.float32)
        model = _model(
            [oh.make_node("Identity", [name], [f"y{i}"]) for i, name in enumerate(names)],
            [],
            [_value(f"y{i}", [size]) for i in range(len(names))],
            [onh.from_array(values, name) for name in names],
        )
        code = textwrap.dedent("""
            import os
            import resource
            import sys
            from onnx_light.onnx import load
            from onnx_light.onnx_core.graph_builder import ConstantFoldingOptions, GraphBuilder

            model = load(sys.argv[1], defer_weights=True, raw_data_threshold=0)
            builder = GraphBuilder(model)
            options = ConstantFoldingOptions()
            options.max_element_count = 1
            with open("/proc/self/statm") as stream:
                virtual_size = int(stream.read().split()[0]) * os.sysconf("SC_PAGE_SIZE")
            _, hard = resource.getrlimit(resource.RLIMIT_AS)
            limit = virtual_size + 32 * 2**20
            if hard != resource.RLIM_INFINITY and limit > hard:
                sys.exit(77)
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
            assert builder.constant_fold(options) == 0
            result = builder.to_model()
            assert all(not result.is_weight_loaded(f"w{i}") for i in range(6))
        """)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            save(model, path)
            result = subprocess.run(
                [sys.executable, "-c", code, str(path)],
                capture_output=True,
                text=True,
                timeout=60,
            )
        if result.returncode == 77:
            self.skipTest("The process hard address-space limit leaves insufficient headroom.")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(HAS_KERNELS, "Requires the compiled kernels extension.")
    def test_folding_checks_source_only_when_values_are_needed(self):
        for external in (False, True):
            for missing in (False, True):
                with (
                    self.subTest(external=external, missing=missing),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    path = Path(directory) / "model.onnx"
                    model, _ = _fold_model()
                    save(model, path, save_as_external_data=external, size_threshold=0)
                    source = load(path, defer_weights=True)
                    builder = optim.GraphBuilder(source)
                    backing = Path(str(path) + ".data") if external else path
                    if missing:
                        backing.unlink()
                    else:
                        with backing.open("ab") as stream:
                            stream.write(b"x")
                    options = ConstantFoldingOptions()
                    options.excluded_ops = {("", "Transpose")}
                    self.assertEqual(builder.constant_fold(options), 0)
                    with self.assertRaises(RuntimeError) as error:
                        builder.constant_fold()
                    if not missing:
                        self.assertIn("changed", str(error.exception))
                    self.assertFalse(builder.to_model().is_weight_loaded("w"))
                    self.assertFalse(source.is_weight_loaded("w"))

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
            for external in (False, True):
                for name in ("RepeatedLayer", "PartitionGroup", "TransposeTranspose"):
                    with self.subTest(external=external, pattern=name):
                        model = _matmul_model(
                            size,
                            count=3 if name == "RepeatedLayer" else 1,
                            transpose=name == "TransposeTranspose",
                        )
                        save(model, path, save_as_external_data=external, size_threshold=0)
                        del model
                        gc.collect()
                        source = load(path, defer_weights=True)
                        before_io = metric("/proc/self/io", "rchar")
                        before_rss = metric("/proc/self/status", "VmRSS") * 1024
                        builder = optim.GraphBuilder(source)
                        pattern = (
                            optim.PartitionGroupPattern(max_group_nodes=2)
                            if name == "PartitionGroup"
                            else name
                        )
                        plan = optim.GraphGraph(builder, [pattern]).optimize()
                        result = builder.to_model()
                        io_bytes = metric("/proc/self/io", "rchar") - before_io
                        rss_bytes = metric("/proc/self/status", "VmRSS") * 1024 - before_rss
                        self.assertIn(name, [edit.pattern_name for edit in plan.rewrites])
                        self.assertFalse(result.is_weight_loaded("w"))
                        self.assertLess(io_bytes, payload_bytes // 8, f"{name}: read {io_bytes}")
                        self.assertLess(
                            rss_bytes, 3 * payload_bytes // 4, f"{name}: allocated {rss_bytes}"
                        )


if __name__ == "__main__":
    unittest.main()
