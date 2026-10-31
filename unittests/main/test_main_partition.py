# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests weight-preserving partitioning from the command line."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy

import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh
from onnx_light.__main__ import _build_parser, main
from onnx_light.ext_test_case import ExtTestCase, import_or_skip
from onnx_light.onnx import TensorProto, load, save
from onnx_light.onnx_proto.verify import verify_model


def _repeated_model(layers=3):
    """Builds distinct weighted layers following a shape-driven reshape."""
    nodes = [oh.make_node("Reshape", ["X", "shape"], ["flat"])]
    initializers = [onh.from_array(numpy.array([1, 16], dtype=numpy.int64), name="shape")]
    previous = "flat"
    for index in range(layers):
        weight = f"W{index}"
        product = f"product{index}"
        output = "Y" if index + 1 == layers else f"hidden{index}"
        nodes.append(oh.make_node("MatMul", [previous, weight], [product]))
        nodes.append(oh.make_node("Relu", [product], [output]))
        initializers.append(
            onh.from_array((numpy.eye(16) + (index + 1) / 100).astype(numpy.float32), name=weight)
        )
        previous = output
    return oh.make_model(
        oh.make_graph(
            nodes,
            "repeated",
            [oh.make_tensor_value_info("X", TensorProto.FLOAT, [2, 8])],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
            initializers,
        ),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=10,
    )


class TestMainPartition(ExtTestCase):
    def test_both_methods_report_compact_symbolic_input_output_totals_per_call(self):
        for method, verbose in (("layer", 0), ("layer", 1), ("mincut", 0), ("mincut", 1)):
            with (
                self.subTest(method=method, verbose=verbose),
                tempfile.TemporaryDirectory() as temporary,
            ):
                directory = Path(temporary)
                source = self._save_external(directory, layers=20 if method == "mincut" else 3)
                model = load(source, load_external_data=False)
                model.graph.input[0].CopyFrom(
                    oh.make_tensor_value_info("X", TensorProto.FLOAT, ["batch", 8])
                )
                save(model, source)
                (directory / "shape.bin").write_bytes(
                    numpy.array([-1, 16], dtype=numpy.int64).tobytes()
                )
                (directory / "weights.bin").unlink()
                text = StringIO()
                output = directory / "partition.onnx"
                with redirect_stdout(text):
                    main(
                        ["partition", method, str(source), "-o", str(output), "-v", str(verbose)]
                    )
                report = text.getvalue()
                self.assertIn(f"[partition {method}] selection:", report)
                self.assertIn("inputs: logical bytes=", report)
                self.assertIn("outputs: logical bytes=", report)
                self.assertIn("transfer bytes=", report)
                self.assertIn("batch", report)
                self.assertNotIn("(constant)", report)
                self.assertNotIn("FLOAT[16,16]", report)
                self.assertNotIn("known transfer bytes=", report)
                lines = [line.strip() for line in report.splitlines()]
                call_count = sum(line.startswith("call ") for line in lines)
                self.assertGreater(call_count, 0)
                self.assertEqual(
                    sum(line.startswith("inputs: logical bytes=") for line in lines), call_count
                )
                self.assertEqual(
                    sum(line.startswith("outputs: logical bytes=") for line in lines), call_count
                )
                self.assertEqual(sum("bytes=" in line for line in lines), 2 * call_count)
                self.assertIn("not compared by the selector", report)
                self.assertIn(
                    "interval dynamic programming" if method == "mincut" else "equivalent bodies",
                    report,
                )
                result = load(output, load_external_data=False)
                self.assertGreater(len(result.functions), 0)

    def _save_external(self, directory, layers=3):
        """Writes external weights and an independently stored tiny shape tensor."""
        source = directory / "model.onnx"
        save(
            _repeated_model(layers),
            source,
            save_as_external_data=True,
            location="weights.bin",
            size_threshold=128,
        )
        model = load(source, load_external_data=False)
        shape = TensorProto()
        shape.name = "shape"
        shape.data_type = TensorProto.INT64
        shape.dims.append(2)
        shape.data_location = TensorProto.EXTERNAL
        for key, value in (("location", "shape.bin"), ("offset", "0"), ("length", "16")):
            entry = shape.external_data.add()
            entry.key = key
            entry.value = value
        (directory / "shape.bin").write_bytes(numpy.array([1, 16], dtype=numpy.int64).tobytes())
        model.graph.initializer[0].CopyFrom(shape)
        save(model, source)
        return source

    def test_extracts_functions_and_preserves_numerics_and_source_files(self):
        ReferenceEvaluator = import_or_skip("onnx_light.onnx.reference", "ReferenceEvaluator")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = self._save_external(directory)
            output = directory / "layers.onnx"
            original_bytes = source.read_bytes()
            weights_before = (directory / "weights.bin").read_bytes()
            shape_before = (directory / "shape.bin").read_bytes()
            text = StringIO()
            with redirect_stdout(text):
                main(["partition", "layer", str(source), "-o", str(output), "--verbose"])
            result = load(output, load_external_data=False)
            layers = [function for function in result.functions if function.domain == "ai.rt"]
            self.assertEqual(len(layers), 1)
            self.assertEqual(layers[0].name, "Layer_MatMul_0")
            self.assertEqual([node.op_type for node in layers[0].node], ["MatMul", "Relu"])
            self.assertEqual(sum(node.domain == "ai.rt" for node in result.graph.node), 3)
            self.assertIn("1 layer function(s)", text.getvalue())
            self.assertIn("3 call site(s)", text.getvalue())
            self.assertIn("ai.rt::Layer_MatMul_0", text.getvalue())
            feeds = {"X": numpy.arange(16, dtype=numpy.float32).reshape(2, 8) / 8 - 1}
            expected = ReferenceEvaluator(load(source, load_external_data=True)).run(None, feeds)
            actual = ReferenceEvaluator(load(output, load_external_data=True)).run(None, feeds)
            numpy.testing.assert_allclose(actual[0], expected[0], rtol=1e-6, atol=1e-6)
            self.assertEqual(source.read_bytes(), original_bytes)
            self.assertEqual((directory / "weights.bin").read_bytes(), weights_before)
            self.assertEqual((directory / "shape.bin").read_bytes(), shape_before)

    def test_reads_tiny_shapes_but_does_not_open_large_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = self._save_external(directory)
            (directory / "weights.bin").unlink()
            output = directory / "layers.onnx"
            with redirect_stdout(StringIO()):
                main(
                    [
                        "partition",
                        "layer",
                        str(source),
                        "-o",
                        str(output),
                        "--tiny-tensor-threshold",
                        "128",
                    ]
                )
            result = load(output, load_external_data=False)
            weights = [
                tensor for tensor in result.graph.initializer if tensor.name.startswith("W")
            ]
            self.assertEqual(len(weights), 3)
            for weight in weights:
                self.assertEqual(weight.data_location, TensorProto.EXTERNAL)
                self.assertFalse(weight.raw_data)
            shape = next(tensor for tensor in result.graph.initializer if tensor.name == "shape")
            numpy.testing.assert_array_equal(onh.to_array(shape), [1, 16])
            self.assertEqual(
                [
                    dimension.dim_value
                    for dimension in result.graph.output[0].type.tensor_type.shape.dim
                ],
                [1, 16],
            )

    def test_mincut_preserves_numerics_and_external_weight_files(self):
        ReferenceEvaluator = import_or_skip("onnx_light.onnx.reference", "ReferenceEvaluator")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = self._save_external(directory, layers=20)
            output = directory / "partition.onnx"
            original_bytes = source.read_bytes()
            weights_before = (directory / "weights.bin").read_bytes()
            text = StringIO()
            with redirect_stdout(text):
                main(["partition", "mincut", str(source), "-o", str(output), "--verbose"])
            result = load(output, load_external_data=False)
            functions = [
                function
                for function in result.functions
                if function.domain == "ai.rt" and function.name.startswith("PartitionGroup")
            ]
            self.assertEqual(len(functions), 1)
            self.assertEqual(functions[0].name, "PartitionGroup_MatMul_0")
            self.assertTrue(all(len(function.node) == 2 for function in functions))
            self.assertEqual(len(result.graph.node), 21)
            self.assertEqual(result.graph.node[0].op_type, "Reshape")
            self.assertIn("1 partition function(s)", text.getvalue())
            self.assertIn("constant inputs <= 1 (5% of 21 model weights", text.getvalue())
            for node in list(result.graph.node)[1:]:
                self.assertEqual(node.op_type, functions[0].name)
                self.assertEqual(sum(name.startswith("W") for name in node.input), 1)
            self.assertIn("20 call site(s)", text.getvalue())
            self.assertIn("ai.rt::PartitionGroup_MatMul_0", text.getvalue())
            feeds = {"X": numpy.arange(16, dtype=numpy.float32).reshape(2, 8) / 8 - 1}
            expected = ReferenceEvaluator(load(source, load_external_data=True)).run(None, feeds)
            actual = ReferenceEvaluator(load(output, load_external_data=True)).run(None, feeds)
            numpy.testing.assert_allclose(actual[0], expected[0], rtol=1e-6, atol=1e-6)
            self.assertEqual(source.read_bytes(), original_bytes)
            self.assertEqual((directory / "weights.bin").read_bytes(), weights_before)

    def test_mincut_keeps_over_budget_operators_outside_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = self._save_external(directory)
            (directory / "weights.bin").unlink()
            output = directory / "partition.onnx"
            text = StringIO()
            with redirect_stdout(text):
                main(["partition", "mincut", str(source), "-o", str(output)])
            result = load(output, load_external_data=False)
            self.assertIn("constant inputs <= 0 (5% of 4 model weights", text.getvalue())
            self.assertFalse(any(function.domain == "ai.rt" for function in result.functions))
            self.assertEqual(sum(node.op_type == "MatMul" for node in result.graph.node), 3)

    def test_mincut_partitions_symbolic_and_large_boundaries_without_a_default_byte_cap(self):
        ReferenceEvaluator = import_or_skip("onnx_light.onnx.reference", "ReferenceEvaluator")
        for symbolic in (False, True):
            with self.subTest(symbolic=symbolic), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                source = self._save_external(directory, layers=20)
                model = load(source, load_external_data=False)
                model.graph.input[0].CopyFrom(
                    oh.make_tensor_value_info(
                        "X", TensorProto.FLOAT, ["batch", 8] if symbolic else [4096, 8]
                    )
                )
                save(model, source)
                (directory / "shape.bin").write_bytes(
                    numpy.array([-1, 16], dtype=numpy.int64).tobytes()
                )
                original = source.read_bytes()
                weights = (directory / "weights.bin").read_bytes()
                output = directory / "partition.onnx"
                text = StringIO()
                with redirect_stdout(text):
                    main(["partition", "mincut", str(source), "-o", str(output), "-v", "1"])
                self.assertIn("1 partition function(s)", text.getvalue())
                result = load(output, load_external_data=True)
                self.assertEqual(len(result.graph.node), 21)
                self.assertTrue(result.graph.node[1].op_type.startswith("PartitionGroup_"))
                for batch in ((2, 4) if symbolic else (4096,)):
                    feeds = {"X": numpy.arange(batch * 8, dtype=numpy.float32).reshape(batch, 8)}
                    expected = ReferenceEvaluator(load(source, load_external_data=True)).run(
                        None, feeds
                    )
                    actual = ReferenceEvaluator(result).run(None, feeds)
                    self.assertEqualArray(actual[0], expected[0])
                self.assertEqual(source.read_bytes(), original)
                self.assertEqual((directory / "weights.bin").read_bytes(), weights)

    def test_mincut_respects_transfer_and_group_size_limits_without_reading_weights(self):
        for options, expected_groups in (
            (["--max-boundary-bytes", "127"], 0),
            (["--max-boundary-bytes", "128"], 1),
            (["--min-group-nodes", "2", "--max-group-nodes", "2"], 1),
        ):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                source = self._save_external(directory, layers=20)
                (directory / "weights.bin").unlink()
                output = directory / "partition.onnx"
                with redirect_stdout(StringIO()):
                    main(["partition", "mincut", str(source), "-o", str(output), *options])
                result = load(output, load_external_data=False)
                functions = [
                    function
                    for function in result.functions
                    if function.domain == "ai.rt" and function.name.startswith("PartitionGroup")
                ]
                self.assertEqual(len(functions), expected_groups)
                if "--max-group-nodes" in options:
                    self.assertTrue(all(len(function.node) == 2 for function in functions))
                weights = [
                    tensor for tensor in result.graph.initializer if tensor.name.startswith("W")
                ]
                self.assertEqual(len(weights), 20)
                self.assertTrue(
                    all(
                        weight.data_location == TensorProto.EXTERNAL and not weight.raw_data
                        for weight in weights
                    )
                )

    def test_mincut_rejects_invalid_limits_before_loading_or_overwriting(self):
        for options in (
            ["--min-group-nodes", "1"],
            ["--min-group-nodes", "4", "--max-group-nodes", "3"],
            ["--max-boundary-bytes", str(1 << 64)],
            ["--max-group-nodes", str(1 << 64)],
        ):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                output = directory / "output.onnx"
                output.write_bytes(b"existing output")
                with self.assertRaisesRegex(ValueError, "Partition limits"):
                    main(
                        [
                            "partition",
                            "mincut",
                            str(directory / "missing.onnx"),
                            "-o",
                            str(output),
                            *options,
                        ]
                    )
                self.assertEqual(output.read_bytes(), b"existing output")

    def test_missing_tiny_shape_fails_without_writing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = self._save_external(directory)
            (directory / "shape.bin").unlink()
            output = directory / "layers.onnx"
            with self.assertRaises((RuntimeError, ValueError)):
                main(["partition", "layer", str(source), "-o", str(output)])
            self.assertFalse(output.exists())

    def test_rejects_missing_graph_without_writing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "empty.onnx"
            source.write_bytes(b"")
            output = directory / "layers.onnx"
            with self.assertRaisesRegex(ValueError, "requires a model graph"):
                main(["partition", "layer", str(source), "-o", str(output)])
            self.assertFalse(output.exists())

    def test_preserves_inline_weights_and_numerics_including_in_place(self):
        ReferenceEvaluator = import_or_skip("onnx_light.onnx.reference", "ReferenceEvaluator")
        for method in ("layer", "mincut"):
            for in_place in (False, True):
                with (
                    self.subTest(method=method, in_place=in_place),
                    tempfile.TemporaryDirectory() as temporary,
                ):
                    directory = Path(temporary)
                    source = directory / "model.onnx"
                    output = source if in_place else directory / "partition.onnx"
                    model = _repeated_model(layers=20 if method == "mincut" else 3)
                    save(model, source)
                    original = source.read_bytes()
                    with redirect_stdout(StringIO()):
                        main(["partition", method, str(source), "-o", str(output)])
                    result = load(output)
                    verify_model(result)
                    self.assertGreater(
                        sum(node.domain == "ai.rt" for node in result.graph.node), 0
                    )
                    expected_weights = {
                        str(tensor.name): bytes(tensor.raw_data)
                        for tensor in model.graph.initializer
                        if tensor.name.startswith("W")
                    }
                    actual_weights = {
                        str(tensor.name): bytes(tensor.raw_data)
                        for tensor in result.graph.initializer
                        if tensor.name.startswith("W")
                    }
                    self.assertEqual(actual_weights, expected_weights)
                    feeds = {"X": numpy.arange(16, dtype=numpy.float32).reshape(2, 8) / 8 - 1}
                    expected = ReferenceEvaluator(model).run(None, feeds)
                    actual = ReferenceEvaluator(result).run(None, feeds)
                    self.assertEqualArray(actual[0], expected[0])
                    if not in_place:
                        self.assertEqual(source.read_bytes(), original)

    def test_preserves_mixed_inline_and_unloaded_external_weights(self):
        for method in ("layer", "mincut"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                source = self._save_external(directory, layers=20 if method == "mincut" else 3)
                model = load(source, load_external_data=False)
                inline = _repeated_model().graph.initializer[1]
                model.graph.initializer[1].CopyFrom(inline)
                save(model, source)
                original = source.read_bytes()
                (directory / "weights.bin").unlink()
                output = directory / "partition.onnx"
                with redirect_stdout(StringIO()):
                    main(["partition", method, str(source), "-o", str(output)])
                result = load(output, load_external_data=False)
                verify_model(result)
                weights = {
                    str(tensor.name): tensor
                    for tensor in result.graph.initializer
                    if tensor.name.startswith("W")
                }
                self.assertEqual(bytes(weights["W0"].raw_data), bytes(inline.raw_data))
                for name, weight in weights.items():
                    if name != "W0":
                        self.assertEqual(weight.data_location, TensorProto.EXTERNAL)
                        self.assertFalse(weight.raw_data)
                        expected = next(t for t in model.graph.initializer if t.name == name)
                        self.assertEqual(
                            [(entry.key, entry.value) for entry in weight.external_data],
                            [(entry.key, entry.value) for entry in expected.external_data],
                        )
                self.assertEqual(source.read_bytes(), original)

    def test_inline_weights_keep_parallel_projections_distinct(self):
        ReferenceEvaluator = import_or_skip("onnx_light.onnx.reference", "ReferenceEvaluator")
        for constants in (False, True):
            with self.subTest(constants=constants), tempfile.TemporaryDirectory() as temporary:
                weights = [
                    tensor
                    for tensor in _repeated_model().graph.initializer
                    if tensor.name.startswith("W")
                ]
                nodes = []
                for index, weight in enumerate(weights):
                    if constants:
                        weight.name = ""
                        nodes.append(oh.make_node("Constant", [], [f"W{index}"], value=weight))
                    nodes.extend(
                        [
                            oh.make_node("MatMul", ["X", f"W{index}"], [f"M{index}"]),
                            oh.make_node("Relu", [f"M{index}"], [f"Y{index}"]),
                        ]
                    )
                model = oh.make_model(
                    oh.make_graph(
                        nodes,
                        "parallel",
                        [oh.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16])],
                        [
                            oh.make_tensor_value_info(f"Y{index}", TensorProto.FLOAT, [1, 16])
                            for index in range(3)
                        ],
                        [] if constants else weights,
                    ),
                    opset_imports=[oh.make_opsetid("", 18)],
                )
                source = Path(temporary) / "model.onnx"
                output = Path(temporary) / "layers.onnx"
                save(model, source)
                with redirect_stdout(StringIO()):
                    main(["partition", "layer", str(source), "-o", str(output)])
                result = load(output)
                verify_model(result)
                calls = [node for node in result.graph.node if node.domain == "ai.rt"]
                self.assertEqual(len(calls), 3)
                self.assertEqual(len(result.graph.initializer), 0 if constants else 3)
                self.assertEqual({str(node.input[1]) for node in calls}, {"W0", "W1", "W2"})
                feeds = {"X": numpy.arange(16, dtype=numpy.float32).reshape(1, 16)}
                expected = ReferenceEvaluator(model).run(None, feeds)
                actual = ReferenceEvaluator(result).run(None, feeds)
                for expected_output, actual_output in zip(expected, actual):
                    self.assertEqualArray(actual_output, expected_output)

    def test_overwrites_existing_output_with_optional_verbose_notice(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = self._save_external(directory)
            output = directory / "existing.onnx"
            for verbose in (False, True):
                with self.subTest(verbose=verbose):
                    output.write_bytes(b"old output")
                    text = StringIO()
                    arguments = ["partition", "layer", str(source), "-o", str(output)]
                    if verbose:
                        arguments.append("--verbose")
                    with redirect_stdout(text):
                        main(arguments)
                    result = load(output, load_external_data=False)
                    self.assertEqual(sum(node.domain == "ai.rt" for node in result.graph.node), 3)
                    notice = f"[partition layer] overwriting {str(output)!r}"
                    if verbose:
                        self.assertIn(notice, text.getvalue())
                    else:
                        self.assertNotIn(notice, text.getvalue())

    def test_overwrites_input_model_in_place(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = self._save_external(Path(temporary))
            text = StringIO()
            with redirect_stdout(text):
                main(["partition", "layer", str(source), "-o", str(source), "--verbose"])
            result = load(source, load_external_data=False)
            self.assertEqual(sum(node.domain == "ai.rt" for node in result.graph.node), 3)
            self.assertIn(f"[partition layer] overwriting {str(source)!r}", text.getvalue())

    def test_accepts_another_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = self._save_external(directory)
            elsewhere = directory / "elsewhere"
            elsewhere.mkdir()
            output = elsewhere / "layers.onnx"
            with redirect_stdout(StringIO()):
                main(["partition", "layer", str(source), "-o", str(output)])
            result = load(output, load_external_data=False)
            weights = [
                tensor for tensor in result.graph.initializer if tensor.name.startswith("W")
            ]
            self.assertEqual(len(weights), 3)
            for weight in weights:
                self.assertEqual(
                    next(
                        entry.value for entry in weight.external_data if entry.key == "location"
                    ),
                    "weights.bin",
                )

    def test_saves_unchanged_model_when_no_repeated_layer_exists(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Relu", ["X"], ["Y"])],
                "plain",
                [oh.make_tensor_value_info("X", TensorProto.FLOAT, [2])],
                [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [2])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "plain.onnx"
            save(model, source)
            output = directory / "plain.layers.onnx"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "onnx_light",
                    "partition",
                    "layer",
                    str(source),
                    "-o",
                    str(output),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            self.assertIn("0 layer function(s), 0 extraction rewrite(s)", completed.stdout)
            self.assertEqual([node.op_type for node in load(output).graph.node], ["Relu"])

    def test_parser_requires_output_and_positive_threshold(self):
        for method in ("layer", "mincut"):
            for arguments in (
                ["model.onnx"],
                ["model.onnx", "-o", "output.onnx", "--tiny-tensor-threshold", "0"],
                ["model.onnx", "-o", "output.onnx", "--tiny-tensor-threshold", "-1"],
            ):
                with (
                    self.subTest(method=method, arguments=arguments),
                    redirect_stderr(StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    _build_parser().parse_args(["partition", method, *arguments])


if __name__ == "__main__":
    unittest.main()
