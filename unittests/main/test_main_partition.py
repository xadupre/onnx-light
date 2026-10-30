# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests metadata-only partitioning from the command line."""

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


def _repeated_model():
    """Builds three distinct weighted layers following a shape-driven reshape."""
    nodes = [oh.make_node("Reshape", ["X", "shape"], ["flat"])]
    initializers = [onh.from_array(numpy.array([1, 16], dtype=numpy.int64), name="shape")]
    previous = "flat"
    for index in range(3):
        weight = f"W{index}"
        product = f"product{index}"
        output = "Y" if index == 2 else f"hidden{index}"
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
    def _save_external(self, directory):
        """Writes external weights and an independently stored tiny shape tensor."""
        source = directory / "model.onnx"
        save(
            _repeated_model(),
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
            self.assertEqual([node.op_type for node in layers[0].node], ["MatMul", "Relu"])
            self.assertEqual(sum(node.domain == "ai.rt" for node in result.graph.node), 3)
            self.assertIn("1 layer function(s)", text.getvalue())
            self.assertIn("3 call site(s)", text.getvalue())
            self.assertIn("ai.rt::Layer", text.getvalue())
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
            source = self._save_external(directory)
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
            self.assertEqual(len(functions[0].node), 7)
            self.assertEqual(len(result.graph.node), 1)
            self.assertEqual(result.graph.node[0].op_type, functions[0].name)
            self.assertIn("1 partition function(s)", text.getvalue())
            self.assertIn("1 call site(s)", text.getvalue())
            feeds = {"X": numpy.arange(16, dtype=numpy.float32).reshape(2, 8) / 8 - 1}
            expected = ReferenceEvaluator(load(source, load_external_data=True)).run(None, feeds)
            actual = ReferenceEvaluator(load(output, load_external_data=True)).run(None, feeds)
            numpy.testing.assert_allclose(actual[0], expected[0], rtol=1e-6, atol=1e-6)
            self.assertEqual(source.read_bytes(), original_bytes)
            self.assertEqual((directory / "weights.bin").read_bytes(), weights_before)

    def test_mincut_respects_transfer_and_group_size_limits_without_reading_weights(self):
        for options, expected_groups in (
            (["--max-boundary-bytes", "127"], 0),
            (["--max-boundary-bytes", "128"], 1),
            (["--min-group-nodes", "2", "--max-group-nodes", "2"], 3),
        ):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                source = self._save_external(directory)
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
                self.assertEqual(len(weights), 3)
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

    def test_extracts_layers_with_skipped_inline_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "model.onnx"
            output = directory / "layers.onnx"
            save(_repeated_model(), source)
            original = source.read_bytes()
            with redirect_stdout(StringIO()):
                main(["partition", "layer", str(source), "-o", str(output)])
            result = load(output)
            self.assertEqual(sum(node.domain == "ai.rt" for node in result.graph.node), 3)
            weights = [
                tensor for tensor in result.graph.initializer if tensor.name.startswith("W")
            ]
            self.assertEqual(len(weights), 3)
            self.assertTrue(all(not weight.raw_data for weight in weights))
            self.assertEqual(source.read_bytes(), original)

    def test_skipped_weights_keep_parallel_projections_distinct(self):
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
                calls = [node for node in result.graph.node if node.domain == "ai.rt"]
                self.assertEqual(len(calls), 3)
                self.assertEqual(len(result.graph.initializer), 0 if constants else 3)
                self.assertEqual({str(node.input[1]) for node in calls}, {"W0", "W1", "W2"})

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
