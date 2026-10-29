# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests the model optimization command."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os
from pathlib import Path
import tempfile
import unittest

import numpy

from onnx_light.__main__ import main
from onnx_light.ext_test_case import ExtTestCase, import_or_skip
from onnx_light.onnx import PipelineProto, TensorProto, load, save
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh


def _make_model():
    """Builds independent double-negation and double-transpose paths."""
    return oh.make_model(
        oh.make_graph(
            [
                oh.make_node("Not", ["X"], ["n"]),
                oh.make_node("Not", ["n"], ["Y"]),
                oh.make_node("Transpose", ["X"], ["t"], perm=[1, 0]),
                oh.make_node("Transpose", ["t"], ["Z"], perm=[1, 0]),
            ],
            "g",
            [oh.make_tensor_value_info("X", TensorProto.BOOL, [2, 3])],
            [
                oh.make_tensor_value_info("Y", TensorProto.BOOL, [2, 3]),
                oh.make_tensor_value_info("Z", TensorProto.BOOL, [2, 3]),
            ],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
    )


class TestMainOptimize(ExtTestCase):
    def test_registered_custom_pattern(self):
        from onnx_light.onnx_core.optimization import (
            PatternOptimization,
            register_pattern,
            register_pattern_set,
            unregister_pattern,
            unregister_pattern_set,
        )

        class CustomNotNot(PatternOptimization):
            def __init__(self):
                super().__init__(priority=1, name="CustomNotNot")

            def fast_op_type(self):
                return {"Not"}

            def match(self, graph, node):
                previous = graph.node_before(node.input[0])
                if previous is None or previous.op_type != "Not":
                    return self.no_match(node, "no preceding Not")
                return self.result([previous, node], insert_at=node)

            def apply(self, graph, nodes):
                del graph
                return [oh.make_node("Identity", [nodes[0].input[0]], list(nodes[1].output))]

        register_pattern(CustomNotNot())
        self.addCleanup(unregister_pattern, "CustomNotNot")
        register_pattern_set("CustomOptimizeGroup", ["CustomNotNot"])
        self.addCleanup(unregister_pattern_set, "CustomOptimizeGroup")
        for selection in ("CustomNotNot", "CustomOptimizeGroup", "L1,CustomNotNot"):
            with self.subTest(patterns=selection), tempfile.TemporaryDirectory() as tmp:
                source = Path(tmp) / "model.onnx"
                output = Path(tmp) / "optimized.onnx"
                save(_make_model(), source)
                main(["optimize", str(source), "-o", str(output), "--patterns", selection])
                self.assertNotIn("Not", [node.op_type for node in load(output).graph.node])

    def test_default_and_explicit_patterns(self):
        for selection, expected in (
            (None, (0, 0)),
            ("L1", (2, 2)),
            ("L1,L2", (0, 0)),
            ("NotNot", (0, 2)),
            ("TransposeTranspose", (2, 0)),
            (" NotNot, TransposeTranspose,NotNot ", (0, 0)),
            ("L1,NotNot", (0, 2)),
        ):
            with self.subTest(patterns=selection), tempfile.TemporaryDirectory() as tmp:
                source = Path(tmp) / "model.onnx"
                output = Path(tmp) / "optimized.onnx"
                save(_make_model(), source)
                before = source.read_bytes()
                args = ["optimize", str(source), "-o", str(output)]
                if selection is not None:
                    args += ["--patterns", selection]
                main(args)
                result = load(output)
                operators = [node.op_type for node in result.graph.node]
                self.assertEqual((operators.count("Not"), operators.count("Transpose")), expected)
                self.assertEqual([value.name for value in result.graph.output], ["Y", "Z"])
                self.assertEqual(source.read_bytes(), before)

    def test_external_weights_are_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "model.onnx"
            output = Path(tmp) / "optimized.onnx"
            weights = Path(tmp) / "weights.bin"
            values = numpy.arange(256, dtype=numpy.float32).reshape(16, 16)
            model = oh.make_model(
                oh.make_graph(
                    [
                        oh.make_node("Transpose", ["X"], ["t"], perm=[1, 0]),
                        oh.make_node("Transpose", ["t"], ["r"], perm=[1, 0]),
                        oh.make_node("MatMul", ["r", "W"], ["Y"]),
                    ],
                    "g",
                    [oh.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16])],
                    [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16])],
                    initializer=[onh.from_array(values, name="W")],
                ),
                opset_imports=[oh.make_opsetid("", 18)],
            )
            save(
                model, source, save_as_external_data=True, location=str(weights), size_threshold=0
            )
            before = weights.read_bytes()
            metadata = [
                (entry.key, entry.value)
                for entry in load(source, load_external_data=False)
                .graph.initializer[0]
                .external_data
            ]
            parked = Path(tmp) / "parked.bin"
            weights.rename(parked)
            main(["optimize", str(source), "-o", str(output)])
            self.assertFalse(weights.exists())
            parked.rename(weights)
            result = load(output, load_external_data=False)
            operators = [node.op_type for node in result.graph.node]
            self.assertNotIn("Transpose", operators)
            self.assertEqual(operators.count("MatMul"), 1)
            tensor = result.graph.initializer[0]
            self.assertEqual(tensor.data_location, TensorProto.EXTERNAL)
            self.assertEqual(
                [(entry.key, entry.value) for entry in tensor.external_data], metadata
            )
            self.assertEqual(weights.read_bytes(), before)
            self.assertEqual(
                {path.name for path in Path(tmp).iterdir()},
                {"model.onnx", "optimized.onnx", "weights.bin"},
            )
            loaded = load(output, load_external_data=True)
            self.assertEqualArray(onh.to_array(loaded.graph.initializer[0]), values)

    def test_optimized_model_preserves_outputs(self):
        import_or_skip("onnx_light.onnx_py._onnxpykernels")
        from onnx_light.onnx.reference import ReferenceEvaluator

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "model.onnx"
            output = Path(tmp) / "optimized.onnx"
            model = _make_model()
            save(model, source)
            main(["optimize", str(source), "-o", str(output)])
            feeds = {"X": numpy.array([[True, False, True], [False, True, False]])}
            expected = ReferenceEvaluator(model).run(None, feeds)
            actual = ReferenceEvaluator(load(output)).run(None, feeds)
            for left, right in zip(expected, actual):
                self.assertEqualArray(left, right)

    def test_inline_weights_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "model.onnx"
            output = Path(tmp) / "optimized.onnx"
            values = numpy.arange(512 * 512, dtype=numpy.float32).reshape(512, 512)
            model = oh.make_model(
                oh.make_graph(
                    [oh.make_node("MatMul", ["X", "W"], ["Y"])],
                    "g",
                    [oh.make_tensor_value_info("X", TensorProto.FLOAT, [1, 512])],
                    [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 512])],
                    initializer=[onh.from_array(values, name="W")],
                ),
                opset_imports=[oh.make_opsetid("", 18)],
            )
            save(model, source)
            main(["optimize", str(source), "-o", str(output)])
            result = load(output)
            self.assertEqualArray(onh.to_array(result.graph.initializer[0]), values)

    def test_missing_external_location_cannot_be_used_as_output(self):
        for nested in (False, True):
            with self.subTest(nested=nested), tempfile.TemporaryDirectory() as tmp:
                source = Path(tmp) / "model.onnx"
                output = Path(tmp) / "missing.bin"
                value = oh.make_tensor(
                    "W",
                    TensorProto.FLOAT,
                    [256],
                    external_data={
                        "location": f"./{output.name}" if nested else output.name,
                        "length": "1024",
                    },
                )
                branch = oh.make_graph(
                    [],
                    "branch",
                    [],
                    [oh.make_tensor_value_info("W", TensorProto.FLOAT, [256])],
                    initializer=[value],
                )
                if nested:
                    model = oh.make_model(
                        oh.make_graph(
                            [
                                oh.make_node(
                                    "If",
                                    ["condition"],
                                    ["Y"],
                                    then_branch=branch,
                                    else_branch=branch,
                                )
                            ],
                            "g",
                            [oh.make_tensor_value_info("condition", TensorProto.BOOL, [])],
                            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [256])],
                        ),
                        opset_imports=[oh.make_opsetid("", 18)],
                    )
                else:
                    model = oh.make_model(branch, opset_imports=[oh.make_opsetid("", 18)])
                save(model, source)
                before = source.read_bytes()
                with self.assertRaisesRegex(ValueError, "external-data location"):
                    main(["optimize", str(source), "-o", str(output)])
                self.assertFalse(output.exists())
                self.assertEqual(source.read_bytes(), before)

    def test_rejects_non_binary_output_suffixes(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "model.onnx"
            save(_make_model(), source)
            for suffix in (".textproto", ".pipeline", ".pipe-onnx"):
                output = Path(tmp) / f"optimized{suffix}"
                with (
                    self.subTest(suffix=suffix),
                    self.assertRaisesRegex(ValueError, "binary ONNX"),
                ):
                    main(["optimize", str(source), "-o", str(output)])
                self.assertFalse(output.exists())

    def test_small_external_shapes_in_binary_and_text_models(self):
        for suffix in (".onnx", ".textproto"):
            for length_key in (None, "length", "size"):
                with (
                    self.subTest(suffix=suffix, length_key=length_key),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    source = Path(tmp) / f"model{suffix}"
                    output = Path(tmp) / "optimized.onnx"
                    shape_path = Path(tmp) / "shape.bin"
                    shape_path.write_bytes(
                        b"skip"
                        + numpy.array([3, 2], dtype=numpy.int64).tobytes()
                        + b"trailing weights" * 1000
                    )
                    metadata = {"location": shape_path.name, "offset": "4"}
                    if length_key is not None:
                        metadata[length_key] = "16"
                    model = oh.make_model(
                        oh.make_graph(
                            [oh.make_node("Reshape", ["X", "shape"], ["Y"])],
                            "g",
                            [oh.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])],
                            [
                                oh.make_tensor_value_info("Y", TensorProto.FLOAT, None),
                                oh.make_tensor_value_info("W", TensorProto.FLOAT, [256]),
                            ],
                            initializer=[
                                oh.make_tensor(
                                    "shape", TensorProto.INT64, [2], external_data=metadata
                                ),
                                oh.make_tensor(
                                    "W",
                                    TensorProto.FLOAT,
                                    [256],
                                    external_data={
                                        "location": "missing-large.bin",
                                        "length": "1024",
                                    },
                                ),
                            ],
                        ),
                        opset_imports=[oh.make_opsetid("", 18)],
                    )
                    save(model, source)
                    main(["optimize", str(source), "-o", str(output)])
                    result = load(output)
                    self.assertEqual(
                        [
                            dim.dim_value
                            for dim in result.graph.output[0].type.tensor_type.shape.dim
                        ],
                        [3, 2],
                    )
                    tensors = {tensor.name: tensor for tensor in result.graph.initializer}
                    self.assertEqual(tensors["W"].data_location, TensorProto.EXTERNAL)
                    self.assertFalse((Path(tmp) / "missing-large.bin").exists())

    def test_small_external_read_failure_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "model.onnx"
            output = Path(tmp) / "optimized.onnx"
            shape = oh.make_tensor(
                "shape",
                TensorProto.INT64,
                [2],
                external_data={"location": "shape.bin", "length": "16"},
            )
            model = oh.make_model(
                oh.make_graph(
                    [oh.make_node("Reshape", ["X", "shape"], ["Y"])],
                    "g",
                    [oh.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])],
                    [oh.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
                    initializer=[shape],
                ),
                opset_imports=[oh.make_opsetid("", 18)],
            )
            (Path(tmp) / "shape.bin").write_bytes(b"short")
            for overrides, error in (
                ({}, RuntimeError),
                ({"offset": "-1"}, ValueError),
                ({"length": "32"}, ValueError),
            ):
                with self.subTest(metadata=overrides):
                    tensor = model.graph.initializer[0]
                    tensor.external_data.clear()
                    for key, value in {
                        "location": "shape.bin",
                        "length": "16",
                        **overrides,
                    }.items():
                        entry = tensor.external_data.add()
                        entry.key, entry.value = key, value
                    save(model, source)
                    with self.assertRaises(error):
                        main(["optimize", str(source), "-o", str(output)])
                    self.assertFalse(output.exists())

    def test_unloaded_external_weights_do_not_block_rewrites(self):
        import_or_skip("onnx_light.onnx_py._onnxpykernels")
        from onnx_light.onnx.reference import ReferenceEvaluator

        for constant_node in (False, True):
            with self.subTest(constant_node=constant_node), tempfile.TemporaryDirectory() as tmp:
                source = Path(tmp) / "model.onnx"
                output = Path(tmp) / "optimized.onnx"
                weight = oh.make_tensor(
                    "W",
                    TensorProto.FLOAT,
                    [16, 16],
                    external_data={"location": "missing.bin", "offset": "0", "length": "1024"},
                )
                nodes = (
                    [oh.make_node("Constant", [], ["W"], value=weight)] if constant_node else []
                )
                nodes += [
                    oh.make_node("Transpose", ["W"], ["t"], perm=[0, 1]),
                    oh.make_node("Transpose", ["t"], ["r"], perm=[1, 0]),
                    oh.make_node("Add", ["X", "r"], ["Y"]),
                ]
                model = oh.make_model(
                    oh.make_graph(
                        nodes,
                        "g",
                        [oh.make_tensor_value_info("X", TensorProto.FLOAT, [16, 16])],
                        [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [16, 16])],
                        initializer=[] if constant_node else [weight],
                    ),
                    opset_imports=[oh.make_opsetid("", 18)],
                )
                save(model, source)
                main(["optimize", str(source), "-o", str(output)])
                result = load(output)
                self.assertEqual(
                    [node.op_type for node in result.graph.node].count("Transpose"), 1
                )
                tensor = (
                    next(node for node in result.graph.node if node.op_type == "Constant")
                    .attribute[0]
                    .t
                    if constant_node
                    else result.graph.initializer[0]
                )
                self.assertEqual(tensor.data_location, TensorProto.EXTERNAL)
                self.assertFalse((Path(tmp) / "missing.bin").exists())
                values = numpy.arange(256, dtype=numpy.float32).reshape(16, 16)
                (Path(tmp) / "missing.bin").write_bytes(values.tobytes())
                feeds = {"X": numpy.ones((16, 16), dtype=numpy.float32)}
                actual = ReferenceEvaluator(load(output, load_external_data=True)).run(
                    None, feeds
                )
                self.assertEqualArray(actual[0], feeds["X"] + values.T)

    def test_invalid_selection_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "model.onnx"
            output = Path(tmp) / "optimized.onnx"
            save(_make_model(), source)
            for selection in ("", "NotNot,", ",L1", "L1,,L2", "NoSuchPattern", "Not.*"):
                with self.subTest(patterns=selection), self.assertRaises(ValueError):
                    main(["optimize", str(source), "-o", str(output), "--patterns", selection])
                self.assertFalse(output.exists())

    def test_output_must_be_in_same_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "model.onnx"
            other = Path(tmp) / "other"
            other.mkdir()
            save(_make_model(), source)
            output = other / "optimized.onnx"
            with self.assertRaisesRegex(ValueError, "same directory"):
                main(["optimize", str(source), "-o", str(output)])
            self.assertFalse(output.exists())
            if os.name != "nt":
                link = Path(tmp) / "linked"
                link.symlink_to(other, target_is_directory=True)
                with self.assertRaisesRegex(ValueError, "same directory"):
                    main(["optimize", str(source), "-o", str(link / "optimized.onnx")])
                self.assertFalse(output.exists())

    def test_existing_outputs_and_links_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "model.onnx"
            save(_make_model(), source)
            original = source.read_bytes()
            output = Path(tmp) / "weights.bin"
            output.write_bytes(b"existing weights")
            for path in (source, output):
                with self.subTest(path=path), self.assertRaises(FileExistsError):
                    main(["optimize", str(source), "-o", str(path)])
            if os.name != "nt":
                link = Path(tmp) / "link.onnx"
                link.symlink_to(output)
                with self.assertRaises(FileExistsError):
                    main(["optimize", str(source), "-o", str(link)])
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(output.read_bytes(), b"existing weights")

    def test_missing_input_and_required_output(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(FileNotFoundError):
            main(["optimize", str(Path(tmp) / "missing.onnx"), "-o", str(Path(tmp) / "out.onnx")])
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            main(["optimize", "model.onnx"])

    def test_rejects_pipeline_without_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "model.pipeline"
            output = Path(tmp) / "optimized.onnx"
            save(PipelineProto(prefill=_make_model()), source)
            with self.assertRaisesRegex(ValueError, "not a pipeline"):
                main(["optimize", str(source), "-o", str(output)])
            self.assertFalse(output.exists())

    def test_verbose(self):
        for directory in ("plain", r"windows\path"):
            with self.subTest(directory=directory), tempfile.TemporaryDirectory() as tmp:
                workdir = Path(tmp) / directory
                workdir.mkdir(parents=True)
                source = workdir / "model.onnx"
                output = workdir / "optimized.onnx"
                save(_make_model(), source)
                text = StringIO()
                with redirect_stdout(text):
                    main(["optimize", str(source), "-o", str(output), "--verbose"])
                self.assertIn("[optimize] optimize", text.getvalue())
                self.assertIn(repr(str(output)), text.getvalue())


if __name__ == "__main__":
    unittest.main()
