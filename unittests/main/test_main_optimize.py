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
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "model.onnx"
            output = Path(tmp) / "optimized.onnx"
            save(_make_model(), source)
            text = StringIO()
            with redirect_stdout(text):
                main(["optimize", str(source), "-o", str(output), "--verbose"])
            self.assertIn("[optimize] optimize", text.getvalue())
            self.assertIn(str(output), text.getvalue())


if __name__ == "__main__":
    unittest.main()
