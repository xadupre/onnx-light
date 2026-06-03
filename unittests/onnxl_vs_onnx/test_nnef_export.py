"""Unit tests for :mod:`onnx_light.nnef`.

The tests use the upstream :mod:`onnx` package (already a development
dependency of :mod:`onnx_light`) to build small models on the fly and
then exercise the NNEF exporter end-to-end.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

import onnx
from onnx import TensorProto, helper, numpy_helper

from onnx_light.ext_test_case import ExtTestCase
from onnx_light.nnef import (
    NNEFExportError,
    export_to_nnef,
    register_op_converter,
    save_nnef,
    supported_ops,
    to_nnef_text,
)
from onnx_light.nnef.tensor_io import read_nnef_tensor, write_nnef_tensor


def _conv_model() -> onnx.ModelProto:
    rng = np.random.default_rng(0)
    W = numpy_helper.from_array(rng.standard_normal((8, 3, 3, 3)).astype(np.float32), name="W")
    B = numpy_helper.from_array(np.zeros(8, dtype=np.float32), name="B")
    nodes = [
        helper.make_node("Conv", ["X", "W", "B"], ["c"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c"], ["r"]),
        helper.make_node("GlobalAveragePool", ["r"], ["p"]),
        helper.make_node("Flatten", ["p"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "demo",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 32, 32])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8])],
        [W, B],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


class TestNNEFTensorIO(ExtTestCase):
    """Round-trip tests for the NNEF ``*.dat`` binary tensor format."""

    def _roundtrip(self, array: np.ndarray) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.dat")
            write_nnef_tensor(path, array)
            self.assertGreaterEqual(os.path.getsize(path), 128)
            back = read_nnef_tensor(path)
        np.testing.assert_array_equal(back, array)
        self.assertEqual(back.dtype, array.dtype)
        self.assertEqual(back.shape, array.shape)

    def test_roundtrip_float32(self):
        self._roundtrip(np.linspace(-1, 1, 24, dtype=np.float32).reshape(2, 3, 4))

    def test_roundtrip_float16(self):
        self._roundtrip(np.arange(6, dtype=np.float16).reshape(2, 3))

    def test_roundtrip_int64(self):
        self._roundtrip(np.array([[-1, 2, 3], [4, 5, 6]], dtype=np.int64))

    def test_roundtrip_uint8(self):
        self._roundtrip(np.arange(10, dtype=np.uint8))

    def test_roundtrip_bool(self):
        self._roundtrip(np.array([True, False, True, True], dtype=np.bool_))

    def test_roundtrip_scalar(self):
        self._roundtrip(np.array(3.14, dtype=np.float32))

    def test_rejects_rank_too_large(self):
        with self.assertRaises(ValueError):
            write_nnef_tensor(os.devnull, np.zeros((1,) * 9, dtype=np.float32))


class TestNNEFExporter(ExtTestCase):
    """Tests for the ONNX → NNEF text/file exporter."""

    def test_supported_ops_contains_common(self):
        ops = set(supported_ops())
        for op in ("Conv", "Relu", "Add", "MatMul", "Gemm", "Softmax", "Reshape"):
            self.assertIn(op, ops)

    def test_export_conv_model_text(self):
        model = _conv_model()
        text = to_nnef_text(model)
        self.assertIn("version 1.0;", text)
        self.assertIn("graph demo(X) -> (Y)", text)
        self.assertIn("X = external(shape = [1, 3, 32, 32]);", text)
        self.assertIn("W = variable(shape = [8, 3, 3, 3], label = 'W');", text)
        self.assertIn("B = variable(shape = [8], label = 'B');", text)
        self.assertIn("c = conv(X, W, B", text)
        self.assertIn("padding = [[1, 1], [1, 1]]", text)
        self.assertIn("r = relu(c);", text)
        self.assertIn("mean_reduce(r, axes = [2, 3]);", text)
        self.assertIn("Y = reshape(p, shape = [0, -1]);", text)

    def test_save_nnef_creates_directory(self):
        model = _conv_model()
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "model_nnef")
            save_nnef(model, out)
            files = sorted(os.listdir(out))
            self.assertIn("graph.nnef", files)
            self.assertIn("W.dat", files)
            self.assertIn("B.dat", files)
            # Tensors round-trip back to numpy arrays of the right shape.
            w = read_nnef_tensor(os.path.join(out, "W.dat"))
            self.assertEqual(w.shape, (8, 3, 3, 3))
            self.assertEqual(w.dtype, np.float32)

    def test_save_nnef_no_overwrite(self):
        model = _conv_model()
        with tempfile.TemporaryDirectory() as d:
            save_nnef(model, d)
            with self.assertRaises(FileExistsError):
                save_nnef(model, d, overwrite=False)

    def test_export_gemm_handles_transB(self):
        rng = np.random.default_rng(1)
        W = numpy_helper.from_array(rng.standard_normal((10, 5)).astype(np.float32), name="W")
        B = numpy_helper.from_array(np.zeros(10, dtype=np.float32), name="B")
        nodes = [helper.make_node("Gemm", ["X", "W", "B"], ["Y"], transB=1)]
        graph = helper.make_graph(
            nodes,
            "gemm",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 5])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 10])],
            [W, B],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        text = to_nnef_text(model)
        # The transposed B must show up as an intermediate transpose call.
        self.assertIn("transpose(W)", text)
        self.assertIn("matmul(X,", text)
        self.assertIn("add(", text)

    def test_export_reshape_requires_constant_shape(self):
        # Shape is a graph input, not an initializer → must raise.
        nodes = [helper.make_node("Reshape", ["X", "shape"], ["Y"])]
        graph = helper.make_graph(
            nodes,
            "bad",
            [
                helper.make_tensor_value_info("X", TensorProto.FLOAT, [4]),
                helper.make_tensor_value_info("shape", TensorProto.INT64, [2]),
            ],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2])],
            [],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        with self.assertRaises(NNEFExportError):
            to_nnef_text(model)

    def test_unsupported_op_raises(self):
        nodes = [helper.make_node("ThisOpDoesNotExist", ["X"], ["Y"])]
        graph = helper.make_graph(
            nodes,
            "u",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [3])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3])],
            [],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        with self.assertRaises(NNEFExportError) as cm:
            to_nnef_text(model)
        self.assertIn("ThisOpDoesNotExist", str(cm.exception))

    def test_register_custom_op_converter(self):
        nodes = [helper.make_node("MyCustomOp", ["X"], ["Y"])]
        graph = helper.make_graph(
            nodes,
            "c",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [3])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3])],
            [],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        try:
            register_op_converter(
                "MyCustomOp",
                lambda ctx, node, attrs, inputs, outputs: ctx.add_statement(
                    f"{outputs[0]} = my_custom({inputs[0]});"
                ),
            )
            text = to_nnef_text(model)
            self.assertIn("Y = my_custom(X);", text)
        finally:
            # Avoid leaking the registration to other tests.
            from onnx_light.nnef import exporter as _exp

            _exp._CONVERTERS.pop("MyCustomOp", None)

    def test_export_to_nnef_returns_graph_object(self):
        model = _conv_model()
        nnef = export_to_nnef(model)
        self.assertEqual(nnef.inputs, ["X"])
        self.assertEqual(nnef.outputs, ["Y"])
        self.assertIn("W", nnef.initializers)
        self.assertEqual(nnef.initializers["W"].shape, (8, 3, 3, 3))


if __name__ == "__main__":
    unittest.main()
