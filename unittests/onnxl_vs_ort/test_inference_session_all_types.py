"""Unit tests for InferenceSessionAllTypes class.

Tests the wrapper class that supports all ONNX dtypes including special types
(FLOAT8, BFLOAT16, INT2, INT4, etc.) that require IOBinding workarounds.
"""

import unittest

import numpy as np

from onnx_light.ext_test_case import ExtTestCase, InferenceSessionAllTypes
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh
from onnx_light.onnx import TensorProto


class TestInferenceSessionAllTypes(ExtTestCase):
    """Tests for InferenceSessionAllTypes wrapper class."""

    def test_standard_dtype_float32(self):
        """Tests standard FLOAT32 dtype uses standard path."""
        # Create a simple Add model
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Add", ["X", "Y"], ["Z"])],
                "add_float32",
                [
                    oh.make_tensor_value_info("X", TensorProto.FLOAT, [3]),
                    oh.make_tensor_value_info("Y", TensorProto.FLOAT, [3]),
                ],
                [oh.make_tensor_value_info("Z", TensorProto.FLOAT, [3])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=13,
        )

        # Run inference
        sess = InferenceSessionAllTypes(model)
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        outputs = sess.run(None, {"X": x, "Y": y})

        # Verify results
        expected = np.array([5.0, 7.0, 9.0], dtype=np.float32)
        np.testing.assert_allclose(outputs[0], expected)

    def test_standard_dtype_int64(self):
        """Tests standard INT64 dtype uses standard path."""
        # Create a simple Add model with INT64
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Add", ["X", "Y"], ["Z"])],
                "add_int64",
                [
                    oh.make_tensor_value_info("X", TensorProto.INT64, [3]),
                    oh.make_tensor_value_info("Y", TensorProto.INT64, [3]),
                ],
                [oh.make_tensor_value_info("Z", TensorProto.INT64, [3])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=13,
        )

        # Run inference
        sess = InferenceSessionAllTypes(model)
        x = np.array([1, 2, 3], dtype=np.int64)
        y = np.array([4, 5, 6], dtype=np.int64)
        outputs = sess.run(None, {"X": x, "Y": y})

        # Verify results
        expected = np.array([5, 7, 9], dtype=np.int64)
        np.testing.assert_array_equal(outputs[0], expected)

    def test_float16_dtype_with_iobinding(self):
        """Tests FLOAT16 dtype uses IOBinding path."""
        # Create a Cast model to test FLOAT16
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Cast", ["X"], ["Y"], to=TensorProto.FLOAT16)],
                "cast_float16",
                [oh.make_tensor_value_info("X", TensorProto.FLOAT, [3])],
                [oh.make_tensor_value_info("Y", TensorProto.FLOAT16, [3])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=13,
        )

        # Run inference
        sess = InferenceSessionAllTypes(model)
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        outputs = sess.run(None, {"X": x})

        # Verify results - output should be FLOAT16
        self.assertEqual(outputs[0].dtype, np.float16)
        expected = np.array([1.0, 2.0, 3.0], dtype=np.float16)
        np.testing.assert_allclose(outputs[0], expected, rtol=1e-3)

    def test_cast_float_to_int2(self):
        """Tests Cast from FLOAT to INT2 uses IOBinding path."""
        import ml_dtypes  # noqa: F401

        from onnx_light.onnx.reference import ReferenceEvaluator

        # Create a Cast model converting FLOAT to INT2
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Cast", ["X"], ["Y"], to=TensorProto.INT2)],
                "cast_float_to_int2",
                [oh.make_tensor_value_info("X", TensorProto.FLOAT, [4])],
                [oh.make_tensor_value_info("Y", TensorProto.INT2, [4])],
            ),
            opset_imports=[oh.make_opsetid("", 23)],
            ir_version=13,
        )

        x = np.array([-2.0, -1.0, 0.0, 1.0], dtype=np.float32)

        # Run inference through onnxruntime (IOBinding path)
        sess = InferenceSessionAllTypes(model)
        outputs = sess.run(None, {"X": x})

        # Compare against the reference evaluator
        expected = ReferenceEvaluator(model).run(None, {"X": x})

        self.assertEqual(outputs[0].dtype, np.dtype("int2"))
        self.assertEqual(outputs[0].dtype, expected[0].dtype)
        np.testing.assert_array_equal(outputs[0].astype(np.int32), expected[0].astype(np.int32))

    def test_cast_int2_to_float(self):
        """Tests Cast from INT2 to FLOAT uses IOBinding path."""
        import ml_dtypes  # noqa: F401
        import onnxruntime as ort

        from onnx_light.onnx.reference import ReferenceEvaluator

        # Create a Cast model converting INT2 to FLOAT
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Cast", ["X"], ["Y"], to=TensorProto.FLOAT)],
                "cast_int2_to_float",
                [oh.make_tensor_value_info("X", TensorProto.INT2, [4])],
                [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [4])],
            ),
            opset_imports=[oh.make_opsetid("", 23)],
            ir_version=13,
        )

        # Not every onnxruntime build registers a Cast kernel consuming INT2
        # inputs. Skip the test when the model cannot be loaded.
        try:
            ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        except ort.capi.onnxruntime_pybind11_state.InvalidGraph as e:
            self.skipTest(f"onnxruntime does not support Cast from INT2: {e}")

        x = np.array([-2, -1, 0, 1], dtype=np.dtype("int2"))

        # Run inference through onnxruntime (IOBinding path)
        sess = InferenceSessionAllTypes(model)
        outputs = sess.run(None, {"X": x})

        # Compare against the reference evaluator
        expected = ReferenceEvaluator(model).run(None, {"X": x})

        self.assertEqual(outputs[0].dtype, np.float32)
        self.assertEqual(outputs[0].dtype, expected[0].dtype)
        np.testing.assert_array_equal(outputs[0], expected[0])

    def test_identity_model(self):
        """Tests Identity op with standard dtype."""
        # Create an Identity model
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Identity", ["X"], ["Y"])],
                "identity",
                [oh.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])],
                [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=13,
        )

        # Run inference
        sess = InferenceSessionAllTypes(model)
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        outputs = sess.run(None, {"X": x})

        # Verify results
        np.testing.assert_array_equal(outputs[0], x)

    def test_matmul_with_initializer(self):
        """Tests MatMul with initializer using standard dtypes."""
        # Create a MatMul model with initializer
        w = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("MatMul", ["X", "W"], ["Y"])],
                "matmul",
                [oh.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])],
                [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2])],
                [onh.from_array(w, name="W")],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=13,
        )

        # Run inference
        sess = InferenceSessionAllTypes(model)
        x = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        outputs = sess.run(None, {"X": x})

        # Verify results - should be identity matrix * w = w
        expected = w
        np.testing.assert_allclose(outputs[0], expected)

    def test_custom_providers(self):
        """Tests custom providers parameter."""
        # Create a simple Identity model
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Identity", ["X"], ["Y"])],
                "identity",
                [oh.make_tensor_value_info("X", TensorProto.FLOAT, [2])],
                [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [2])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=13,
        )

        # Run inference with explicit providers
        sess = InferenceSessionAllTypes(model, providers=["CPUExecutionProvider"])
        x = np.array([1.0, 2.0], dtype=np.float32)
        outputs = sess.run(None, {"X": x})

        # Verify results
        np.testing.assert_array_equal(outputs[0], x)


if __name__ == "__main__":
    unittest.main(verbosity=2)
