# source: https://github.com/onnx/onnx/blob/main/onnx/test/basic_test.py
from __future__ import annotations

import os
import pathlib
import tempfile
import unittest

import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
from onnx_light.onnx import defs, parser
from onnx_light.ext_test_case import ExtTestCase


def _simple_model() -> onnxl.ModelProto:
    """Creates a minimal ModelProto instance for use in tests."""
    model = onnxl.ModelProto()
    model.ir_version = defs.onnx_ir_version()
    model.producer_name = "onnx-test"
    model.graph.name = "test"
    return model


def _simple_tensor() -> onnxl.TensorProto:
    """Creates a minimal float TensorProto instance for use in tests."""
    return oh.make_tensor(
        name="test-tensor",
        data_type=onnxl.TensorProto.FLOAT,
        dims=(2, 3, 4),
        vals=[x + 0.5 for x in range(24)],
    )


class TestIOModel(ExtTestCase):
    """Tests for saving and loading ModelProto (protobuf format only)."""

    def test_load_model_when_input_is_bytes(self) -> None:
        proto = _simple_model()
        proto_bytes = proto.SerializeToString()
        loaded_proto = onnxl.load(proto_bytes)
        self.assertEqual(proto, loaded_proto)

    def test_save_and_load_model_when_input_is_file_name(self) -> None:
        proto = _simple_model()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model.onnx")
            onnxl.save(proto, model_path)
            loaded_proto = onnxl.load(model_path)
            self.assertEqual(proto, loaded_proto)

    def test_save_and_load_model_when_input_is_pathlike(self) -> None:
        proto = _simple_model()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = pathlib.Path(temp_dir, "model.onnx")
            onnxl.save(proto, model_path)
            loaded_proto = onnxl.load(model_path)
            self.assertEqual(proto, loaded_proto)


class TestIOTensor(ExtTestCase):
    """Tests for saving and loading TensorProto (protobuf format only)."""

    def test_load_tensor_when_input_is_bytes(self) -> None:
        proto = _simple_tensor()
        proto_bytes = proto.SerializeToString()
        loaded_proto = onnxl.TensorProto()
        loaded_proto.ParseFromString(proto_bytes)
        self.assertEqual(proto.SerializeToString(), loaded_proto.SerializeToString())

    @unittest.skip("TensorProto.SerializeToFile is not yet implemented in onnx_light")
    def test_save_and_load_tensor_when_input_is_file_name(self) -> None:
        proto = _simple_tensor()
        with tempfile.TemporaryDirectory() as temp_dir:
            tensor_path = os.path.join(temp_dir, "tensor.data")
            proto.SerializeToFile(tensor_path)
            loaded_proto = onnxl.TensorProto()
            loaded_proto.ParseFromFile(tensor_path)
            self.assertEqual(proto.SerializeToString(), loaded_proto.SerializeToString())

    @unittest.skip("TensorProto.SerializeToFile is not yet implemented in onnx_light")
    def test_save_and_load_tensor_when_input_is_pathlike(self) -> None:
        proto = _simple_tensor()
        with tempfile.TemporaryDirectory() as temp_dir:
            tensor_path = pathlib.Path(temp_dir, "tensor.data")
            proto.SerializeToFile(str(tensor_path))
            loaded_proto = onnxl.TensorProto()
            loaded_proto.ParseFromFile(str(tensor_path))
            self.assertEqual(proto.SerializeToString(), loaded_proto.SerializeToString())


class TestBasicFunctions(ExtTestCase):
    def test_protos_exist(self) -> None:
        _ = onnxl.AttributeProto
        _ = onnxl.NodeProto
        _ = onnxl.GraphProto
        _ = onnxl.ModelProto

    def test_version_exists(self) -> None:
        model = onnxl.ModelProto()
        # When we create it, ir_version should not be set.
        self.assertFalse(model.has_ir_version())
        # Set ir_version to the current IR version.
        model.ir_version = defs.onnx_ir_version()
        model_bytes = model.SerializeToString()
        model.ParseFromString(model_bytes)
        self.assertTrue(model.has_ir_version())
        # Check if the version is correct.
        self.assertEqual(model.ir_version, defs.onnx_ir_version())

    def test_model_and_graph_str(self) -> None:
        # Check that str() works without error and contains expected fields.
        model = _simple_model()
        model_str = str(model)
        self.assertIn("ir_version", model_str)
        self.assertIn("onnx-test", model_str)
        self.assertIn("test", model_str)

        text_model = """
           <
             ir_version: 10,
             opset_import: [ "" : 19]
           >
           agraph (float[N] X) => (float[N] C)
           <
             float[1] weight = {1}
           >
           {
              C = Cast<to=1>(X)
           }
        """
        model = parser.parse_model(text_model)
        model_str = str(model)
        self.assertIn("ir_version: 10", model_str)
        self.assertIn("agraph", model_str)

        graph_str = str(model.graph)
        self.assertIn("agraph", graph_str)
        self.assertIn("Cast", graph_str)
        self.assertIn("weight", graph_str)

    def test_function_str(self) -> None:
        text = """
            <
            ir_version: 9,
            opset_import: [ "" : 15, "custom_domain" : 1],
            producer_name: "FunctionProtoTest",
            producer_version: "1.0",
            model_version: 1,
            doc_string: "A test model for model local functions."
          >
         agraph (float[N] x) => (float[N] out)
         {
            out = custom_domain.Selu<alpha=2.0, gamma=3.0>(x)
         }
         <
         domain: "custom_domain",
         opset_import: [ "" : 15],
         doc_string: "Test function proto"
         >
           Selu
           <alpha: float=1.67326319217681884765625, gamma: float=1.05070102214813232421875>
           (X) => (C)
           {
               constant_alpha = Constant<value_float: float=@alpha>()
               constant_gamma = Constant<value_float: float=@gamma>()
               alpha_x = CastLike(constant_alpha, X)
               gamma_x = CastLike(constant_gamma, X)
               exp_x = Exp(X)
               alpha_x_exp_x = Mul(alpha_x, exp_x)
               alpha_x_exp_x_ = Sub(alpha_x_exp_x, alpha_x)
               neg = Mul(gamma_x, alpha_x_exp_x_)
               pos = Mul(gamma_x, X)
               _zero = Constant<value_float=0.0>()
               zero = CastLike(_zero, X)
               less_eq = LessOrEqual(X, zero)
               C = Where(less_eq, neg, pos)
           }
        """
        model = parser.parse_model(text)
        model_str = str(model)
        self.assertIn("FunctionProtoTest", model_str)

        self.assertEqual(len(model.functions), 1)
        function_str = str(model.functions[0])
        self.assertIn("Selu", function_str)
        self.assertIn("custom_domain", function_str)
        self.assertIn("Test function proto", function_str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
