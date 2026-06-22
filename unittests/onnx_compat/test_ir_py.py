import os
import pathlib
import tempfile
import unittest
import numpy as np
from onnx_light.ext_test_case import ExtTestCase, has_ir_py


@unittest.skipIf(not has_ir_py(), "ir-py is not installed")
class TestCompatibilityWithIrPy(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["USE_ONNX_LIGHT"] = "1"
        return super().setUpClass()

    def test_external_tensor_empty_tensor(self):
        import onnx_ir as ir
        import onnx_light.onnx as onnx

        def _to_external_tensor(tensor_proto, dir: str, filename: str):
            onnx.external_data_helper.set_external_data(tensor_proto, location=filename)
            path = pathlib.Path(dir) / filename
            with open(path, "wb") as f:
                f.write(tensor_proto.raw_data)
            tensor_proto.ClearField("raw_data")
            tensor_proto.data_location = onnx.TensorProto.EXTERNAL

        expected_array = np.array([], dtype=np.float32)
        tensor_proto = ir.serde.serialize_tensor(ir.Tensor(expected_array))
        with tempfile.TemporaryDirectory() as temp_dir:
            _to_external_tensor(tensor_proto, temp_dir, "tensor.bin")
            tensor = ir.serde.deserialize_tensor(tensor_proto, temp_dir)
            np.testing.assert_array_equal(tensor.numpy(), expected_array)
            # Close the mmap file by deleting the reference to tensor so Windows doesn't complain
            # about permission errors
            del tensor


if __name__ == "__main__":
    unittest.main(verbosity=2, exit=False)
