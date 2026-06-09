import os
import tempfile
import unittest

import numpy as np

from onnx_light.ext_test_case import ExtTestCase
import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
from onnx_light.onnx.external_data_helper import (
    convert_model_to_external_data,
    load_external_data_for_model,
    uses_external_data,
)


def _make_model(values: np.ndarray) -> onnxl.ModelProto:
    init = oh.make_tensor("W", onnxl.TensorProto.FLOAT, list(values.shape), values.tobytes(), raw=True)
    inp = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, list(values.shape))
    out = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, list(values.shape))
    node = oh.make_node("Add", ["X", "W"], ["Y"])
    graph = oh.make_graph([node], "g", [inp], [out], [init])
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])
    return model


class TestExternalDataHelper(ExtTestCase):
    def test_convert_and_load_round_trip(self):
        values = np.arange(64, dtype=np.float32)
        raw = values.tobytes()
        with tempfile.TemporaryDirectory() as tmp:
            # Write the raw weight bytes to an external file ourselves and
            # build a model whose initializer references those bytes.  This
            # mimics the state of a model after ``save_model`` has flushed
            # external data to disk.
            ext_name = "weights.bin"
            with open(os.path.join(tmp, ext_name), "wb") as f:
                f.write(raw)

            model = _make_model(values)
            convert_model_to_external_data(
                model,
                all_tensors_to_one_file=True,
                location=ext_name,
                size_threshold=0,
            )
            init = model.graph.initializer[0]
            self.assertTrue(uses_external_data(init))
            self.assertTrue(init.has_external_data())
            self.assertEqual(init.data_location, onnxl.TensorProto.EXTERNAL)
            entries = {e.key: e.value for e in init.external_data}
            self.assertEqual(entries["location"], ext_name)

            # After convert, ``raw_data`` is still inline (matches onnx
            # behaviour: the bytes are written by ``save_model`` later).
            self.assertEqual(bytes(init.raw_data), raw)

            # Drop the inline raw_data to simulate ``save_model`` having
            # written it to disk and removed it from the proto.
            init.raw_data = b""
            self.assertFalse(init.raw_data)

            load_external_data_for_model(model, tmp)
            init2 = model.graph.initializer[0]
            self.assertFalse(uses_external_data(init2))
            self.assertEqual(init2.data_location, onnxl.TensorProto.DEFAULT)
            self.assertEqual(len(init2.external_data), 0)
            got = np.frombuffer(init2.raw_data, dtype=np.float32)
            np.testing.assert_array_equal(got, values)

    def test_convert_threshold_skips_small_tensors(self):
        values = np.arange(4, dtype=np.float32)  # 16 bytes
        model = _make_model(values)
        convert_model_to_external_data(model, size_threshold=1024)
        init = model.graph.initializer[0]
        self.assertFalse(uses_external_data(init))

    def test_convert_absolute_location_rejected(self):
        values = np.arange(64, dtype=np.float32)
        model = _make_model(values)
        with self.assertRaises(ValueError):
            convert_model_to_external_data(model, location="/tmp/forbidden.bin")

    def test_convert_existing_location_rejected(self):
        values = np.arange(64, dtype=np.float32)
        model = _make_model(values)
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                with open("existing.bin", "wb") as f:
                    f.write(b"x")
                with self.assertRaises(FileExistsError):
                    convert_model_to_external_data(model, location="existing.bin")
            finally:
                os.chdir(cwd)

    def test_convert_per_tensor_file(self):
        values = np.arange(64, dtype=np.float32)
        model = _make_model(values)
        convert_model_to_external_data(
            model, all_tensors_to_one_file=False, size_threshold=0
        )
        init = model.graph.initializer[0]
        entries = {e.key: e.value for e in init.external_data}
        self.assertEqual(entries["location"], "W")


if __name__ == "__main__":
    unittest.main(verbosity=2)
