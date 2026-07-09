import os
import tempfile
import unittest
import warnings

import numpy as np

from onnx_light.ext_test_case import ExtTestCase
import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
from onnx_light.onnx_lib.external_data_helper import (
    convert_model_to_external_data,
    load_external_data_for_model,
    uses_external_data,
)
from onnx_light.onnx.external_data_helper import ExternalDataInfo


def _make_model(values: np.ndarray) -> onnxl.ModelProto:
    init = oh.make_tensor(
        "W", onnxl.TensorProto.FLOAT, list(values.shape), values.tobytes(), raw=True
    )
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
                model, all_tensors_to_one_file=True, location=ext_name, size_threshold=0
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
        convert_model_to_external_data(model, all_tensors_to_one_file=False, size_threshold=0)
        init = model.graph.initializer[0]
        entries = {e.key: e.value for e in init.external_data}
        self.assertEqual(entries["location"], "W")

    def test_external_data_info_parses_location(self):
        """Tests that ExternalDataInfo parses the location field."""
        tensor = onnxl.TensorProto()
        tensor.name = "test_tensor"
        tensor.data_type = onnxl.TensorProto.FLOAT
        tensor.data_location = onnxl.TensorProto.EXTERNAL
        entry = tensor.external_data.add()
        entry.key = "location"
        entry.value = "weights.bin"

        info = ExternalDataInfo(tensor)
        self.assertEqual(info.location, "weights.bin")
        self.assertIsNone(info.offset)
        self.assertIsNone(info.length)

    def test_external_data_info_parses_offset_and_length(self):
        """Tests that ExternalDataInfo parses offset and length fields."""
        tensor = onnxl.TensorProto()
        tensor.name = "test_tensor"
        tensor.data_type = onnxl.TensorProto.FLOAT
        tensor.data_location = onnxl.TensorProto.EXTERNAL
        entry1 = tensor.external_data.add()
        entry1.key = "location"
        entry1.value = "data.bin"
        entry2 = tensor.external_data.add()
        entry2.key = "offset"
        entry2.value = "1024"
        entry3 = tensor.external_data.add()
        entry3.key = "length"
        entry3.value = "512"

        info = ExternalDataInfo(tensor)
        self.assertEqual(info.location, "data.bin")
        self.assertEqual(info.offset, 1024)
        self.assertEqual(info.length, 512)

    def test_external_data_info_empty_tensor(self):
        """Tests that ExternalDataInfo handles tensors with no external data."""
        tensor = onnxl.TensorProto()
        tensor.name = "inline_tensor"
        tensor.data_type = onnxl.TensorProto.FLOAT

        info = ExternalDataInfo(tensor)
        self.assertEqual(info.location, "")
        self.assertIsNone(info.offset)
        self.assertIsNone(info.length)

    def test_external_data_info_checksum_and_basepath(self):
        """Tests that ExternalDataInfo parses checksum and basepath fields."""
        tensor = onnxl.TensorProto()
        tensor.name = "t"
        tensor.data_type = onnxl.TensorProto.FLOAT
        tensor.data_location = onnxl.TensorProto.EXTERNAL
        for k, v in [("location", "data.bin"), ("checksum", "abc123"), ("basepath", "/some/dir")]:
            entry = tensor.external_data.add()
            entry.key = k
            entry.value = v

        info = ExternalDataInfo(tensor)
        self.assertEqual(info.location, "data.bin")
        self.assertEqual(info.checksum, "abc123")
        self.assertEqual(info.basepath, "/some/dir")

    # ------------------------------------------------------------------
    # Security tests: GHSA-cjhm-j56f-fj5v (CVE-2026-34445)
    # ------------------------------------------------------------------

    def test_external_data_info_unknown_key_warns(self):
        """Unknown external data keys must trigger a warning and be ignored."""
        tensor = onnxl.TensorProto()
        tensor.name = "t"
        tensor.data_type = onnxl.TensorProto.FLOAT
        tensor.data_location = onnxl.TensorProto.EXTERNAL
        entry_loc = tensor.external_data.add()
        entry_loc.key = "location"
        entry_loc.value = "data.bin"
        entry_bad = tensor.external_data.add()
        entry_bad.key = "__class__"
        entry_bad.value = "malicious"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            info = ExternalDataInfo(tensor)

        self.assertEqual(info.location, "data.bin")
        self.assertIsNone(info.offset)
        self.assertEqual(len(caught), 1)
        self.assertIn("__class__", str(caught[0].message))

    def test_external_data_info_negative_offset_raises(self):
        """ExternalDataInfo must reject a negative offset."""
        tensor = onnxl.TensorProto()
        tensor.name = "t"
        tensor.data_type = onnxl.TensorProto.FLOAT
        tensor.data_location = onnxl.TensorProto.EXTERNAL
        for k, v in [("location", "data.bin"), ("offset", "-1")]:
            entry = tensor.external_data.add()
            entry.key = k
            entry.value = v

        with self.assertRaises(ValueError):
            ExternalDataInfo(tensor)

    def test_external_data_info_negative_length_raises(self):
        """ExternalDataInfo must reject a negative length."""
        tensor = onnxl.TensorProto()
        tensor.name = "t"
        tensor.data_type = onnxl.TensorProto.FLOAT
        tensor.data_location = onnxl.TensorProto.EXTERNAL
        for k, v in [("location", "data.bin"), ("length", "-5")]:
            entry = tensor.external_data.add()
            entry.key = k
            entry.value = v

        with self.assertRaises(ValueError):
            ExternalDataInfo(tensor)

    def test_load_external_data_offset_past_eof_raises(self):
        """_load_external_data_for_tensor must raise when offset > file size."""
        from onnx_light.onnx_proto._numpy_helper import _load_external_data_for_tensor

        values = np.arange(4, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            ext_name = "data.bin"
            with open(os.path.join(tmp, ext_name), "wb") as f:
                f.write(values.tobytes())  # 16 bytes

            tensor = onnxl.TensorProto()
            tensor.name = "t"
            tensor.data_type = onnxl.TensorProto.FLOAT
            tensor.data_location = onnxl.TensorProto.EXTERNAL
            for k, v in [("location", ext_name), ("offset", "99999")]:
                entry = tensor.external_data.add()
                entry.key = k
                entry.value = v

            with self.assertRaises(ValueError):
                _load_external_data_for_tensor(tensor, tmp)

    def test_load_external_data_length_exceeds_available_raises(self):
        """_load_external_data_for_tensor must raise when length > available bytes."""
        from onnx_light.onnx_proto._numpy_helper import _load_external_data_for_tensor

        values = np.arange(4, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            ext_name = "data.bin"
            with open(os.path.join(tmp, ext_name), "wb") as f:
                f.write(values.tobytes())  # 16 bytes

            tensor = onnxl.TensorProto()
            tensor.name = "t"
            tensor.data_type = onnxl.TensorProto.FLOAT
            tensor.data_location = onnxl.TensorProto.EXTERNAL
            for k, v in [("location", ext_name), ("offset", "0"), ("length", "99999")]:
                entry = tensor.external_data.add()
                entry.key = k
                entry.value = v

            with self.assertRaises(ValueError):
                _load_external_data_for_tensor(tensor, tmp)

    def test_load_external_data_negative_offset_raises(self):
        """_load_external_data_for_tensor must raise when offset is negative."""
        from onnx_light.onnx_proto._numpy_helper import _load_external_data_for_tensor

        values = np.arange(4, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            ext_name = "data.bin"
            with open(os.path.join(tmp, ext_name), "wb") as f:
                f.write(values.tobytes())

            tensor = onnxl.TensorProto()
            tensor.name = "t"
            tensor.data_type = onnxl.TensorProto.FLOAT
            tensor.data_location = onnxl.TensorProto.EXTERNAL
            for k, v in [("location", ext_name), ("offset", "-1")]:
                entry = tensor.external_data.add()
                entry.key = k
                entry.value = v

            with self.assertRaises(ValueError):
                _load_external_data_for_tensor(tensor, tmp)

    def test_load_external_data_negative_length_raises(self):
        """_load_external_data_for_tensor must raise when length is negative."""
        from onnx_light.onnx_proto._numpy_helper import _load_external_data_for_tensor

        values = np.arange(4, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            ext_name = "data.bin"
            with open(os.path.join(tmp, ext_name), "wb") as f:
                f.write(values.tobytes())

            tensor = onnxl.TensorProto()
            tensor.name = "t"
            tensor.data_type = onnxl.TensorProto.FLOAT
            tensor.data_location = onnxl.TensorProto.EXTERNAL
            for k, v in [("location", ext_name), ("length", "-1")]:
                entry = tensor.external_data.add()
                entry.key = k
                entry.value = v

            with self.assertRaises(ValueError):
                _load_external_data_for_tensor(tensor, tmp)

    def test_load_external_data_valid_offset_and_length(self):
        """_load_external_data_for_tensor reads the correct slice when given offset/length."""
        from onnx_light.onnx_proto._numpy_helper import _load_external_data_for_tensor

        values = np.arange(8, dtype=np.float32)  # 32 bytes total
        with tempfile.TemporaryDirectory() as tmp:
            ext_name = "data.bin"
            with open(os.path.join(tmp, ext_name), "wb") as f:
                f.write(values.tobytes())

            # Read only elements [4:8] (16 bytes starting at byte offset 16)
            tensor = onnxl.TensorProto()
            tensor.name = "t"
            tensor.data_type = onnxl.TensorProto.FLOAT
            tensor.data_location = onnxl.TensorProto.EXTERNAL
            for k, v in [("location", ext_name), ("offset", "16"), ("length", "16")]:
                entry = tensor.external_data.add()
                entry.key = k
                entry.value = v

            _load_external_data_for_tensor(tensor, tmp)
            got = np.frombuffer(tensor.raw_data, dtype=np.float32)
            np.testing.assert_array_equal(got, values[4:])


if __name__ == "__main__":
    unittest.main(verbosity=2)
