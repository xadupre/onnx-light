import os
import tempfile
import unittest

import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
from onnx_light.ext_test_case import ExtTestCase


def _make_simple_model():
    """Builds a minimal ModelProto with a Relu node."""
    return oh.make_model(
        oh.make_graph(
            [oh.make_node("Relu", ["X"], ["Y"])],
            "test_graph",
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [3])],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [3])],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=9,
    )


def _openssl_available():
    return hasattr(onnxl.ModelProto(), "SerializeToEncryptedFile")


@unittest.skipUnless(_openssl_available(), "OpenSSL not available in this build")
class TestEncryptedIO(ExtTestCase):
    def test_save_and_load_encrypted_round_trip(self):
        model = _make_simple_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.onnxc")
            onnxl.save_encrypted(model, path, "my_password")
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)

            loaded = onnxl.load_encrypted(path, "my_password")
            self.assertEqual(model.SerializeToString(), loaded.SerializeToString())

    def test_save_and_load_encrypted_bytes_key(self):
        model = _make_simple_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model_bytes_key.onnxc")
            key = b"binary\x00key\xff"
            onnxl.save_encrypted(model, path, key)
            loaded = onnxl.load_encrypted(path, key)
            self.assertEqual(model.SerializeToString(), loaded.SerializeToString())

    def test_wrong_key_raises(self):
        model = _make_simple_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model_wk.onnxc")
            onnxl.save_encrypted(model, path, "correct_key")
            with self.assertRaises(RuntimeError):
                onnxl.load_encrypted(path, "wrong_key")

    def test_different_keys_different_ciphertext(self):
        model = _make_simple_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = os.path.join(tmpdir, "m1.onnxc")
            path2 = os.path.join(tmpdir, "m2.onnxc")
            onnxl.save_encrypted(model, path1, "key_alpha")
            onnxl.save_encrypted(model, path2, "key_beta")
            # Same plaintext → same ciphertext length (AES-CBC is length-preserving).
            self.assertEqual(os.path.getsize(path1), os.path.getsize(path2))
            with open(path1, "rb") as f1, open(path2, "rb") as f2:
                self.assertNotEqual(f1.read(), f2.read())

    def test_corrupt_file_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "corrupt.onnxc")
            with open(path, "wb") as f:
                f.write(b"BADMAGIC1234567890")
            with self.assertRaises(RuntimeError):
                onnxl.load_encrypted(path, "any_key")

    def test_empty_model_round_trip(self):
        model = onnxl.ModelProto()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "empty.onnxc")
            onnxl.save_encrypted(model, path, "pw")
            loaded = onnxl.load_encrypted(path, "pw")
            self.assertEqual(model.SerializeToString(), loaded.SerializeToString())

    def test_model_with_initializers_round_trip(self):
        import numpy as np
        import onnx.numpy_helper as nph

        w = nph.from_array(np.ones((4, 4), dtype=np.float32), name="W")
        # Convert onnx.TensorProto to onnx_light TensorProto via serialization.
        w_light = onnxl.TensorProto()
        w_light.ParseFromString(w.SerializeToString())
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Relu", ["W"], ["Y"])],
                "g",
                [],
                [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [4, 4])],
                initializer=[w_light],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model_init.onnxc")
            onnxl.save_encrypted(model, path, "secret")
            loaded = onnxl.load_encrypted(path, "secret")
            self.assertEqual(model.SerializeToString(), loaded.SerializeToString())

    def test_save_encrypted_str_path(self):
        model = _make_simple_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path

            path = Path(tmpdir) / "model_path.onnxc"
            onnxl.save_encrypted(model, path, "pw")
            loaded = onnxl.load_encrypted(path, "pw")
            self.assertEqual(model.SerializeToString(), loaded.SerializeToString())

    def test_save_encrypted_string_round_trip(self):
        model = _make_simple_model()
        blob = onnxl.save_encrypted_string(model, "my_password")
        self.assertIsInstance(blob, bytes)
        self.assertGreater(len(blob), 40)
        loaded = onnxl.load_encrypted_string(blob, "my_password")
        self.assertEqual(model.SerializeToString(), loaded.SerializeToString())

    def test_save_encrypted_string_bytes_key(self):
        model = _make_simple_model()
        key = b"binary\x00key\xff"
        blob = onnxl.save_encrypted_string(model, key)
        loaded = onnxl.load_encrypted_string(blob, key)
        self.assertEqual(model.SerializeToString(), loaded.SerializeToString())

    def test_load_encrypted_string_wrong_key_raises(self):
        model = _make_simple_model()
        blob = onnxl.save_encrypted_string(model, "correct")
        with self.assertRaises(RuntimeError):
            onnxl.load_encrypted_string(blob, "wrong")

    def test_load_encrypted_string_bad_magic_raises(self):
        with self.assertRaises(RuntimeError):
            onnxl.load_encrypted_string(b"BADMAGIC1234567890", "any")

    def test_string_and_file_blobs_are_compatible(self):
        model = _make_simple_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "compat.onnxc")
            onnxl.save_encrypted(model, path, "compat_key")
            with open(path, "rb") as fh:
                blob = fh.read()
            loaded = onnxl.load_encrypted_string(blob, "compat_key")
            self.assertEqual(model.SerializeToString(), loaded.SerializeToString())


if __name__ == "__main__":
    unittest.main()
