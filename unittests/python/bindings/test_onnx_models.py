import os
import gc
import unittest
from unittest.mock import patch
import numpy as np
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh
import onnx_light.onnx as onnxl
import onnx_light.onnx_proto._io_helper as io_helper
from onnx_light.ext_test_case import ExtTestCase, import_or_skip
from onnx_light.onnx import TensorProto

# The deterministic random helpers are only available in the full build; skip
# this module on a reduced build (ONNX_LIGHT_BUILD_KERNELS=OFF).
rand = import_or_skip("onnx_light.onnx_lib.backend.random", "rand")


class TestOnnxLightHelper(ExtTestCase):
    def assertEqualModelProto(self, model1, model2):
        self.assertEqual(type(model1), type(model2))
        search = 'domain: ""'
        s1 = model1.SerializeToString()
        s2 = model2.SerializeToString()
        if len(s1) < 100000:
            spl1 = str(model1).split(search)
            spl2 = str(model2).split(search)
            if len(spl1) != len(spl2) or s1 != s2:
                n1 = self.get_dump_file("model1.onnxl.txt")
                with open(n1, "w") as f:
                    f.write(str(model1))
                n2 = self.get_dump_file("model2.onnxl.txt")
                with open(n2, "w") as f:
                    f.write(str(model2))
            self.assertEqual(len(spl1), len(spl2))
        self.assertEqual(s1, s2)

    @classmethod
    def make_model_gemm(cls, oh, tp):
        itype = tp.FLOAT
        return oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Gemm", ["X", "Y"], ["XY"]),
                    oh.make_node("Gemm", ["X", "Z"], ["XZ"]),
                    oh.make_node("Concat", ["XY", "XZ"], ["XYZ"], axis=1),
                ],
                "gemm_graph",
                [
                    oh.make_tensor_value_info("X", itype, [None, None]),
                    oh.make_tensor_value_info("Y", itype, [None, None]),
                    oh.make_tensor_value_info("Z", itype, [None, None]),
                ],
                [oh.make_tensor_value_info("XYZ", itype, [None, None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )

    def test_model_gemm_onnx_to_onnxlight(self):
        name = self.get_dump_file("test_model_gemm_onnx_to_onnxlight.onnx")
        model = self.make_model_gemm(oh, onnxl.TensorProto)
        onnxl.save(model, name)
        model2 = onnxl.load(name)
        self.assertEqual(len(model.graph.node), len(model2.graph.node))
        name2 = self.get_dump_file("test_model_gemm_onnx_to_onnxlight_2.onnx")
        onnxl.save(model2, name2)
        model3 = onnxl.load(name2)
        self.assertEqualModelProto(model, model3)

    def test_model_gemm_onnxlight_to_onnx(self):
        name2 = self.get_dump_file("test_model_gemm_onnxlight_to_onnx_2.onnx")
        model2 = self.make_model_gemm(oh, onnxl.TensorProto)
        onnxl.save(model2, name2)
        model = onnxl.load(name2)
        self.assertEqual(len(model.graph.node), len(model2.graph.node))
        name = self.get_dump_file("test_model_gemm_onnxlight_to_onnx.onnx")
        onnxl.save(model, name)
        model3 = onnxl.load(name)
        self.assertEqualModelProto(model, model3)

    def _get_model_with_initializers(self, oh, onh):
        TFLOAT = TensorProto.FLOAT
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Unsqueeze", ["X", "zero"], ["xu1"]),
                    oh.make_node("Unsqueeze", ["xu1", "un"], ["xu2"]),
                    oh.make_node("Reshape", ["xu2", "shape1"], ["xm1"]),
                    oh.make_node("Add", ["Y1", "Y2"], ["Y"]),
                    oh.make_node("Reshape", ["Y", "shape2"], ["xm2c"]),
                    oh.make_node("Cast", ["xm2c"], ["xm2"], to=1),
                    oh.make_node("MatMul", ["xm1", "xm2"], ["xm"]),
                    oh.make_node("Reshape", ["xm", "shape3"], ["Z"]),
                ],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, [32, 128])],
                [oh.make_tensor_value_info("Z", TFLOAT, [3, 5, 32, 64])],
                [
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(rand(3, 5, 128, 64, seed=1).astype(np.float32), name="Y1"),
                    onh.from_array(np.array([1], dtype=np.int64), name="un"),
                    onh.from_array(rand(3, 5, 128, 64, seed=2).astype(np.float32), name="Y2"),
                    onh.from_array(np.array([1, 32, 128], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([15, 128, 64], dtype=np.int64), name="shape2"),
                    onh.from_array(np.array([3, 5, 32, 64], dtype=np.int64), name="shape3"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        return model

    def test_parallelized_loading(self):
        # saving with onnx
        name = self.get_dump_file("test_parallelized_loading.onnx")
        model = self._get_model_with_initializers(oh, onnxl.numpy_helper)
        onnxl.save(model, name)
        # loading with onnxlight
        model2 = onnxl.load(name, num_threads=2)
        self.assertEqual(len(model.graph.node), len(model2.graph.node))
        name2 = self.get_dump_file("test_parallelized_loading.onnx")
        onnxl.save(model2, name2)
        model3 = onnxl.load(name2)
        self.assertEqualModelProto(model, model3)

    def test_repeated_proto_field_iterates_by_reference(self):
        tensor = oh.make_tensor("W", onnxl.TensorProto.FLOAT, [1], [1.0])
        model = oh.make_model(oh.make_graph([], "g", [], [], [tensor]))
        first = next(iter(model.graph.initializer))
        first.name = "W2"
        self.assertEqual(model.graph.initializer[0].name, "W2")

    def test_parse_from_string_bytes_with_parallel_options(self):
        model = self._get_model_with_initializers(oh, onnxl.numpy_helper)
        serialized = model.SerializeToString()
        opts = onnxl.ParseOptions()
        opts.num_threads = 2
        parsed = onnxl.ModelProto()
        parsed.ParseFromString(serialized, opts)
        parsed_ref = onnxl.ModelProto()
        parsed_ref.ParseFromString(serialized)
        self.assertEqual(len(parsed.graph.initializer), len(model.graph.initializer))
        self.assertEqual(parsed.SerializeToString(), parsed_ref.SerializeToString())

    def test_parse_options_touch_raw_data_pages_flag(self):
        opts = onnxl.ParseOptions()
        self.assertFalse(opts._touch_raw_data_pages)
        opts._touch_raw_data_pages = True
        self.assertTrue(opts._touch_raw_data_pages)

    def test_parse_options_tiny_external_data_threshold_flag(self):
        opts = onnxl.ParseOptions()
        self.assertEqual(opts.tiny_external_data_threshold, -1)
        opts.tiny_external_data_threshold = 128
        self.assertEqual(opts.tiny_external_data_threshold, 128)

    def test_parse_options_file_load_mode(self):
        opts = onnxl.ParseOptions()
        # Default is AUTO.
        self.assertEqual(opts.file_load_mode, onnxl.FileLoadMode.AUTO)
        opts.file_load_mode = onnxl.FileLoadMode.MMAP
        self.assertEqual(opts.file_load_mode, onnxl.FileLoadMode.MMAP)
        opts.file_load_mode = onnxl.FileLoadMode.IFSTREAM
        self.assertEqual(opts.file_load_mode, onnxl.FileLoadMode.IFSTREAM)

    def test_parse_options_raw_data_callback(self):
        # Default is None (no callback).
        opts = onnxl.ParseOptions()
        self.assertIsNone(opts.raw_data_callback)

        # Round-trips the exact Python callable that was assigned.
        def callback(tensor):
            return None

        opts.raw_data_callback = callback
        self.assertIs(opts.raw_data_callback, callback)

        # Assigning None disables the callback again.
        opts.raw_data_callback = None
        self.assertIsNone(opts.raw_data_callback)

    def test_parse_options_raw_data_callback_invoked(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor = onh.from_array(arr, name="W")
        model = oh.make_model(oh.make_graph([], "g", [], [], [tensor]))
        serialized = model.SerializeToString()

        seen = []
        deleted = []

        def callback(parsed_tensor):
            seen.append(parsed_tensor.name)
            return lambda: deleted.append(parsed_tensor.name)

        opts = onnxl.ParseOptions()
        opts.raw_data_callback = callback
        parsed = onnxl.ModelProto()
        parsed.ParseFromString(serialized, opts)
        # The callback fires once for the only tensor carrying raw_data.
        self.assertEqual(seen, ["W"])
        # Data is still readable: attaching a deleter does not move the bytes.
        np.testing.assert_array_equal(onh.to_array(parsed.graph.initializer[0]), arr)
        # The deleter runs once the tensor's raw_data is released.
        del parsed
        gc.collect()
        self.assertEqual(deleted, ["W"])

    def test_raw_data_callback_object_reports_progress(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor = onh.from_array(arr, name="W")
        model = oh.make_model(oh.make_graph([], "g", [], [], [tensor]))
        serialized = model.SerializeToString()

        seen = []
        cb = onnxl.RawDataCallback(lambda parsed_tensor: seen.append(parsed_tensor.name))

        opts = onnxl.ParseOptions()
        opts.raw_data_callback = cb
        # The callback object round-trips through ParseOptions.
        self.assertIs(opts.raw_data_callback, cb)

        parsed = onnxl.ModelProto()
        parsed.ParseFromString(serialized, opts)
        # The progress callable fires once for the tensor carrying raw_data.
        self.assertEqual(seen, ["W"])
        # The default C++ allocation is preserved: data stays readable.
        np.testing.assert_array_equal(onh.to_array(parsed.graph.initializer[0]), arr)

    def test_raw_data_callback_object_default_is_noop(self):
        arr = np.array([4.0, 5.0], dtype=np.float32)
        tensor = onh.from_array(arr, name="W")
        model = oh.make_model(oh.make_graph([], "g", [], [], [tensor]))
        serialized = model.SerializeToString()

        cb = onnxl.RawDataCallback()
        # Without an ``on_tensor`` callable the object is a pure no-op.
        self.assertIsNone(cb.on_tensor)
        # Calling it directly returns None (default allocation unchanged).
        self.assertIsNone(cb(tensor))

        opts = onnxl.ParseOptions()
        opts.raw_data_callback = cb
        parsed = onnxl.ModelProto()
        parsed.ParseFromString(serialized, opts)
        np.testing.assert_array_equal(onh.to_array(parsed.graph.initializer[0]), arr)

        # ``on_tensor`` can be cleared back to None.
        cb.on_tensor = lambda parsed_tensor: None
        self.assertIsNotNone(cb.on_tensor)
        cb.on_tensor = None
        self.assertIsNone(cb.on_tensor)

    def test_parse_options_raw_data_callback_returning_none(self):
        arr = np.array([4.0, 5.0], dtype=np.float32)
        tensor = onh.from_array(arr, name="W")
        model = oh.make_model(oh.make_graph([], "g", [], [], [tensor]))
        serialized = model.SerializeToString()

        opts = onnxl.ParseOptions()
        opts.raw_data_callback = lambda parsed_tensor: None
        parsed = onnxl.ModelProto()
        parsed.ParseFromString(serialized, opts)
        # Returning None leaves the tensor ownership and data unchanged.
        np.testing.assert_array_equal(onh.to_array(parsed.graph.initializer[0]), arr)

    def test_serialize_options_raw_data_callback(self):
        opts = onnxl.SerializeOptions()
        self.assertIsNone(opts.raw_data_callback)

        def callback(tensor, buffer, size_only):
            return 0

        opts.raw_data_callback = callback
        self.assertIs(opts.raw_data_callback, callback)
        opts.raw_data_callback = None
        self.assertIsNone(opts.raw_data_callback)

        # Round-trip a model under each FileLoadMode and verify the parsed
        # model is byte-identical to the saved one.
        name = self.get_dump_file("test_load_with_file_load_mode.onnx")
        model = self._get_model_with_initializers(oh, onnxl.numpy_helper)
        onnxl.save(model, name)

        # Enum value, string spelling and explicit AUTO must all work and
        # produce identical results.
        baseline = onnxl.load(name)
        for mode in (
            onnxl.FileLoadMode.AUTO,
            onnxl.FileLoadMode.MMAP,
            onnxl.FileLoadMode.IFSTREAM,
            "auto",
            "mmap",
            "ifstream",
        ):
            with self.subTest(mode=mode):
                loaded = onnxl.load(name, file_load_mode=mode)
                self.assertEqualModelProto(baseline, loaded)

    def test_load_file_load_mode_invalid_string_raises(self):
        name = self.get_dump_file("test_load_file_load_mode_invalid.onnx")
        model = self._get_model_with_initializers(oh, onh)
        onnxl.save(model, name)
        with self.assertRaises(ValueError):
            onnxl.load(name, file_load_mode="not-a-mode")

    def test_load_file_load_mode_mmap_with_no_copy_raises(self):
        # Explicitly forcing MMAP with no_copy=True on a single-file model
        # is rejected by the binding because the mmap mapping is released
        # when the binding returns, which would dangle the borrowed pointers.
        name = self.get_dump_file("test_load_file_load_mode_mmap_no_copy.onnx")
        model = self._get_model_with_initializers(oh, onnxl.numpy_helper)
        onnxl.save(model, name)
        with self.assertRaises(RuntimeError):
            onnxl.load(name, no_copy=True, file_load_mode=onnxl.FileLoadMode.MMAP)

    def test_load_with_touch_raw_data_pages_option(self):
        name = self.get_dump_file("test_load_with_touch_raw_data_pages_option.onnx")
        model = self._get_model_with_initializers(oh, onnxl.numpy_helper)
        onnxl.save(model, name)
        model2 = onnxl.load(name, no_copy=True, touch_raw_data_pages=True)
        self.assertEqual(len(model.graph.node), len(model2.graph.node))

    def test_parallelized_loading_min_block_size(self):
        # Verifies that min_block_size causes small tensor blocks to be read
        # on the calling thread while large ones are still parallelised.
        name = self.get_dump_file("test_parallelized_loading_min_block_size.onnx")
        model = self._get_model_with_initializers(oh, onnxl.numpy_helper)
        onnxl.save(model, name)
        # Use a very large min_block_size so every block is read on the main thread.
        model2 = onnxl.load(name, num_threads=2, min_block_size=10 * 1024 * 1024)
        self.assertEqual(len(model.graph.node), len(model2.graph.node))
        name2 = self.get_dump_file("test_parallelized_loading_min_block_size_out.onnx")
        onnxl.save(model2, name2)
        model3 = onnxl.load(name2)
        self.assertEqualModelProto(model, model3)

    def test_parallelized_loading_min_block_size_partial(self):
        # A min_block_size between the smallest and largest initializer means
        # some blocks are parallel and some are not; the result must still be correct.
        name = self.get_dump_file("test_parallelized_loading_min_block_size_partial.onnx")
        model = self._get_model_with_initializers(oh, onnxl.numpy_helper)
        onnxl.save(model, name)
        # Setting min_block_size=1 means every block of at least 1 byte is
        # parallelised, so all non-empty tensors go through the thread pool.
        # This is functionally equivalent to min_block_size=0 for typical models
        # but exercises the threshold code path.
        model2 = onnxl.load(name, num_threads=2, min_block_size=1)
        self.assertEqual(len(model.graph.node), len(model2.graph.node))
        name2 = self.get_dump_file("test_parallelized_loading_min_block_size_partial_out.onnx")
        onnxl.save(model2, name2)
        model3 = onnxl.load(name2)
        self.assertEqualModelProto(model, model3)

    def test_parallelized_saving(self):
        name = self.get_dump_file("test_parallelized_saving.onnx")
        model = self._get_model_with_initializers(oh, onnxl.numpy_helper)
        onnxl.save(model, name)
        model2 = onnxl.load(name)
        self.assertEqual(len(model.graph.node), len(model2.graph.node))
        name2 = self.get_dump_file("test_parallelized_saving_out.onnx")
        onnxl.save(model2, name2, num_threads=2, min_block_size=1)
        model3 = onnxl.load(name2)
        self.assertEqualModelProto(model, model3)

    def test_parallelized_serialize_to_string(self):
        name = self.get_dump_file("test_parallelized_serialize_to_string.onnx")
        model = self._get_model_with_initializers(oh, onnxl.numpy_helper)
        onnxl.save(model, name)
        loaded_model = onnxl.load(name)

        opts = onnxl.SerializeOptions()
        opts.num_threads = 2
        opts.min_parallel_block_size = 1

        serialized = loaded_model.SerializeToString()
        serialized_parallel = loaded_model.SerializeToString(opts)
        self.assertEqual(serialized, serialized_parallel)

        reparsed_model = onnxl.ModelProto()
        reparsed_model.ParseFromString(serialized_parallel)
        self.assertEqual(len(loaded_model.graph.node), len(reparsed_model.graph.node))

    def test_serialize_size_uses_raw_data_callback(self):
        original = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        replacement = np.arange(25, dtype=np.float32) + 50
        replacement_bytes = np.frombuffer(replacement.tobytes(), dtype=np.uint8)
        model = oh.make_model(
            oh.make_graph([], "g", [], [], [onh.from_array(original, name="W")])
        )

        def callback(tensor, buffer, size_only):
            expected_size = replacement.nbytes if tensor.name == "W" else len(tensor.raw_data)
            if size_only:
                return expected_size
            if tensor.name == "W":
                tensor.ClearField("dims")
                tensor.dims.extend([25])
                np.copyto(np.asarray(buffer), replacement_bytes)
            else:
                np.copyto(np.asarray(buffer), np.frombuffer(tensor.raw_data, dtype=np.uint8))
            return expected_size

        opts = onnxl.SerializeOptions()
        opts.raw_data_callback = callback
        size_result = model.SerializeSize(opts)
        serialized = model.SerializeToString(opts)
        self.assertEqual(size_result.size(), len(serialized))

        reparsed = onnxl.ModelProto()
        reparsed.ParseFromString(serialized)
        np.testing.assert_array_equal(onh.to_array(reparsed.graph.initializer[0]), replacement)
        self.assertEqual(list(model.graph.initializer[0].dims), [3])

    def test_serialize_parse_callback_chacha20_weights(self):
        if not hasattr(onnxl.ModelProto(), "SerializeToEncryptedString"):
            self.skipTest("OpenSSL not available in this build")

        key = "callback-secret"
        original = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        model = oh.make_model(
            oh.make_graph([], "g", [], [], [onh.from_array(original, name="W")])
        )

        encrypted_by_name = {}
        meta_by_name = {}

        def _encrypt_bytes_chacha20(raw: bytes) -> bytes:
            """Encrypts raw bytes into an ONNXCRY2 blob."""
            payload = onh.from_array(np.frombuffer(raw, dtype=np.uint8), name="PAYLOAD")
            holder = oh.make_model(oh.make_graph([], "encrypt_payload", [], [], [payload]))
            return onnxl.save_encrypted_string(holder, key, encryption="ChaCha20-Poly1305")

        def _decrypt_bytes_chacha20(blob: bytes) -> bytes:
            """Decrypts an ONNXCRY2 blob back to raw bytes."""
            holder = onnxl.load_encrypted_string(blob, key)
            return onh.to_array(holder.graph.initializer[0]).tobytes()

        def serialize_callback(tensor: onnxl.TensorProto, buffer, size_only: bool) -> int:
            """Encrypts tensor bytes and writes them into the provided serialization buffer."""
            encrypted = encrypted_by_name.get(tensor.name)
            if encrypted is None:
                encrypted = _encrypt_bytes_chacha20(bytes(tensor.raw_data))
                encrypted_by_name[tensor.name] = encrypted
                meta_by_name[tensor.name] = (int(tensor.data_type), list(tensor.dims))
            if size_only:
                return len(encrypted)
            orig_dtype, orig_dims = meta_by_name[tensor.name]
            tensor.data_type = onnxl.TensorProto.UINT8
            tensor.ClearField("dims")
            tensor.dims.extend([len(encrypted)])
            tensor.doc_string = f"chacha20:{orig_dtype}:{','.join(map(str, orig_dims))}"
            np.copyto(np.asarray(buffer), np.frombuffer(encrypted, dtype=np.uint8))
            return len(encrypted)

        sopts = onnxl.SerializeOptions()
        sopts.raw_data_callback = serialize_callback
        serialized = model.SerializeToString(sopts)

        def parse_callback(tensor: onnxl.TensorProto) -> None:
            """Decrypts callback-encrypted tensor bytes and restores original metadata."""
            if not tensor.doc_string.startswith("chacha20:"):
                return None
            _, dtype_text, dims_text = tensor.doc_string.split(":", 2)
            tensor.data_type = int(dtype_text)
            tensor.ClearField("dims")
            if dims_text:
                tensor.dims.extend(int(value) for value in dims_text.split(",") if value)
            tensor.raw_data = _decrypt_bytes_chacha20(bytes(tensor.raw_data))
            return None

        popts = onnxl.ParseOptions()
        popts.raw_data_callback = parse_callback

        reparsed = onnxl.ModelProto()
        reparsed.ParseFromString(serialized, popts)
        np.testing.assert_array_equal(onh.to_array(reparsed.graph.initializer[0]), original)

    def test_writing_external_weights_write(self):
        nameo = self.get_dump_file("test_writing_external_weights.original.onnx")
        name = self.get_dump_file("test_writing_external_weights.onnx")
        weights = self.get_dump_file("test_writing_external_weights.data")
        model = self._get_model_with_initializers(oh, onnxl.numpy_helper)
        proto = onnxl.ModelProto()
        s = model.SerializeToString()
        with open(nameo, "wb") as f:
            f.write(s)
        proto.ParseFromString(s)
        proto.SerializeToFile(name, external_data_file=weights)
        reload = onnxl.load(name)
        self.assertEqual(len(reload.graph.initializer), len(model.graph.initializer))

    def test_writing_external_weights_read(self):
        nameo = self.get_dump_file("test_writing_external_weights.original.onnx")
        name = self.get_dump_file("test_writing_external_weights.onnx")
        weights = self.get_dump_file("test_writing_external_weights.data")
        model = self._get_model_with_initializers(oh, onnxl.numpy_helper)
        proto = onnxl.ModelProto()
        s = model.SerializeToString()
        with open(nameo, "wb") as f:
            f.write(s)
        proto.ParseFromString(s)
        proto.SerializeToFile(name, external_data_file=weights)
        reload = onnxl.load(name)
        self.assertEqual(len(reload.graph.initializer), len(model.graph.initializer))
        proto2 = onnxl.ModelProto()
        proto2.ParseFromFile(name, external_data_file=weights)
        self.assertEqual(len(proto2.graph.initializer), len(model.graph.initializer))

    def test_writing_external_weights_read_with_file_load_mode(self):
        # Parsing a two-file model must honour AUTO/IFSTREAM file_load_mode
        # (both backed by std::ifstream via TwoFilesStream) and reject MMAP.
        nameo = self.get_dump_file(
            "test_writing_external_weights_flm.original.onnx"
        )
        name = self.get_dump_file("test_writing_external_weights_flm.onnx")
        weights = self.get_dump_file("test_writing_external_weights_flm.data")
        model = self._get_model_with_initializers(oh, onnxl.numpy_helper)
        proto = onnxl.ModelProto()
        s = model.SerializeToString()
        with open(nameo, "wb") as f:
            f.write(s)
        proto.ParseFromString(s)
        proto.SerializeToFile(name, external_data_file=weights)
        for mode in (onnxl.FileLoadMode.AUTO, onnxl.FileLoadMode.IFSTREAM):
            with self.subTest(mode=mode):
                opts = onnxl.ParseOptions()
                opts.file_load_mode = mode
                proto2 = onnxl.ModelProto()
                proto2.ParseFromFile(name, opts, external_data_file=weights)
                self.assertEqual(
                    len(proto2.graph.initializer), len(model.graph.initializer)
                )
        opts = onnxl.ParseOptions()
        opts.file_load_mode = onnxl.FileLoadMode.MMAP
        proto3 = onnxl.ModelProto()
        with self.assertRaises(RuntimeError):
            proto3.ParseFromFile(name, opts, external_data_file=weights)
        model = self._get_model_with_initializers(oh, onnxl.numpy_helper)
        expected = [onh.to_array(i) for i in model.graph.initializer]
        name = self.get_dump_file("test_writing_external_weights_read_from_onnx.onnx")
        weights = self.get_dump_file(
            "test_writing_external_weights_read_from_onnx.data", clean=True
        )
        onnxl.save(model, name, save_as_external_data=True, location=os.path.split(weights)[-1])
        # onnxl.save does not modify the source model in-place; verify via the saved file.
        saved_no_ext = onnxl.load(name, load_external_data=False)
        location = [int(init.data_location) for init in saved_no_ext.graph.initializer]
        self.assertEqual(location, [0, 1, 0, 1, 0, 0, 0])
        proto2 = onnxl.ModelProto()
        proto2.ParseFromFile(name, external_data_file=weights)
        self.assertEqual(len(proto2.graph.initializer), len(model.graph.initializer))

        def tweak(i):
            t = onnxl.TensorProto()
            t.ParseFromString(i.SerializeToString())
            return t

        got = [onh.to_array(tweak(i)) for i in proto2.graph.initializer]
        self.assertEqual(len(expected), len(got))
        for a, b in zip(expected, got):
            self.assertEqualArray(a, b)

    def test_loading_external_weights(self):
        name = self.get_dump_file("test_loading_external_weights.onnx")
        weights = self.get_dump_file("test_loading_external_weights.data")
        model = self._get_model_with_initializers(oh, onnxl.numpy_helper)
        onnxl.save(model, name, location=os.path.split(weights)[-1], save_as_external_data=True)
        proto = onnxl.load(name, location=weights)
        self.assertEqual(len(proto.graph.initializer), len(model.graph.initializer))
        proto_name = self.get_dump_file("test_loading_external_weights.2.onnx")
        onnxl.save(proto, proto_name)
        restored = onnxl.load(proto_name)
        self.assertEqual(len(restored.graph.initializer), len(model.graph.initializer))

    def test_resave_after_load_external_data_matches(self):
        # After calling ``tensor.load_external_data`` on a TensorProto, the
        # tensor carries both the external_data metadata and an in-memory
        # raw_data buffer. When the model holding this tensor is saved again
        # with external data, we assume the in-memory buffer matches the
        # original on-disk data (since loading does not modify it), so:
        #   - the external weights file on disk must remain byte-identical
        #     to the original (the data of the tensor is not rewritten
        #     with anything different);
        #   - the new ``.onnx`` file must not embed the raw_data inline
        #     for external initializers (the data is not duplicated).
        weights_name = "test_resave_after_load_external.bin"
        weights = self.get_dump_file(weights_name, clean=True)
        data = np.arange(100, dtype=np.float32)
        with open(weights, "wb") as fobj:
            fobj.write(data.tobytes())
        original_weights_bytes = data.tobytes()

        tensor = onnxl.TensorProto()
        tensor.name = "W"
        tensor.data_type = onnxl.TensorProto.FLOAT
        tensor.dims.extend([100])
        tensor.data_location = onnxl.TensorProto.EXTERNAL
        for key, value in (
            ("location", weights_name),
            ("offset", "0"),
            ("length", str(data.nbytes)),
        ):
            entry = tensor.external_data.add()
            entry.key = key
            entry.value = value

        # Load the data into raw_data while preserving the external metadata.
        tensor.load_external_data(os.path.dirname(weights))
        self.assertEqual(int(tensor.data_location), int(onnxl.TensorProto.EXTERNAL))
        self.assertEqual(len(tensor.raw_data), data.nbytes)
        self.assertEqual(len(tensor.external_data), 3)

        # Embed the tensor in a tiny model and save again with external data
        # pointing to the same weights file.
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Identity", ["W"], ["Y"])],
                "g",
                [],
                [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [100])],
                [tensor],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        resaved = self.get_dump_file("test_resave_after_load_external.onnx")
        onnxl.save(model, resaved, location=weights_name, save_as_external_data=True)

        # The on-disk weights file must be byte-identical to the original:
        # the in-memory buffer is assumed to match what was on disk.
        with open(weights, "rb") as fobj:
            new_weights_bytes = fobj.read()
        self.assertEqual(original_weights_bytes, new_weights_bytes)

        # The new ``.onnx`` file must not contain inline raw_data for the
        # external initializer; only the external_data metadata is kept.
        saved_no_ext = onnxl.load(resaved, load_external_data=False)
        ext_inits = [
            init
            for init in saved_no_ext.graph.initializer
            if int(init.data_location) == int(onnxl.TensorProto.EXTERNAL)
        ]
        self.assertEqual(len(ext_inits), 1)
        self.assertEqual(len(ext_inits[0].raw_data), 0)
        self.assertGreater(len(ext_inits[0].external_data), 0)

    def test_serialize_to_string_raw_data_callback_changed_size(self):
        data = np.arange(8, dtype=np.float32)
        tensor = onh.from_array(data, name="W")
        model = oh.make_model(
            oh.make_graph([], "g", [], [], [tensor]),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )

        replacement = np.arange(3, dtype=np.float32) + 10
        seen = []
        replacement_bytes = np.frombuffer(replacement.tobytes(), dtype=np.uint8)

        def callback(serialized_tensor, buffer, size_only):
            seen.append(
                (serialized_tensor.name, size_only, None if buffer is None else len(buffer))
            )
            if size_only:
                return replacement.nbytes
            serialized_tensor.ClearField("dims")
            serialized_tensor.dims.extend([replacement.size])
            np.copyto(np.asarray(buffer), replacement_bytes)
            return replacement.nbytes

        opts = onnxl.SerializeOptions()
        opts.raw_data_callback = callback
        serialized = model.SerializeToString(opts)

        self.assertEqual(seen, [("W", True, None), ("W", False, replacement.nbytes)])
        np.testing.assert_array_equal(onh.to_array(model.graph.initializer[0]), data)

        parsed = onnxl.ModelProto()
        parsed.ParseFromString(serialized)
        self.assertEqual(list(parsed.graph.initializer[0].dims), [3])
        np.testing.assert_array_equal(onh.to_array(parsed.graph.initializer[0]), replacement)

    def test_save_raw_data_callback_rewrites_external_tensor_changed_size(self):
        weights_name = "test_save_raw_data_callback.bin"
        weights = self.get_dump_file(weights_name, clean=True)
        data = np.arange(100, dtype=np.float32)
        with open(weights, "wb") as fobj:
            fobj.write(data.tobytes())

        tensor = onnxl.TensorProto()
        tensor.name = "W"
        tensor.data_type = onnxl.TensorProto.FLOAT
        tensor.dims.extend([100])
        tensor.data_location = onnxl.TensorProto.EXTERNAL
        for key, value in (
            ("location", weights_name),
            ("offset", "0"),
            ("length", str(data.nbytes)),
        ):
            entry = tensor.external_data.add()
            entry.key = key
            entry.value = value
        tensor.load_external_data(os.path.dirname(weights))

        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Identity", ["W"], ["Y"])],
                "g",
                [],
                [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [None])],
                [tensor],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )

        replacement = np.arange(25, dtype=np.float32) + 50
        seen = []
        replacement_bytes = np.frombuffer(replacement.tobytes(), dtype=np.uint8)
        saved = self.get_dump_file("test_save_raw_data_callback.onnx")

        def callback(serialized_tensor, buffer, size_only):
            seen.append(
                (serialized_tensor.name, size_only, None if buffer is None else len(buffer))
            )
            if size_only:
                return replacement.nbytes
            serialized_tensor.ClearField("dims")
            serialized_tensor.dims.extend([replacement.size])
            np.copyto(np.asarray(buffer), replacement_bytes)
            return replacement.nbytes

        onnxl.save(
            model,
            saved,
            location=weights_name,
            save_as_external_data=True,
            size_threshold=0,
            raw_data_callback=callback,
        )

        self.assertEqual(seen, [("W", True, None), ("W", False, replacement.nbytes)])
        self.assertEqual(list(model.graph.initializer[0].dims), [100])
        self.assertEqual(bytes(model.graph.initializer[0].raw_data), data.tobytes())

        with open(weights, "rb") as fobj:
            self.assertEqual(fobj.read(), replacement.tobytes())

        saved_no_ext = onnxl.load(saved, load_external_data=False)
        self.assertEqual(list(saved_no_ext.graph.initializer[0].dims), [25])
        self.assertEqual(int(saved_no_ext.graph.initializer[0].data_location), 1)
        entries = {e.key: e.value for e in saved_no_ext.graph.initializer[0].external_data}
        self.assertEqual(entries["location"], weights_name)
        self.assertEqual(entries["offset"], "0")
        self.assertEqual(entries["length"], str(replacement.nbytes))

        reloaded = onnxl.load(saved, location=weights)
        np.testing.assert_array_equal(onh.to_array(reloaded.graph.initializer[0]), replacement)

    def test_resave_external_mixed_with_inline_raises(self):
        # Mix one tensor that already has external_data metadata + loaded
        # raw_data (its metadata pins it to offset 0 in the weights file)
        # with a fresh inline initializer big enough to also be promoted
        # to external. Saving the model with ``save_as_external_data=True``
        # pointing at the same weights file must raise because the two
        # tensors cannot both live at offset 0 in the same file.
        weights_name = "test_resave_mixed.bin"
        weights = self.get_dump_file(weights_name, clean=True)
        data1 = np.arange(100, dtype=np.float32)
        with open(weights, "wb") as fobj:
            fobj.write(data1.tobytes())

        t1 = onnxl.TensorProto()
        t1.name = "W1"
        t1.data_type = onnxl.TensorProto.FLOAT
        t1.dims.extend([100])
        t1.data_location = onnxl.TensorProto.EXTERNAL
        for key, value in (
            ("location", weights_name),
            ("offset", "0"),
            ("length", str(data1.nbytes)),
        ):
            entry = t1.external_data.add()
            entry.key = key
            entry.value = value
        t1.load_external_data(os.path.dirname(weights))

        # Inline initializer big enough to cross the default size_threshold
        # (1024 bytes), so save_as_external_data will try to promote it.
        t2 = onh.from_array(np.arange(500, dtype=np.float32), name="W2")

        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Identity", ["W1"], ["Y"])],
                "g",
                [],
                [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [100])],
                [t1, t2],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        resaved = self.get_dump_file("test_resave_mixed.onnx")
        with self.assertRaises(RuntimeError):
            onnxl.save(model, resaved, location=weights_name, save_as_external_data=True)

    def test_loading_external_weights_reordered_metadata(self):
        source = self.get_dump_file("test_loading_external_weights_reordered.source.onnx")
        name = self.get_dump_file("test_loading_external_weights_reordered.onnx")
        weights = self.get_dump_file("test_loading_external_weights_reordered.data")
        model = self._get_model_with_initializers(oh, onnxl.numpy_helper)
        expected = [onh.to_array(i) for i in model.graph.initializer]
        onnxl.save(model, source, location=os.path.split(weights)[-1], save_as_external_data=True)
        rewrite = onnxl.load(source, load_external_data=False)

        for init in rewrite.graph.initializer:
            if init.data_location != onnxl.TensorProto.EXTERNAL:
                continue
            metadata = {entry.key: entry.value for entry in init.external_data}
            if "location" not in metadata or "offset" not in metadata:
                continue
            size_key = "size" if "size" in metadata else "length" if "length" in metadata else ""
            if not size_key:
                continue
            del init.external_data[:]
            entry = init.external_data.add()
            entry.key = "offset"
            entry.value = metadata["offset"]
            entry = init.external_data.add()
            entry.key = size_key
            entry.value = metadata[size_key]
            entry = init.external_data.add()
            entry.key = "location"
            entry.value = metadata["location"]
        onnxl.save(rewrite, name)

        proto = onnxl.load(name, location=weights)
        self.assertEqual(len(proto.graph.initializer), len(model.graph.initializer))

        def tweak(t):
            copy = onnxl.TensorProto()
            copy.ParseFromString(t.SerializeToString())
            return copy

        got = [onh.to_array(tweak(i)) for i in proto.graph.initializer]
        self.assertEqual(len(expected), len(got))
        for a, b in zip(expected, got):
            self.assertEqualArray(a, b)

    def test_save_external_data_default_location(self):
        # Verifies that save_as_external_data=True without an explicit location
        # places the weights file next to the model file with a ".data" suffix,
        # instead of creating a file literally named "None" in the cwd.
        onnx_path = self.get_dump_file("test_save_ext_default_src.onnx")
        name = self.get_dump_file("test_save_external_data_default_location.onnx")
        expected_data = name + ".data"
        if os.path.exists(expected_data):
            os.remove(expected_data)
        model = self._get_model_with_initializers(oh, onnxl.numpy_helper)
        onnxl.save(model, onnx_path)
        proto = onnxl.load(onnx_path)
        onnxl.save(proto, name, save_as_external_data=True)
        self.assertTrue(
            os.path.exists(expected_data),
            f"Expected weights file {expected_data} was not created.",
        )
        # The original model file directory must not contain a file named "None"
        self.assertFalse(
            os.path.exists(os.path.join(os.path.dirname(name), "None")),
            "A file named 'None' was incorrectly created in the model directory.",
        )
        # The saved model must be loadable by onnx
        reload = onnxl.load(name)
        self.assertEqual(len(reload.graph.initializer), len(model.graph.initializer))

    def test_save_external_data_after_metadata_only_load_raises_clear_error(self):
        onnx_path = self.get_dump_file("test_save_ext_metadata_only_src.onnx")
        name = self.get_dump_file("test_save_ext_metadata_only_dst.onnx")
        model = self._get_model_with_initializers(oh, onnxl.numpy_helper)
        onnxl.save(model, onnx_path, save_as_external_data=True)

        proto = onnxl.load(onnx_path, load_external_data=False)
        with self.assertRaises(RuntimeError) as cm:
            onnxl.save(proto, name, save_as_external_data=True)
        msg = str(cm.exception)
        self.assertIn("load_external_data=False", msg)
        self.assertIn("load_external_data=True", msg)
        self.assertIn("save_model_with_shared_external_data()", msg)

    def test_save_external_data_does_not_mutate_modelproto(self):
        # Verifies that saving to two files (model + external data) does not
        # modify the in-memory onnx_light ModelProto.
        onnx_path = self.get_dump_file("test_save_ext_no_mutation_src.onnx")
        name = self.get_dump_file("test_save_ext_no_mutation.onnx")
        model = self._get_model_with_initializers(oh, onnxl.numpy_helper)
        onnxl.save(model, onnx_path)
        proto = onnxl.load(onnx_path)
        before = proto.SerializeToString()
        onnxl.save(proto, name, save_as_external_data=True)
        after = proto.SerializeToString()
        self.assertEqual(before, after)

    def test_parallel_external_data_write(self):
        # Verifies that parallel external data writing produces
        # byte-for-byte identical output to sequential writing.
        onnx_path = self.get_dump_file("test_parallel_ext_write_src.onnx")
        name_seq = self.get_dump_file("test_parallel_ext_write_seq.onnx")
        name_par = self.get_dump_file("test_parallel_ext_write_par.onnx")
        model = self._get_model_with_initializers(oh, onnxl.numpy_helper)
        onnxl.save(model, onnx_path)
        proto = onnxl.load(onnx_path)

        onnxl.save(proto, name_seq, save_as_external_data=True)
        onnxl.save(proto, name_par, save_as_external_data=True, num_threads=4)

        with open(name_seq + ".data", "rb") as f:
            seq_bytes = f.read()
        with open(name_par + ".data", "rb") as f:
            par_bytes = f.read()
        self.assertEqual(len(seq_bytes), len(par_bytes))
        self.assertEqual(seq_bytes, par_bytes)

        # Both files must be loadable and have the same number of initializers
        r_seq = onnxl.load(name_seq)
        r_par = onnxl.load(name_par)
        self.assertEqual(len(r_seq.graph.initializer), len(r_par.graph.initializer))
        self.assertEqual(len(r_seq.graph.initializer), len(model.graph.initializer))

    def test_parallel_external_data_write_auto_threads(self):
        # num_threads=-1 means one thread per hardware core;
        # verify that the output is byte-for-byte identical to sequential writing.
        onnx_path = self.get_dump_file("test_parallel_ext_write_auto_src.onnx")
        name_seq = self.get_dump_file("test_parallel_ext_write_auto_seq.onnx")
        name_par = self.get_dump_file("test_parallel_ext_write_auto_par.onnx")
        model = self._get_model_with_initializers(oh, onnxl.numpy_helper)
        onnxl.save(model, onnx_path)
        proto = onnxl.load(onnx_path)

        onnxl.save(proto, name_seq, save_as_external_data=True)
        onnxl.save(proto, name_par, save_as_external_data=True, num_threads=-1)

        with open(name_seq + ".data", "rb") as f:
            seq_bytes = f.read()
        with open(name_par + ".data", "rb") as f:
            par_bytes = f.read()
        self.assertEqual(seq_bytes, par_bytes)

    def test_loading_with_location_keeps_non_parallel_default(self):
        class FakeModelProto:
            def __init__(self):
                self.calls = []

            def ParseFromFile(self, *args, **kwargs):
                self.calls.append((args, kwargs))

            def ParseFromString(self, *args, **kwargs):
                self.calls.append((args, kwargs))

        with patch.object(io_helper, "ModelProto", FakeModelProto):
            model = io_helper.load("model.onnx", location="model.data", num_threads=1)

        self.assertEqual(len(model.calls), 1)
        args, kwargs = model.calls[0]
        self.assertEqual(args, ("model.onnx",))
        self.assertEqual(kwargs, {"external_data_file": "model.data"})

    def test_loading_with_location_and_parallel_uses_parse_options(self):
        class FakeParseOptions:
            def __init__(self):
                self.skip_raw_data = False
                self.raw_data_threshold = -1
                self.num_threads = 1
                self.min_parallel_block_size = -1

        class FakeModelProto:
            def __init__(self):
                self.calls = []

            def ParseFromFile(self, *args, **kwargs):
                self.calls.append((args, kwargs))

            def ParseFromString(self, *args, **kwargs):
                self.calls.append((args, kwargs))

        with (
            patch.object(io_helper, "ParseOptions", FakeParseOptions),
            patch.object(io_helper, "ModelProto", FakeModelProto),
        ):
            model = io_helper.load("model.onnx", location="model.data", num_threads=2)

        self.assertEqual(len(model.calls), 1)
        args, kwargs = model.calls[0]
        self.assertEqual(args[0], "model.onnx")
        self.assertEqual(args[1].num_threads, 2)
        self.assertEqual(kwargs, {"external_data_file": "model.data"})

    def test_loading_without_location_keeps_non_parallel_default(self):
        class FakeModelProto:
            def __init__(self):
                self.calls = []

            def ParseFromFile(self, *args, **kwargs):
                self.calls.append((args, kwargs))

            def ParseFromString(self, *args, **kwargs):
                self.calls.append((args, kwargs))

        with patch.object(io_helper, "ModelProto", FakeModelProto):
            model = io_helper.load("model.onnx", num_threads=1)

        self.assertEqual(len(model.calls), 1)
        args, kwargs = model.calls[0]
        self.assertEqual(args, ("model.onnx",))
        self.assertEqual(kwargs, {})

    def test_loading_with_tiny_external_threshold_uses_parse_options(self):
        class FakeParseOptions:
            def __init__(self):
                self.skip_raw_data = False
                self.raw_data_threshold = -1
                self.num_threads = 1
                self.min_parallel_block_size = -1
                self.tiny_external_data_threshold = -1

        class FakeModelProto:
            def __init__(self):
                self.calls = []

            def ParseFromFile(self, *args, **kwargs):
                self.calls.append((args, kwargs))

            def ParseFromString(self, *args, **kwargs):
                self.calls.append((args, kwargs))

        with (
            patch.object(io_helper, "ParseOptions", FakeParseOptions),
            patch.object(io_helper, "ModelProto", FakeModelProto),
        ):
            model = io_helper.load("model.onnx", tiny_external_data_threshold=128)

        self.assertEqual(len(model.calls), 1)
        args, kwargs = model.calls[0]
        self.assertEqual(args[0], "model.onnx")
        self.assertEqual(args[1].tiny_external_data_threshold, 128)
        self.assertEqual(kwargs, {})

    def test_saving_without_external_data_keeps_non_parallel_default(self):
        class FakeModelProto:
            def __init__(self):
                self.calls = []

            def SerializeToFile(self, *args, **kwargs):
                self.calls.append((args, kwargs))

        with patch.object(io_helper, "ModelProto", FakeModelProto):
            model = FakeModelProto()
            io_helper.save(model, "model.onnx", num_threads=1)

        self.assertEqual(len(model.calls), 1)
        args, kwargs = model.calls[0]
        self.assertEqual(args, ("model.onnx",))
        self.assertEqual(kwargs, {})

    def test_saving_with_parallel_uses_serialize_options(self):
        class FakeSerializeOptions:
            def __init__(self):
                self.raw_data_threshold = -1
                self.num_threads = 1
                self.min_parallel_block_size = -1
                self.max_external_file_size = -1

        class FakeModelProto:
            def __init__(self):
                self.calls = []

            def SerializeToFile(self, *args, **kwargs):
                self.calls.append((args, kwargs))

        with (
            patch.object(io_helper, "SerializeOptions", FakeSerializeOptions),
            patch.object(io_helper, "ModelProto", FakeModelProto),
        ):
            model = FakeModelProto()
            io_helper.save(model, "model.onnx", num_threads=3, min_block_size=256)

        self.assertEqual(len(model.calls), 1)
        args, kwargs = model.calls[0]
        self.assertEqual(args[0], "model.onnx")
        self.assertEqual(args[1].raw_data_threshold, 1024)
        self.assertEqual(args[1].num_threads, 3)
        self.assertEqual(args[1].min_parallel_block_size, 256)
        self.assertEqual(args[1].max_external_file_size, 0)
        self.assertEqual(kwargs, {})

    def test_saving_with_external_data_uses_max_external_file_size(self):
        class FakeSerializeOptions:
            def __init__(self):
                self.raw_data_threshold = -1
                self.num_threads = 1
                self.min_parallel_block_size = -1
                self.max_external_file_size = -1

        class FakeModelProto:
            def __init__(self):
                self.calls = []

            def SerializeToFile(self, *args, **kwargs):
                self.calls.append((args, kwargs))

        with (
            patch.object(io_helper, "SerializeOptions", FakeSerializeOptions),
            patch.object(io_helper, "ModelProto", FakeModelProto),
        ):
            model = FakeModelProto()
            io_helper.save(
                model, "model.onnx", location="model.data", max_external_file_size=4096
            )

        self.assertEqual(len(model.calls), 1)
        args, kwargs = model.calls[0]
        self.assertEqual(args[0], "model.onnx")
        self.assertEqual(args[1].max_external_file_size, 4096)
        self.assertEqual(args[2], "model.data")
        self.assertEqual(kwargs, {})

    def test_loading_external_weights_split_files_explicit_location(self):
        # Verify that loading a model whose external data has been split across
        # multiple files works correctly when the primary data file is given
        # explicitly via the ``location`` parameter.
        name = self.get_dump_file("test_loading_split_ext_explicit.onnx")
        location = self.get_dump_file("test_loading_split_ext_explicit.data")
        model = self._get_model_with_initializers(oh, onnxl.numpy_helper)
        onnxl.save(model, name)
        src = onnxl.load(name)
        # Save with a small max_external_file_size to force splitting
        # (use a size smaller than the large initializer tensors ~786 KB each).
        onnxl.save(src, name, location=location, max_external_file_size=500_000)
        self.assertTrue(os.path.exists(location), "Primary data file was not created.")
        self.assertTrue(os.path.exists(location + ".1"), "Secondary data file was not created.")
        # Load with explicit location (all split files are resolved automatically)
        loaded = onnxl.load(name, location=location)
        self.assertEqual(len(loaded.graph.initializer), len(model.graph.initializer))

    def test_save_split_external_data_from_no_copy_model_same_location(self):
        # What this test validates:
        # 1) Build a model and save its weights externally into a single data file.
        # 2) Reload that model with no_copy=True so raw tensor bytes can be borrowed
        #    from the external file instead of copied in memory.
        # 3) Save again to the same external-data location while forcing split files.
        #    This rewrites/truncates the source file, so serialization must first
        #    materialize borrowed buffers and avoid reading overwritten bytes.
        # 4) Reload and compare initializer arrays with the original model to ensure
        #    tensor payload integrity is preserved.
        name = self.get_dump_file("test_split_ext_nocopy_same_location.onnx")
        location = self.get_dump_file("test_split_ext_nocopy_same_location.data")
        model = self._get_model_with_initializers(oh, onnxl.numpy_helper)
        onnxl.save(model, name)
        src = onnxl.load(name)
        onnxl.save(src, name, location=location)
        borrowed = onnxl.load(name, location=location, no_copy=True)
        onnxl.save(borrowed, name, location=location, max_external_file_size=500_000)
        self.assertTrue(os.path.exists(location), "Primary data file was not created.")
        self.assertTrue(os.path.exists(location + ".1"), "Secondary data file was not created.")
        loaded = onnxl.load(name, location=location)
        self.assertEqual(len(loaded.graph.initializer), len(model.graph.initializer))
        for i, expected in enumerate(model.graph.initializer):
            got = loaded.graph.initializer[i]
            np.testing.assert_array_equal(
                onnxl.numpy_helper.to_array(expected),
                onnxl.numpy_helper.to_array(got),
                err_msg=f"Mismatch at initializer {i}",
            )

    def test_save_single_external_file_alignment_change_from_no_copy(self):
        # Regression test: no-copy external-data reload + re-save to the same
        # single external-data file with a different alignment must preserve
        # tensor contents.
        name = self.get_dump_file("test_single_ext_align_nocopy_same_location.onnx")
        location = self.get_dump_file("test_single_ext_align_nocopy_same_location.data")
        model = self._get_model_with_initializers(oh, onnxl.numpy_helper)
        onnxl.save(model, name)
        src = onnxl.load(name)
        onnxl.save(src, name, location=location)

        borrowed = onnxl.ModelProto()
        popts = onnxl.ParseOptions()
        popts.no_copy = True
        borrowed.ParseFromFile(name, popts, location)

        sopts = onnxl.SerializeOptions()
        sopts.alignment = 4096
        borrowed.SerializeToFile(name, sopts, location)

        loaded = onnxl.load(name, location=location)
        self.assertEqual(len(loaded.graph.initializer), len(model.graph.initializer))
        for i, expected in enumerate(model.graph.initializer):
            got = loaded.graph.initializer[i]
            np.testing.assert_array_equal(
                onnxl.numpy_helper.to_array(expected),
                onnxl.numpy_helper.to_array(got),
                err_msg=f"Mismatch at initializer {i}",
            )

    def test_loading_external_weights_split_files_auto_discovery(self):
        # Verify that ``load_external_data=True`` without an explicit ``location``
        # automatically discovers the primary external data file and loads all
        # split data files correctly.
        name = self.get_dump_file("test_loading_split_ext_auto.onnx")
        location = self.get_dump_file("test_loading_split_ext_auto.data")
        model = self._get_model_with_initializers(oh, onnxl.numpy_helper)
        onnxl.save(model, name)
        src = onnxl.load(name)
        onnxl.save(src, name, location=location, max_external_file_size=500_000)
        self.assertTrue(os.path.exists(location), "Primary data file was not created.")
        self.assertTrue(os.path.exists(location + ".1"), "Secondary data file was not created.")
        # Load with auto-discovery: no explicit location required
        loaded = onnxl.load(name, load_external_data=True)
        self.assertEqual(len(loaded.graph.initializer), len(model.graph.initializer))

    def test_loading_external_weights_nested_graph_auto_discovery(self):
        # Verify that auto-discovery finds external data even when the only
        # external tensors are inside nested sub-graphs (If/Loop/Scan nodes),
        # with no external initializers at the top-level graph.
        TFLOAT = onnxl.TensorProto.FLOAT
        nested_init = onnxl.numpy_helper.from_array(
            rand(3, 5, 128, 64, seed=3).astype(np.float32), name="nested_weight"
        )
        then_graph = oh.make_graph(
            [oh.make_node("Identity", ["nested_weight"], ["result"])],
            "then_graph",
            [],
            [oh.make_tensor_value_info("result", TFLOAT, [None])],
            [nested_init],
        )
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "If", ["cond"], ["output"], then_branch=then_graph, else_branch=then_graph
                    )
                ],
                "outer_graph",
                [oh.make_tensor_value_info("cond", onnxl.TensorProto.BOOL, [])],
                [oh.make_tensor_value_info("output", TFLOAT, [None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        name = self.get_dump_file("test_loading_nested_ext_auto.onnx")
        location = self.get_dump_file("test_loading_nested_ext_auto.data")
        onnxl.save(model, name)
        src = onnxl.load(name)
        # Force splitting so at least two data files are created
        onnxl.save(src, name, location=location, max_external_file_size=500_000)
        self.assertTrue(
            os.path.exists(location) or os.path.exists(location + ".1"),
            "No external data file was created.",
        )
        # Load with auto-discovery; external data lives only in nested sub-graphs
        loaded = onnxl.load(name, load_external_data=True)
        if_node = loaded.graph.node[0]
        for j in range(len(if_node.attribute)):
            attr = if_node.attribute[j]
            init = attr.g.initializer[0]
            self.assertEqual(len(init.raw_data), 3 * 5 * 128 * 64 * 4)


@unittest.skipIf(
    not hasattr(os, "symlink") or os.name == "nt",
    "Symlinks require POSIX or elevated privileges on Windows",
)
class TestSaveExternalDataSymlinkProtection(ExtTestCase):
    """Verifies that saving external data rejects symlink targets (GHSA-8qff-7g33-75mx).

    GHSA-8qff-7g33-75mx: TOCTOU arbitrary file read/write in save_external_data.
    An attacker who places a symlink at the external-data target path before the
    model is saved could redirect writes to an arbitrary file.  onnx-light guards
    against this at two layers:
      1. Python: validate_external_data_path rejects symlinks at the target path.
      2. C++:    FileWriteStream / validate_weights_file_is_next_to_model reject
                 symlinks at open time (defense-in-depth, closes the TOCTOU window).
    """

    def setUp(self):
        import tempfile

        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_simple_model(self):
        """Returns a small model with a large enough initializer to trigger external data."""
        TFLOAT = TensorProto.FLOAT
        init = onh.from_array(np.ones((16, 16), dtype=np.float32), name="weight")
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Identity", ["weight"], ["out"])],
                "g",
                [],
                [oh.make_tensor_value_info("out", TFLOAT, [16, 16])],
                [init],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        return model

    def test_save_rejects_symlink_at_explicit_location(self):
        """Saving must fail when the explicit external-data location is a symlink."""
        sensitive = os.path.join(self.tmpdir, "sensitive.txt")
        with open(sensitive, "w") as f:
            f.write("SENSITIVE DATA")

        model_path = os.path.join(self.tmpdir, "model.onnx")
        data_path = os.path.join(self.tmpdir, "model.data")
        # Place a symlink at the external-data target path before saving.
        os.symlink(sensitive, data_path)

        model = self._make_simple_model()
        with self.assertRaises((ValueError, RuntimeError)):
            onnxl.save(model, model_path, save_as_external_data=True, location="model.data")

        # The sensitive file must not be modified.
        with open(sensitive) as f:
            self.assertEqual(f.read(), "SENSITIVE DATA")

    def test_save_rejects_symlink_at_default_location(self):
        """Saving must fail when the default external-data location is a symlink."""
        sensitive = os.path.join(self.tmpdir, "sensitive2.txt")
        with open(sensitive, "w") as f:
            f.write("SENSITIVE DATA 2")

        model_path = os.path.join(self.tmpdir, "model2.onnx")
        default_data_path = model_path + ".data"
        # Place a symlink at the default external-data path before saving.
        os.symlink(sensitive, default_data_path)

        model = self._make_simple_model()
        with self.assertRaises((ValueError, RuntimeError)):
            onnxl.save(model, model_path, save_as_external_data=True)

        # The sensitive file must not be modified.
        with open(sensitive) as f:
            self.assertEqual(f.read(), "SENSITIVE DATA 2")


if __name__ == "__main__":
    unittest.main(verbosity=2)
