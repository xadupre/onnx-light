"""Tests for the ``SerializeFormat`` option on ``ParseOptions`` / ``SerializeOptions``.

The flatbuffer format used by ``onnxruntime`` (``.ort`` files) is exposed through
``SerializeFormat.ORT_FLATBUFFERS`` but is not implemented yet.  The current API
contract is:

* the enum members ``ONNX`` and ``ORT_FLATBUFFERS`` exist;
* the default value of ``ParseOptions.format`` and ``SerializeOptions.format``
  is ``SerializeFormat.ONNX``;
* using ``SerializeFormat.ONNX`` keeps the existing protobuf round-trip working
  unchanged;
* using ``SerializeFormat.ORT_FLATBUFFERS`` raises a clear error from every
  parse/serialize entry point until the flatbuffer path is implemented.

Once the flatbuffer path lands, the ``test_ort_flatbuffers_round_trip_with_onnxruntime``
test verifies that a model serialized with ``SerializeFormat.ORT_FLATBUFFERS``
loads and runs in ``onnxruntime`` for several opset versions.  Until then it is
skipped so the suite stays green.
"""

from __future__ import annotations

import os
import unittest

import numpy as np

import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh
from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx import TensorProto


def _make_simple_model(opset: int = 18) -> tuple[onnxl.ModelProto, np.ndarray, np.ndarray]:
    """Builds a tiny ``MatMul`` model and an input/expected-output pair."""
    rng = np.random.default_rng(0)
    w = rng.standard_normal((4, 3)).astype(np.float32)
    tfloat = TensorProto.FLOAT
    model = oh.make_model(
        oh.make_graph(
            [oh.make_node("MatMul", ["X", "W"], ["Y"])],
            "g",
            [oh.make_tensor_value_info("X", tfloat, [None, 4])],
            [oh.make_tensor_value_info("Y", tfloat, [None, 3])],
            [onh.from_array(w, name="W")],
        ),
        opset_imports=[oh.make_opsetid("", opset)],
        ir_version=9,
    )
    x = rng.standard_normal((2, 4)).astype(np.float32)
    expected = x @ w
    return model, x, expected


class TestSerializeFormat(ExtTestCase):
    def test_enum_values_exist(self) -> None:
        self.assertTrue(hasattr(onnxl, "SerializeFormat"))
        self.assertTrue(hasattr(onnxl.SerializeFormat, "ONNX"))
        self.assertTrue(hasattr(onnxl.SerializeFormat, "ORT_FLATBUFFERS"))

    def test_defaults_are_onnx(self) -> None:
        self.assertEqual(onnxl.ParseOptions().format, onnxl.SerializeFormat.ONNX)
        self.assertEqual(onnxl.SerializeOptions().format, onnxl.SerializeFormat.ONNX)

    def test_format_is_writable(self) -> None:
        sopts = onnxl.SerializeOptions()
        sopts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
        self.assertEqual(sopts.format, onnxl.SerializeFormat.ORT_FLATBUFFERS)
        popts = onnxl.ParseOptions()
        popts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
        self.assertEqual(popts.format, onnxl.SerializeFormat.ORT_FLATBUFFERS)

    def test_onnx_format_round_trip(self) -> None:
        # SerializeFormat.ONNX (the default) must keep the existing behaviour.
        model, _, _ = _make_simple_model()
        sopts = onnxl.SerializeOptions()
        sopts.format = onnxl.SerializeFormat.ONNX
        data = model.SerializeToString(sopts)

        popts = onnxl.ParseOptions()
        popts.format = onnxl.SerializeFormat.ONNX
        parsed = onnxl.ModelProto()
        parsed.ParseFromString(data, popts)
        self.assertEqual(parsed.graph.name, "g")

    def test_ort_flatbuffers_serialize_to_string_raises(self) -> None:
        model, _, _ = _make_simple_model()
        sopts = onnxl.SerializeOptions()
        sopts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
        with self.assertRaises(RuntimeError):
            model.SerializeToString(sopts)

    def test_ort_flatbuffers_serialize_to_file_raises(self) -> None:
        model, _, _ = _make_simple_model()
        path = self.get_dump_file("test_ort_serialize_to_file_unimpl.ort")
        sopts = onnxl.SerializeOptions()
        sopts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
        with self.assertRaises(RuntimeError):
            model.SerializeToFile(path, sopts)

    def test_ort_flatbuffers_parse_from_string_raises(self) -> None:
        model, _, _ = _make_simple_model()
        # Produce a valid ONNX-format buffer, then ask the parser to interpret
        # it as an ORT flatbuffer file. The parser must refuse instead of
        # silently producing garbage.
        data = model.SerializeToString()
        popts = onnxl.ParseOptions()
        popts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
        parsed = onnxl.ModelProto()
        with self.assertRaises(RuntimeError):
            parsed.ParseFromString(data, popts)

    def test_ort_flatbuffers_parse_from_file_raises(self) -> None:
        model, _, _ = _make_simple_model()
        path = self.get_dump_file("test_ort_parse_from_file_unimpl.onnx")
        model.SerializeToFile(path)
        popts = onnxl.ParseOptions()
        popts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
        parsed = onnxl.ModelProto()
        with self.assertRaises(RuntimeError):
            parsed.ParseFromFile(path, popts)

    def test_ort_flatbuffers_serialize_to_string_raises_with_parallelization(self) -> None:
        # Combining the unimplemented ORT flatbuffer writer with parallel writing
        # must still produce a clean RuntimeError: the format guard is expected
        # to fire before any thread pool is spun up. This prevents silently
        # falling back to a parallel ONNX-protobuf write while the user asked
        # for the flatbuffer format.
        model, _, _ = _make_simple_model()
        for num_threads in (2, 4, -1):
            with self.subTest(num_threads=num_threads):
                sopts = onnxl.SerializeOptions()
                sopts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
                sopts.num_threads = num_threads
                with self.assertRaises(RuntimeError):
                    model.SerializeToString(sopts)

    def test_ort_flatbuffers_serialize_to_string_raises_with_alignment(self) -> None:
        # Same contract for the alignment knob: the format guard takes
        # precedence so the user gets a clear "format not implemented" error
        # instead of an aligned ONNX-protobuf payload.
        model, _, _ = _make_simple_model()
        for alignment in (16, 64, 4096):
            with self.subTest(alignment=alignment):
                sopts = onnxl.SerializeOptions()
                sopts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
                sopts.alignment = alignment
                with self.assertRaises(RuntimeError):
                    model.SerializeToString(sopts)

    def test_ort_flatbuffers_serialize_to_string_raises_with_parallel_and_alignment(self) -> None:
        # Both knobs together must still surface the format guard.
        model, _, _ = _make_simple_model()
        sopts = onnxl.SerializeOptions()
        sopts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
        sopts.num_threads = 4
        sopts.alignment = 4096
        with self.assertRaises(RuntimeError):
            model.SerializeToString(sopts)

    def test_ort_flatbuffers_serialize_to_file_raises_with_parallelization(self) -> None:
        model, _, _ = _make_simple_model()
        for num_threads in (2, 4, -1):
            with self.subTest(num_threads=num_threads):
                path = self.get_dump_file(
                    f"test_ort_serialize_to_file_unimpl_threads_{num_threads}.ort"
                )
                sopts = onnxl.SerializeOptions()
                sopts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
                sopts.num_threads = num_threads
                with self.assertRaises(RuntimeError):
                    model.SerializeToFile(path, sopts)

    def test_ort_flatbuffers_serialize_to_file_raises_with_alignment(self) -> None:
        model, _, _ = _make_simple_model()
        for alignment in (16, 64, 4096):
            with self.subTest(alignment=alignment):
                path = self.get_dump_file(
                    f"test_ort_serialize_to_file_unimpl_align_{alignment}.ort"
                )
                sopts = onnxl.SerializeOptions()
                sopts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
                sopts.alignment = alignment
                with self.assertRaises(RuntimeError):
                    model.SerializeToFile(path, sopts)

    def test_ort_flatbuffers_serialize_to_file_raises_with_parallel_and_alignment(self) -> None:
        model, _, _ = _make_simple_model()
        path = self.get_dump_file("test_ort_serialize_to_file_unimpl_parallel_align.ort")
        sopts = onnxl.SerializeOptions()
        sopts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
        sopts.num_threads = 4
        sopts.alignment = 4096
        with self.assertRaises(RuntimeError):
            model.SerializeToFile(path, sopts)

    def test_ort_flatbuffers_parse_from_string_raises_with_parallelization(self) -> None:
        # Symmetric coverage for the parser: combining the unimplemented
        # ORT flatbuffer reader with parallel reading must raise cleanly too.
        model, _, _ = _make_simple_model()
        data = model.SerializeToString()
        for num_threads in (2, 4, -1):
            with self.subTest(num_threads=num_threads):
                popts = onnxl.ParseOptions()
                popts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
                popts.num_threads = num_threads
                parsed = onnxl.ModelProto()
                with self.assertRaises(RuntimeError):
                    parsed.ParseFromString(data, popts)

    def test_ort_flatbuffers_parse_from_string_raises_with_alignment(self) -> None:
        model, _, _ = _make_simple_model()
        data = model.SerializeToString()
        for alignment in (16, 64, 4096):
            with self.subTest(alignment=alignment):
                popts = onnxl.ParseOptions()
                popts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
                popts.alignment = alignment
                parsed = onnxl.ModelProto()
                with self.assertRaises(RuntimeError):
                    parsed.ParseFromString(data, popts)

    def test_ort_flatbuffers_parse_from_string_zero_recursion_depth_raises(self) -> None:
        # max_recursion_depth must be > 0 for the ORT flatbuffer path.  The
        # guard fires before the "not implemented" stub so that when the real
        # parser lands the protection is already wired up.
        model, _, _ = _make_simple_model()
        data = model.SerializeToString()
        popts = onnxl.ParseOptions()
        popts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
        popts.max_recursion_depth = 0
        parsed = onnxl.ModelProto()
        with self.assertRaisesRegex(RuntimeError, "max_recursion_depth"):
            parsed.ParseFromString(data, popts)

    def test_ort_flatbuffers_parse_from_string_negative_recursion_depth_raises(self) -> None:
        # A negative max_recursion_depth is also rejected.
        model, _, _ = _make_simple_model()
        data = model.SerializeToString()
        popts = onnxl.ParseOptions()
        popts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
        popts.max_recursion_depth = -1
        parsed = onnxl.ModelProto()
        with self.assertRaisesRegex(RuntimeError, "max_recursion_depth"):
            parsed.ParseFromString(data, popts)

    def test_ort_flatbuffers_parse_from_file_zero_recursion_depth_raises(self) -> None:
        # ParseFromFile path enforces max_recursion_depth > 0 too.
        model, _, _ = _make_simple_model()
        path = self.get_dump_file("test_ort_parse_from_file_depth0.onnx")
        model.SerializeToFile(path)
        popts = onnxl.ParseOptions()
        popts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
        popts.max_recursion_depth = 0
        parsed = onnxl.ModelProto()
        with self.assertRaisesRegex(RuntimeError, "max_recursion_depth"):
            parsed.ParseFromFile(path, popts)

    def test_parse_options_max_tensor_size_bytes_default_is_zero(self) -> None:
        # Default value for max_tensor_size_bytes must be 0 (no limit).
        popts = onnxl.ParseOptions()
        self.assertEqual(popts.max_tensor_size_bytes, 0)

    def test_max_tensor_size_bytes_raw_data_throws(self) -> None:
        # Parsing a TensorProto whose raw_data exceeds the limit must raise.
        import numpy as np

        w = np.ones((5,), dtype=np.float32)  # 20 bytes
        tp = onh.from_array(w, name="w")
        data = tp.SerializeToString()
        popts = onnxl.ParseOptions()
        popts.max_tensor_size_bytes = 10  # 10 bytes < 20 bytes
        parsed = onnxl.TensorProto()
        with self.assertRaisesRegex(RuntimeError, "max_tensor_size_bytes"):
            parsed.ParseFromString(data, popts)

    def test_max_tensor_size_bytes_raw_data_exact_limit_allowed(self) -> None:
        # Parsing a TensorProto whose raw_data equals the limit must succeed.
        import numpy as np

        w = np.ones((5,), dtype=np.float32)  # 20 bytes
        tp = onh.from_array(w, name="w")
        data = tp.SerializeToString()
        popts = onnxl.ParseOptions()
        popts.max_tensor_size_bytes = 20  # Exactly the raw_data size — must pass.
        parsed = onnxl.TensorProto()
        parsed.ParseFromString(data, popts)
        np.testing.assert_array_equal(onh.to_array(parsed), w)

    def test_max_tensor_size_bytes_zero_means_no_limit(self) -> None:
        # max_tensor_size_bytes == 0 disables the limit.
        import numpy as np

        w = np.ones((100,), dtype=np.float32)  # 400 bytes
        tp = onh.from_array(w, name="w")
        data = tp.SerializeToString()
        popts = onnxl.ParseOptions()
        popts.max_tensor_size_bytes = 0  # No limit.
        parsed = onnxl.TensorProto()
        parsed.ParseFromString(data, popts)  # Must not raise.

    def test_ort_flatbuffers_negative_max_tensor_size_bytes_raises(self) -> None:
        # A negative max_tensor_size_bytes must be rejected for the ORT flatbuffer path.
        model, _, _ = _make_simple_model()
        data = model.SerializeToString()
        popts = onnxl.ParseOptions()
        popts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
        popts.max_tensor_size_bytes = -1
        parsed = onnxl.ModelProto()
        with self.assertRaisesRegex(RuntimeError, "max_tensor_size_bytes"):
            parsed.ParseFromString(data, popts)

    @unittest.skip(
        "Saving to onnxruntime flatbuffer format is not implemented yet; "
        "this test will be enabled once SerializeFormat.ORT_FLATBUFFERS produces "
        "files that load and run in onnxruntime."
    )
    def test_ort_flatbuffers_round_trip_with_onnxruntime(self) -> None:
        # When the ORT flatbuffer writer is implemented, this test verifies
        # that the produced file works with onnxruntime for multiple opsets.
        try:
            import onnxruntime as ort  # noqa: F401
        except ImportError:
            self.skipTest("onnxruntime is not available")

        for opset in (15, 18, 21):
            with self.subTest(opset=opset):
                model, x, expected = _make_simple_model(opset=opset)
                path = self.get_dump_file(f"test_ort_format_opset{opset}.ort")
                sopts = onnxl.SerializeOptions()
                sopts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
                model.SerializeToFile(path, sopts)
                self.assertTrue(os.path.exists(path))

                sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
                (got,) = sess.run(None, {"X": x})
                np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
