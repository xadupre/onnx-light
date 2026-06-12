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
