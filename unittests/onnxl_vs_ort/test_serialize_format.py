"""Tests for the ``SerializeFormat`` option on ``ParseOptions`` / ``SerializeOptions``.

Model serialization produces ORT FlatBuffers that ONNX Runtime can load and
execute. Parsing accepts ORT FlatBuffers and rejects invalid buffers and options.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import onnxruntime

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

    def assert_ort_model(self, data, x, expected) -> None:
        session = onnxruntime.InferenceSession(data, providers=["CPUExecutionProvider"])
        (got,) = session.run(None, {"X": x})
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
        if isinstance(data, str) or data[4:8] == b"ORTM":
            options = onnxl.ParseOptions()
            options.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
            parsed = onnxl.ModelProto()
            if isinstance(data, str):
                parsed.ParseFromFile(data, options)
            else:
                parsed.ParseFromString(data, options)
            session = onnxruntime.InferenceSession(
                parsed.SerializeToString(), providers=["CPUExecutionProvider"]
            )
            (got,) = session.run(None, {"X": x})
            np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)

    def test_ort_flatbuffers_serialize_to_string(self) -> None:
        model, x, expected = _make_simple_model()
        sopts = onnxl.SerializeOptions()
        sopts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
        data = model.SerializeToString(sopts)
        self.assertEqual(data[4:8], b"ORTM")
        self.assertEqual(model.SerializeSize(sopts).size(), len(data))
        self.assert_ort_model(data, x, expected)

    def test_ort_flatbuffers_serialize_to_file_descriptor(self) -> None:
        model, x, expected = _make_simple_model()
        sopts = onnxl.SerializeOptions()
        sopts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
        with tempfile.TemporaryFile() as stream:
            model.SerializeToFileDescriptor(stream.fileno(), sopts)
            stream.seek(0)
            data = stream.read()
        self.assertEqual(data, model.SerializeToString(sopts))
        self.assert_ort_model(data, x, expected)

    def test_ort_flatbuffers_size_limit(self) -> None:
        model, x, expected = _make_simple_model()
        sopts = onnxl.SerializeOptions()
        sopts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
        data = model.SerializeToString(sopts)
        sopts.max_serialized_size_bytes = len(data)
        self.assertEqual(model.SerializeToString(sopts), data)
        self.assert_ort_model(data, x, expected)
        for limit in (-1, 1, len(data) - 1):
            with self.subTest(limit=limit):
                sopts.max_serialized_size_bytes = limit
                with self.assertRaisesRegex(RuntimeError, "max_serialized_size_bytes"):
                    model.SerializeToString(sopts)
                with tempfile.TemporaryFile() as stream:
                    with self.assertRaisesRegex(RuntimeError, "max_serialized_size_bytes"):
                        model.SerializeToFileDescriptor(stream.fileno(), sopts)
                    self.assertEqual(os.fstat(stream.fileno()).st_size, 0)

    def test_ort_flatbuffers_rejects_standalone_tensor(self) -> None:
        tensor = onh.from_array(np.ones(3, dtype=np.float32))
        sopts = onnxl.SerializeOptions()
        sopts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
        with self.assertRaisesRegex(RuntimeError, "ModelProto"):
            tensor.SerializeToString(sopts)
        with self.assertRaisesRegex(RuntimeError, "ModelProto"):
            tensor.SerializeSize(sopts)

    def test_ort_flatbuffers_callbacks_preserve_model(self) -> None:
        model, x, expected = _make_simple_model()
        before = model.SerializeToString()
        calls = []

        def rewrite_weights(tensor, graph, buffer, size_only):
            values = onh.to_array(tensor) * 2
            calls.append((graph.name, size_only))
            if not size_only:
                buffer[:] = values.view(np.uint8).reshape(-1)
            return values.nbytes

        sopts = onnxl.SerializeOptions()
        sopts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
        sopts.raw_data_callback = rewrite_weights
        self.assert_ort_model(model.SerializeToString(sopts), x, expected * 2)
        self.assertEqual(calls, [("g", True), ("g", False)])
        self.assertEqual(model.SerializeToString(), before)

    def test_ort_flatbuffers_inline_alignment(self) -> None:
        model, _, _ = _make_simple_model()
        raw = bytes(model.graph.initializer[0].raw_data)
        sopts = onnxl.SerializeOptions()
        sopts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
        for alignment in (16, 64, 4096):
            with self.subTest(alignment=alignment):
                sopts.alignment = alignment
                data = model.SerializeToString(sopts)
                offset = data.find(raw)
                self.assertGreaterEqual(offset, 8)
                self.assertEqual(offset % alignment, 0)

    def test_ort_flatbuffers_rejects_unloaded_external_data(self) -> None:
        model, _, _ = _make_simple_model()
        tensor = model.graph.initializer[0]
        tensor.external_data.add(key="location", value="missing.bin")
        tensor.data_location = TensorProto.EXTERNAL
        tensor.ClearField("raw_data")
        sopts = onnxl.SerializeOptions()
        sopts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
        with self.assertRaisesRegex(RuntimeError, "(?i)external"):
            model.SerializeToString(sopts)

    def test_ort_flatbuffers_inlines_loaded_external_data(self) -> None:
        model, x, expected = _make_simple_model()
        tensor = model.graph.initializer[0]
        tensor.external_data.add(key="location", value="missing.bin")
        tensor.data_location = TensorProto.EXTERNAL
        before = model.SerializeToString()
        sopts = onnxl.SerializeOptions()
        sopts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
        self.assert_ort_model(model.SerializeToString(sopts), x, expected)
        self.assertEqual(model.SerializeToString(), before)

    def test_ort_flatbuffers_preserves_overridable_initializer_shape(self) -> None:
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Identity", ["W"], ["Y"])],
                "overridable",
                [oh.make_tensor_value_info("W", TensorProto.FLOAT, [None])],
                [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [None])],
                [onh.from_array(np.array([1, 2], dtype=np.float32), name="W")],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        sopts = onnxl.SerializeOptions()
        sopts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
        session = onnxruntime.InferenceSession(
            model.SerializeToString(sopts), providers=["CPUExecutionProvider"]
        )
        np.testing.assert_array_equal(
            session.run(None, {})[0], np.array([1, 2], dtype=np.float32)
        )
        override = np.array([4, 5, 6], dtype=np.float32)
        np.testing.assert_array_equal(session.run(None, {"W": override})[0], override)

    def test_ort_flatbuffers_rejects_external_output(self) -> None:
        model, _, _ = _make_simple_model()
        sopts = onnxl.SerializeOptions()
        sopts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
        with (
            tempfile.TemporaryDirectory() as folder,
            self.assertRaisesRegex(RuntimeError, "external"),
        ):
            model.SerializeToFile(
                os.path.join(folder, "model.ort"), sopts, os.path.join(folder, "weights.bin")
            )

    def test_ort_flatbuffers_typed_initializers(self) -> None:
        for dtype in (
            np.float32,
            np.float64,
            np.float16,
            np.int8,
            np.uint8,
            np.int16,
            np.uint16,
            np.int32,
            np.uint32,
            np.int64,
            np.uint64,
            np.bool_,
            np.str_,
        ):
            for raw in (False, True):
                if dtype == np.str_ and raw:
                    continue
                with self.subTest(dtype=dtype, raw=raw):
                    if dtype == np.str_:
                        values = np.array(["abc", "caf\u00e9"], dtype=dtype)
                    elif np.issubdtype(dtype, np.integer):
                        info = np.iinfo(dtype)
                        values = np.array([info.min, info.max], dtype=dtype)
                    elif np.issubdtype(dtype, np.floating):
                        values = np.array([-2.5, 3.75], dtype=dtype)
                    else:
                        values = np.array([False, True], dtype=dtype)
                    elem_type = onh.from_array(values).data_type
                    tensor = oh.make_tensor("W", elem_type, [2], values, raw=raw)
                    model = oh.make_model(
                        oh.make_graph(
                            [oh.make_node("Identity", ["W"], ["Y"])],
                            "typed",
                            [],
                            [oh.make_tensor_value_info("Y", elem_type, [2])],
                            [tensor],
                        ),
                        opset_imports=[oh.make_opsetid("", 18)],
                        ir_version=9,
                    )
                    sopts = onnxl.SerializeOptions()
                    sopts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
                    session = onnxruntime.InferenceSession(
                        model.SerializeToString(sopts), providers=["CPUExecutionProvider"]
                    )
                    np.testing.assert_array_equal(session.run(None, {})[0], values)
                    popts = onnxl.ParseOptions()
                    popts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
                    parsed = onnxl.ModelProto()
                    parsed.ParseFromString(model.SerializeToString(sopts), popts)
                    np.testing.assert_array_equal(
                        onh.to_array(parsed.graph.initializer[0]), values
                    )

    def test_ort_flatbuffers_multi_node_and_variadic_inputs(self) -> None:
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Relu", ["X"], ["R"]),
                    oh.make_node("Concat", ["R", "R", "X"], ["C"], axis=0),
                    oh.make_node("Add", ["C", "C"], ["Y"]),
                ],
                "chain",
                [oh.make_tensor_value_info("X", TensorProto.FLOAT, [2])],
                [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [6])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        x = np.array([-1, 2], dtype=np.float32)
        expected = np.concatenate([np.maximum(x, 0), np.maximum(x, 0), x]) * 2
        sopts = onnxl.SerializeOptions()
        sopts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
        self.assert_ort_model(model.SerializeToString(sopts), x, expected)

    def test_ort_flatbuffers_attributes_and_optional_inputs(self) -> None:
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("LeakyRelu", ["X"], ["R"], alpha=0.25),
                    oh.make_node("Pad", ["R", "pads", ""], ["P"], mode="edge"),
                    oh.make_node("Transpose", ["P"], ["Y"], perm=[1, 0]),
                ],
                "attributes",
                [oh.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])],
                [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 2])],
                [onh.from_array(np.array([0, 1, 0, 1], dtype=np.int64), name="pads")],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        x = np.array([[-1, 2], [-3, 4]], dtype=np.float32)
        expected = np.pad(np.where(x >= 0, x, x * 0.25), ((0, 0), (1, 1)), mode="edge").T
        sopts = onnxl.SerializeOptions()
        sopts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
        self.assert_ort_model(model.SerializeToString(sopts), x, expected)

    def test_ort_flatbuffers_constant_tensor_attribute(self) -> None:
        values = np.array([-1, 2], dtype=np.float32)
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Constant", [], ["C"], value=onh.from_array(values)),
                    oh.make_node("Add", ["X", "C"], ["Y"]),
                ],
                "constant",
                [oh.make_tensor_value_info("X", TensorProto.FLOAT, [2])],
                [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [2])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        x = np.array([3, 5], dtype=np.float32)
        sopts = onnxl.SerializeOptions()
        sopts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
        self.assert_ort_model(model.SerializeToString(sopts), x, x + values)

    def test_ort_flatbuffers_subgraph_capture(self) -> None:
        then_branch = oh.make_graph(
            [oh.make_node("Relu", ["X"], ["positive"])],
            "then",
            [],
            [oh.make_tensor_value_info("positive", TensorProto.FLOAT, [2])],
        )
        else_branch = oh.make_graph(
            [oh.make_node("Neg", ["X"], ["negative"])],
            "else",
            [],
            [oh.make_tensor_value_info("negative", TensorProto.FLOAT, [2])],
        )
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "If", ["cond"], ["Y"], then_branch=then_branch, else_branch=else_branch
                    )
                ],
                "conditional",
                [
                    oh.make_tensor_value_info("cond", TensorProto.BOOL, []),
                    oh.make_tensor_value_info("X", TensorProto.FLOAT, [2]),
                ],
                [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [2])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        before = model.SerializeToString()
        sopts = onnxl.SerializeOptions()
        sopts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
        session = onnxruntime.InferenceSession(
            model.SerializeToString(sopts), providers=["CPUExecutionProvider"]
        )
        x = np.array([-1, 2], dtype=np.float32)
        for condition in (False, True):
            expected = np.maximum(x, 0) if condition else -x
            np.testing.assert_array_equal(
                session.run(None, {"X": x, "cond": np.array(condition)})[0], expected
            )
        self.assertEqual(model.SerializeToString(), before)
        popts = onnxl.ParseOptions()
        popts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
        parsed = onnxl.ModelProto()
        parsed.ParseFromString(model.SerializeToString(sopts), popts)
        session = onnxruntime.InferenceSession(
            parsed.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        for condition in (False, True):
            expected = np.maximum(x, 0) if condition else -x
            np.testing.assert_array_equal(
                session.run(None, {"X": x, "cond": np.array(condition)})[0], expected
            )

    def test_ort_flatbuffers_loop_carried_and_scan_outputs(self) -> None:
        body = oh.make_graph(
            [
                oh.make_node("Identity", ["keep"], ["next_keep"]),
                oh.make_node("Add", ["state", "step"], ["next_state"]),
                oh.make_node("Identity", ["next_state"], ["scan"]),
            ],
            "body",
            [
                oh.make_tensor_value_info("iteration", TensorProto.INT64, []),
                oh.make_tensor_value_info("keep", TensorProto.BOOL, []),
                oh.make_tensor_value_info("state", TensorProto.FLOAT, [2]),
            ],
            [
                oh.make_tensor_value_info("next_keep", TensorProto.BOOL, []),
                oh.make_tensor_value_info("next_state", TensorProto.FLOAT, [2]),
                oh.make_tensor_value_info("scan", TensorProto.FLOAT, [2]),
            ],
        )
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Loop", ["count", "condition", "X"], ["Y", "S"], body=body)],
                "loop",
                [oh.make_tensor_value_info("X", TensorProto.FLOAT, [2])],
                [
                    oh.make_tensor_value_info("Y", TensorProto.FLOAT, [2]),
                    oh.make_tensor_value_info("S", TensorProto.FLOAT, [3, 2]),
                ],
                [
                    onh.from_array(np.array(3, dtype=np.int64), name="count"),
                    onh.from_array(np.array(True), name="condition"),
                    onh.from_array(np.ones(2, dtype=np.float32), name="step"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        sopts = onnxl.SerializeOptions()
        sopts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
        session = onnxruntime.InferenceSession(
            model.SerializeToString(sopts), providers=["CPUExecutionProvider"]
        )
        x = np.array([1, 2], dtype=np.float32)
        final, scan = session.run(None, {"X": x})
        np.testing.assert_array_equal(final, x + 3)
        np.testing.assert_array_equal(scan, np.stack([x + 1, x + 2, x + 3]))
        popts = onnxl.ParseOptions()
        popts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
        parsed = onnxl.ModelProto()
        parsed.ParseFromString(model.SerializeToString(sopts), popts)
        session = onnxruntime.InferenceSession(
            parsed.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        final, scan = session.run(None, {"X": x})
        np.testing.assert_array_equal(final, x + 3)
        np.testing.assert_array_equal(scan, np.stack([x + 1, x + 2, x + 3]))

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

    def test_ort_flatbuffers_serialize_to_string_with_parallelization(self) -> None:
        model, x, expected = _make_simple_model()
        for num_threads in (2, 4, -1):
            with self.subTest(num_threads=num_threads):
                sopts = onnxl.SerializeOptions()
                sopts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
                sopts.num_threads = num_threads
                self.assert_ort_model(model.SerializeToString(sopts), x, expected)

    def test_ort_flatbuffers_serialize_to_string_with_alignment(self) -> None:
        model, x, expected = _make_simple_model()
        for alignment in (16, 64, 4096):
            with self.subTest(alignment=alignment):
                sopts = onnxl.SerializeOptions()
                sopts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
                sopts.alignment = alignment
                self.assert_ort_model(model.SerializeToString(sopts), x, expected)

    def test_ort_flatbuffers_serialize_to_string_with_parallel_and_alignment(self) -> None:
        model, x, expected = _make_simple_model()
        sopts = onnxl.SerializeOptions()
        sopts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
        sopts.num_threads = 4
        sopts.alignment = 4096
        self.assert_ort_model(model.SerializeToString(sopts), x, expected)

    def test_ort_flatbuffers_serialize_to_file_with_parallelization(self) -> None:
        model, x, expected = _make_simple_model()
        for num_threads in (2, 4, -1):
            with self.subTest(num_threads=num_threads):
                path = self.get_dump_file(f"test_ort_serialize_to_file_threads_{num_threads}.ort")
                sopts = onnxl.SerializeOptions()
                sopts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
                sopts.num_threads = num_threads
                model.SerializeToFile(path, sopts)
                self.assert_ort_model(path, x, expected)

    def test_ort_flatbuffers_serialize_to_file_with_alignment(self) -> None:
        model, x, expected = _make_simple_model()
        for alignment in (16, 64, 4096):
            with self.subTest(alignment=alignment):
                path = self.get_dump_file(f"test_ort_serialize_to_file_align_{alignment}.ort")
                sopts = onnxl.SerializeOptions()
                sopts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
                sopts.alignment = alignment
                model.SerializeToFile(path, sopts)
                self.assert_ort_model(path, x, expected)

    def test_ort_flatbuffers_serialize_to_file_with_parallel_and_alignment(self) -> None:
        model, x, expected = _make_simple_model()
        path = self.get_dump_file("test_ort_serialize_to_file_parallel_align.ort")
        sopts = onnxl.SerializeOptions()
        sopts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
        sopts.num_threads = 4
        sopts.alignment = 4096
        model.SerializeToFile(path, sopts)
        self.assert_ort_model(path, x, expected)

    def test_ort_flatbuffers_parse_from_string_with_parallelization(self) -> None:
        model, x, expected = _make_simple_model()
        sopts = onnxl.SerializeOptions()
        sopts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
        data = model.SerializeToString(sopts)
        for num_threads in (2, 4, -1):
            with self.subTest(num_threads=num_threads):
                popts = onnxl.ParseOptions()
                popts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
                popts.num_threads = num_threads
                parsed = onnxl.ModelProto()
                parsed.ParseFromString(data, popts)
                self.assert_ort_model(parsed.SerializeToString(), x, expected)

    def test_ort_flatbuffers_parse_from_string_with_alignment(self) -> None:
        model, x, expected = _make_simple_model()
        sopts = onnxl.SerializeOptions()
        sopts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
        data = model.SerializeToString(sopts)
        for alignment in (16, 64, 4096):
            with self.subTest(alignment=alignment):
                popts = onnxl.ParseOptions()
                popts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
                popts.alignment = alignment
                parsed = onnxl.ModelProto()
                parsed.ParseFromString(data, popts)
                self.assert_ort_model(parsed.SerializeToString(), x, expected)

    def test_ort_flatbuffers_parse_from_string_zero_recursion_depth_raises(self) -> None:
        # Invalid options are rejected before decoding starts.
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

    def test_serialize_options_max_serialized_size_bytes_default_is_zero(self) -> None:
        # Default value for max_serialized_size_bytes must be 0 (no limit).
        sopts = onnxl.SerializeOptions()
        self.assertEqual(sopts.max_serialized_size_bytes, 0)

    def test_max_serialized_size_bytes_throws(self) -> None:
        # Serializing a tensor above the configured cap must raise.
        import numpy as np

        w = np.ones((5,), dtype=np.float32)  # 20 bytes raw data (+ protobuf overhead)
        tp = onh.from_array(w, name="w")
        sopts = onnxl.SerializeOptions()
        sopts.max_serialized_size_bytes = 10
        with self.assertRaisesRegex(RuntimeError, "max_serialized_size_bytes"):
            tp.SerializeToString(sopts)

    def test_max_serialized_size_bytes_zero_means_no_limit(self) -> None:
        # max_serialized_size_bytes == 0 disables the limit.
        import numpy as np

        w = np.ones((100,), dtype=np.float32)
        tp = onh.from_array(w, name="w")
        sopts = onnxl.SerializeOptions()
        sopts.max_serialized_size_bytes = 0
        data = tp.SerializeToString(sopts)
        self.assertGreater(len(data), 0)

    def test_negative_max_serialized_size_bytes_raises(self) -> None:
        # A negative max_serialized_size_bytes must be rejected.
        import numpy as np

        w = np.ones((5,), dtype=np.float32)
        tp = onh.from_array(w, name="w")
        sopts = onnxl.SerializeOptions()
        sopts.max_serialized_size_bytes = -1
        with self.assertRaisesRegex(RuntimeError, "max_serialized_size_bytes"):
            tp.SerializeToString(sopts)

    def test_max_serialized_size_bytes_exact_limit_allowed(self) -> None:
        # A cap equal to the exact serialized size must be accepted, while a cap
        # one byte smaller must be rejected (boundary of the ``<=`` check).
        import numpy as np

        w = np.ones((5,), dtype=np.float32)
        tp = onh.from_array(w, name="w")
        exact = len(tp.SerializeToString())

        sopts = onnxl.SerializeOptions()
        sopts.max_serialized_size_bytes = exact
        data = tp.SerializeToString(sopts)
        self.assertEqual(len(data), exact)

        sopts.max_serialized_size_bytes = exact - 1
        with self.assertRaisesRegex(RuntimeError, "max_serialized_size_bytes"):
            tp.SerializeToString(sopts)

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

    def test_ort_flatbuffers_round_trip_with_onnxruntime(self) -> None:
        for opset in (15, 18, 21):
            with self.subTest(opset=opset):
                model, x, expected = _make_simple_model(opset=opset)
                path = self.get_dump_file(f"test_ort_format_opset{opset}.ort")
                sopts = onnxl.SerializeOptions()
                sopts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
                model.SerializeToFile(path, sopts)
                self.assertTrue(os.path.exists(path))

                self.assert_ort_model(path, x, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
