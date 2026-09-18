"""Checks native ORT parsing against independently produced ONNX Runtime files."""

import gc
import os
import struct
import tempfile
import unittest

import numpy as np
import onnxruntime

import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as helper
import onnx_light.onnx.numpy_helper as numpy_helper
from onnx_light.ext_test_case import ExtTestCase


def make_model():
    """Builds a model with an initializer and an intermediate value."""
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("MatMul", ["X", "W"], ["M"]),
                helper.make_node("Relu", ["M"], ["Y"]),
            ],
            "reader",
            [helper.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [None, 2])],
            [helper.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [None, 3])],
            [numpy_helper.from_array(np.arange(6, dtype=np.float32).reshape(2, 3), name="W")],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=9,
    )
    model.producer_name = "native-reader-test"
    model.metadata_props.add(key="purpose", value="round-trip")
    return model


def serialize_ort(model):
    """Serializes a model using the native ORT writer."""
    options = onnxl.SerializeOptions()
    options.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
    return model.SerializeToString(options)


def parse_options():
    """Returns explicit ORT parsing options."""
    options = onnxl.ParseOptions()
    options.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
    return options


def runtime_options():
    """Returns deterministic, unoptimized ONNX Runtime session options."""
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    return options


class TestParseOrtFormat(ExtTestCase):
    def assert_model_runs(self, model):
        x = np.array([[1, -1], [2, 3]], dtype=np.float32)
        expected = np.maximum(x @ np.arange(6, dtype=np.float32).reshape(2, 3), 0)
        session = onnxruntime.InferenceSession(
            model.SerializeToString(),
            sess_options=runtime_options(),
            providers=["CPUExecutionProvider"],
        )
        np.testing.assert_array_equal(session.run(None, {"X": x})[0], expected)

    def test_parse_native_buffer(self):
        original = make_model()
        parsed = onnxl.ModelProto()
        parsed.ParseFromString(serialize_ort(original), parse_options())
        self.assertEqual(parsed.ir_version, original.ir_version)
        self.assertEqual(parsed.producer_name, original.producer_name)
        self.assertEqual(
            [(v.key, v.value) for v in parsed.metadata_props], [("purpose", "round-trip")]
        )
        self.assertEqual([node.op_type for node in parsed.graph.node], ["MatMul", "Relu"])
        self.assert_model_runs(parsed)
        reloaded = onnxl.ModelProto()
        reloaded.ParseFromString(serialize_ort(parsed), parse_options())
        self.assert_model_runs(reloaded)

    def test_parse_onnxruntime_export(self):
        original = make_model()
        with tempfile.TemporaryDirectory() as folder:
            for level in (
                onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL,
                onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL,
            ):
                with self.subTest(level=level):
                    path = os.path.join(folder, "runtime.ort")
                    options = runtime_options()
                    options.graph_optimization_level = level
                    options.optimized_model_filepath = path
                    options.add_session_config_entry("session.save_model_format", "ORT")
                    onnxruntime.InferenceSession(
                        original.SerializeToString(), options, providers=["CPUExecutionProvider"]
                    )
                    parsed = onnxl.ModelProto()
                    parsed.ParseFromFile(path, parse_options())
                    self.assert_model_runs(parsed)
                    with open(path, "rb") as stream:
                        data = stream.read()
                    parsed.ParseFromString(data, parse_options())
                    self.assert_model_runs(parsed)

    def test_parse_onnxruntime_control_flow(self):
        then_branch = helper.make_graph(
            [helper.make_node("Identity", ["X"], ["branch_output"])],
            "then",
            [],
            [helper.make_tensor_value_info("branch_output", onnxl.TensorProto.FLOAT, [2])],
        )
        else_branch = helper.make_graph(
            [helper.make_node("Neg", ["X"], ["branch_output"])],
            "else",
            [],
            [helper.make_tensor_value_info("branch_output", onnxl.TensorProto.FLOAT, [2])],
        )
        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node(
                        "If", ["cond"], ["Y"], then_branch=then_branch, else_branch=else_branch
                    )
                ],
                "conditional",
                [
                    helper.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [2]),
                    helper.make_tensor_value_info("cond", onnxl.TensorProto.BOOL, []),
                ],
                [helper.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [2])],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
            ir_version=9,
        )
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "if.ort")
            options = runtime_options()
            options.optimized_model_filepath = path
            options.add_session_config_entry("session.save_model_format", "ORT")
            onnxruntime.InferenceSession(
                model.SerializeToString(), options, providers=["CPUExecutionProvider"]
            )
            parsed = onnxl.ModelProto()
            parsed.ParseFromFile(path, parse_options())
        session = onnxruntime.InferenceSession(
            parsed.SerializeToString(),
            sess_options=runtime_options(),
            providers=["CPUExecutionProvider"],
        )
        x = np.array([-3, 5], dtype=np.float32)
        for condition in (False, True):
            np.testing.assert_array_equal(
                session.run(None, {"X": x, "cond": np.array(condition)})[0],
                x if condition else -x,
            )

    def test_parse_sequence_types(self):
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node("SequenceConstruct", ["X", "Y"], ["S"])],
                "sequence",
                [
                    helper.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [2]),
                    helper.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [2]),
                ],
                [helper.make_tensor_sequence_value_info("S", onnxl.TensorProto.FLOAT, [2])],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
            ir_version=9,
        )
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "sequence.ort")
            options = runtime_options()
            options.optimized_model_filepath = path
            options.add_session_config_entry("session.save_model_format", "ORT")
            onnxruntime.InferenceSession(
                model.SerializeToString(), options, providers=["CPUExecutionProvider"]
            )
            with open(path, "rb") as stream:
                runtime_data = stream.read()
        for data in (serialize_ort(model), runtime_data):
            parsed = onnxl.ModelProto()
            parsed.ParseFromString(data, parse_options())
            self.assertTrue(parsed.graph.output[0].type.has_sequence_type())
            session = onnxruntime.InferenceSession(
                parsed.SerializeToString(),
                sess_options=runtime_options(),
                providers=["CPUExecutionProvider"],
            )
            x = np.array([1, 2], dtype=np.float32)
            y = np.array([3, 4], dtype=np.float32)
            result = session.run(None, {"X": x, "Y": y})[0]
            np.testing.assert_array_equal(result[0], x)
            np.testing.assert_array_equal(result[1], y)

    def test_parse_file_modes_and_lifetime(self):
        original = make_model()
        data = serialize_ort(original)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "model.ort")
            with open(path, "wb") as stream:
                stream.write(data)
            for mode in (
                onnxl.FileLoadMode.AUTO,
                onnxl.FileLoadMode.IFSTREAM,
                onnxl.FileLoadMode.MMAP,
            ):
                for no_copy in (False, True):
                    with self.subTest(mode=mode, no_copy=no_copy):
                        options = parse_options()
                        options.file_load_mode = mode
                        options.no_copy = no_copy
                        parsed = onnxl.ModelProto()
                        parsed.ParseFromFile(path, options)
                        gc.collect()
                        self.assert_model_runs(parsed)
        self.assert_model_runs(parsed)

    def test_parse_buffer_lifetime_and_options(self):
        for threads in (1, 2, -1):
            with self.subTest(threads=threads):
                options = parse_options()
                options.no_copy = True
                options.num_threads = threads
                options.alignment = 64
                parsed = onnxl.ModelProto()
                parsed.ParseFromString(serialize_ort(make_model()), options)
                gc.collect()
                self.assert_model_runs(parsed)

    def test_tensor_limit_boundary(self):
        data = serialize_ort(make_model())
        options = parse_options()
        options.max_tensor_size_bytes = 24
        parsed = onnxl.ModelProto()
        parsed.ParseFromString(data, options)
        self.assert_model_runs(parsed)
        options.max_tensor_size_bytes = 23
        before = parsed.SerializeToString()
        with self.assertRaisesRegex(RuntimeError, "max_tensor_size_bytes"):
            parsed.ParseFromString(data, options)
        self.assertEqual(before, parsed.SerializeToString())

    def test_string_tensor_limit(self):
        values = np.array(["abc", "abc\u00e9"])
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node("Identity", ["W"], ["Y"])],
                "strings",
                [],
                [helper.make_tensor_value_info("Y", onnxl.TensorProto.STRING, [2])],
                [numpy_helper.from_array(values, name="W")],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
            ir_version=9,
        )
        data = serialize_ort(model)
        options = parse_options()
        options.max_tensor_size_bytes = 8
        parsed = onnxl.ModelProto()
        parsed.ParseFromString(data, options)
        np.testing.assert_array_equal(numpy_helper.to_array(parsed.graph.initializer[0]), values)
        options.max_tensor_size_bytes = 7
        with self.assertRaisesRegex(RuntimeError, "max_tensor_size_bytes"):
            parsed.ParseFromString(data, options)

    def test_scalar_and_empty_tensors(self):
        for values in (np.array(-3, dtype=np.int64), np.empty((0, 3), dtype=np.float32)):
            with self.subTest(shape=values.shape):
                tensor = numpy_helper.from_array(values, name="W")
                model = helper.make_model(
                    helper.make_graph(
                        [helper.make_node("Identity", ["W"], ["Y"])],
                        "tensor",
                        [],
                        [
                            helper.make_tensor_value_info(
                                "Y", tensor.data_type, list(values.shape)
                            )
                        ],
                        [tensor],
                    ),
                    opset_imports=[helper.make_opsetid("", 18)],
                    ir_version=9,
                )
                parsed = onnxl.ModelProto()
                parsed.ParseFromString(serialize_ort(model), parse_options())
                np.testing.assert_array_equal(
                    numpy_helper.to_array(parsed.graph.initializer[0]), values
                )
                session = onnxruntime.InferenceSession(
                    parsed.SerializeToString(),
                    sess_options=runtime_options(),
                    providers=["CPUExecutionProvider"],
                )
                np.testing.assert_array_equal(session.run(None, {})[0], values)

    def test_skip_raw_data(self):
        options = parse_options()
        options.skip_raw_data = True
        parsed = onnxl.ModelProto()
        parsed.ParseFromString(serialize_ort(make_model()), options)
        self.assertEqual(len(parsed.graph.initializer), 1)
        self.assertEqual(list(parsed.graph.initializer[0].dims), [2, 3])
        self.assertEqual(len(parsed.graph.initializer[0].raw_data), 0)

    def test_callbacks_and_ownership(self):
        seen_tensors = []
        seen_nodes = []
        released = []

        def tensor_callback(tensor, graph):
            seen_tensors.append((tensor.name, graph is not None))
            return lambda: released.append("W")

        def node_callback(node, graph):
            seen_nodes.append((node.op_type, graph is not None))
            node.doc_string = "visited"

        options = parse_options()
        options.raw_data_callback = tensor_callback
        options.node_callback = node_callback
        parsed = onnxl.ModelProto()
        parsed.ParseFromString(serialize_ort(make_model()), options)
        self.assertEqual(seen_tensors, [("W", True)])
        self.assertEqual(seen_nodes, [("MatMul", True), ("Relu", True)])
        self.assertEqual([node.doc_string for node in parsed.graph.node], ["visited", "visited"])
        self.assert_model_runs(parsed)
        self.assertEqual(released, [])
        del parsed
        gc.collect()
        self.assertEqual(released, ["W"])

    def test_callback_failure_preserves_target(self):
        target = make_model()
        before = target.SerializeToString()

        def reject_node(node, graph):
            raise ValueError("reader callback rejected the node")

        options = parse_options()
        options.node_callback = reject_node
        with self.assertRaisesRegex(ValueError, "reader callback rejected"):
            target.ParseFromString(serialize_ort(make_model()), options)
        self.assertEqual(target.SerializeToString(), before)

    def test_rejects_standalone_tensor(self):
        with self.assertRaisesRegex(RuntimeError, "ModelProto"):
            onnxl.TensorProto().ParseFromString(serialize_ort(make_model()), parse_options())

    def test_malformed_buffer_preserves_target(self):
        data = serialize_ort(make_model())
        bad_identifier = bytearray(data)
        bad_identifier[4:8] = b"BAD!"
        bad_root = bytearray(data)
        struct.pack_into("<I", bad_root, 0, 0xFFFFFFFC)
        bad_vtable = bytearray(data)
        root = struct.unpack_from("<I", data)[0]
        struct.pack_into("<i", bad_vtable, root, 0x7FFFFFFF)
        target = make_model()
        before = target.SerializeToString()
        for bad in (
            b"",
            data[:7],
            data[: root + 2],
            bytes(bad_identifier),
            bytes(bad_root),
            bytes(bad_vtable),
        ):
            with self.subTest(size=len(bad), prefix=bad[:8]):
                with self.assertRaises(RuntimeError):
                    target.ParseFromString(bad, parse_options())
                self.assertEqual(target.SerializeToString(), before)

    def test_recursion_limit(self):
        data = serialize_ort(make_model())
        options = parse_options()
        options.max_recursion_depth = 1
        with self.assertRaisesRegex(RuntimeError, "(?i)recursion|depth"):
            onnxl.ModelProto().ParseFromString(data, options)


if __name__ == "__main__":
    unittest.main(verbosity=2)
