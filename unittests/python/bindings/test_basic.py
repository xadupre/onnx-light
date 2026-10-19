# source: https://github.com/onnx/onnx/blob/main/onnx/test/basic_test.py
from __future__ import annotations

import io
import os
import pathlib
import tempfile
import unittest
from typing import get_type_hints
import numpy as np
import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh
from onnx_light.onnx import defs, parser
from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx_proto._io_helper import load


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


def _pipeline_with_node(doc_string: str) -> onnxl.PipelineProto:
    """Creates a pipeline whose prefill model contains one node."""
    model = _simple_model()
    node = model.graph.add_node("Identity", ["x"], ["y"])
    node.doc_string = doc_string
    return onnxl.PipelineProto(prefill=model)


class TestIOModel(ExtTestCase):
    """Tests for saving and loading ModelProto (protobuf format only)."""

    def test_load_input_annotation(self) -> None:
        self.assertIs(onnxl.load, load)
        self.assertIs(onnxl.load_model, load)
        self.assertEqual(get_type_hints(load)["f"], str | pathlib.Path | bytes)

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

    def test_save_and_load_model_with_inferred_protobuf_format(self) -> None:
        """Tests protobuf inference for arbitrary filenames and path types."""
        proto = _simple_model()
        with tempfile.TemporaryDirectory() as temp_dir:
            for filename in ("model.pb", "model.bin", "model"):
                for path_type in (str, pathlib.Path):
                    with self.subTest(filename=filename, path_type=path_type):
                        model_path = path_type(os.path.join(temp_dir, filename))
                        onnxl.save(proto, model_path)
                        self.assertEqual(proto, onnxl.load(model_path))
                        self.assertEqual(proto, onnxl.load(model_path, format="protobuf"))

    def test_save_and_load_model_when_input_is_file_descriptor(self) -> None:
        proto = _simple_model()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model_fd.onnx")
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            if hasattr(os, "O_BINARY"):
                flags |= os.O_BINARY
            fd = os.open(model_path, flags, 0o666)
            try:
                proto.SerializeToFileDescriptor(fd)
            finally:
                os.close(fd)
            loaded_proto = onnxl.load(model_path)
            self.assertEqual(proto, loaded_proto)

    def test_save_to_file_descriptor_and_load(self) -> None:
        """Tests that save_to_file_descriptor() writes a valid model."""
        proto = _simple_model()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model_fd2.onnx")
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            if hasattr(os, "O_BINARY"):
                flags |= os.O_BINARY
            fd = os.open(model_path, flags, 0o666)
            try:
                onnxl.save_to_file_descriptor(proto, fd)
            finally:
                os.close(fd)
            loaded_proto = onnxl.load(model_path)
            self.assertEqual(proto, loaded_proto)

    def test_save_to_file_descriptor_thread_options(self) -> None:
        """Tests that ignored threading options preserve bytes and descriptor ownership."""
        proto = _simple_model()
        tensor = _simple_tensor()
        tensor.raw_data = np.arange(24, dtype=np.float32).tobytes()
        tensor.ClearField("float_data")
        proto.graph.initializer.append(tensor)
        expected = proto.SerializeToString()
        for num_threads in (-2, -1, 0, 1, 2, 4):
            for min_block_size in (0, 1, 1024):
                with (
                    self.subTest(num_threads=num_threads, min_block_size=min_block_size),
                    tempfile.TemporaryFile() as handle,
                ):
                    onnxl.save_to_file_descriptor(
                        proto,
                        handle.fileno(),
                        num_threads=num_threads,
                        min_block_size=min_block_size,
                    )
                    handle.seek(0)
                    self.assertEqual(handle.read(), expected)

    def test_save_model_when_input_is_file_like_object(self) -> None:
        """Tests that save() accepts a file-like object with a write() method."""
        proto = _simple_model()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model_stream.onnx")
            with open(model_path, "wb") as handle:
                onnxl.save(proto, handle)
            loaded_proto = onnxl.load(model_path)
            self.assertEqual(proto, loaded_proto)

    def test_serialize_to_ostream_and_load(self) -> None:
        """Tests that SerializeToOstream() writes valid bytes to a file-like object."""
        proto = _simple_model()
        buf = io.BytesIO()
        proto.SerializeToOstream(buf)
        loaded_proto = onnxl.load(buf.getvalue())
        self.assertEqual(proto, loaded_proto)

    def test_serialize_to_ostream_capital_s_and_load(self) -> None:
        """Tests that SerializeToOStream() (capital S) works as an alias."""
        proto = _simple_model()
        buf = io.BytesIO()
        proto.SerializeToOStream(buf)
        loaded_proto = onnxl.load(buf.getvalue())
        self.assertEqual(proto, loaded_proto)

    def test_save_model_file_like_with_external_data_raises(self) -> None:
        """Tests that save() raises when a file-like object is used with external data."""
        proto = _simple_model()
        buf = io.BytesIO()
        with self.assertRaises(ValueError) as ctx:
            onnxl.save(proto, buf, save_as_external_data=True)
        self.assertIn("file-like object", str(ctx.exception))

    def test_load_model_with_format_protobuf(self) -> None:
        """Tests that load() accepts format='protobuf' parameter."""
        proto = _simple_model()
        proto_bytes = proto.SerializeToString()
        loaded_proto = onnxl.load(proto_bytes, format="protobuf")
        self.assertEqual(proto, loaded_proto)

    def test_load_model_with_format_none(self) -> None:
        """Tests that load() accepts format=None parameter."""
        proto = _simple_model()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model.onnx")
            onnxl.save(proto, model_path)
            loaded_proto = onnxl.load(model_path, format=None)
            self.assertEqual(proto, loaded_proto)

    def test_load_model_with_unsupported_format_raises(self) -> None:
        """Tests that load() raises ValueError for unsupported formats."""
        proto = _simple_model()
        proto_bytes = proto.SerializeToString()
        with self.assertRaises(ValueError) as ctx:
            onnxl.load(proto_bytes, format="ort")
        self.assertIn("Unsupported format", str(ctx.exception))
        self.assertIn("onnx-light only supports", str(ctx.exception))


class TestIOTensor(ExtTestCase):
    """Tests standalone TensorProto files and streams."""

    def test_tensor_pb_files_contain_standalone_tensor_messages(self) -> None:
        tensors = [
            _simple_tensor(),
            onh.from_array(np.arange(6, dtype=np.float32).reshape(2, 3), name="raw"),
            onh.from_array(np.array(["alpha", "beta"], dtype=object), name="strings"),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            for tensor in tensors:
                for path_type in (str, pathlib.Path):
                    with self.subTest(tensor=tensor.name, path_type=path_type):
                        path = path_type(os.path.join(temp_dir, "input_0.pb"))
                        onnxl.save_tensor(tensor, path)
                        loaded = onnxl.load_tensor(path)
                        self.assertIsInstance(loaded, onnxl.TensorProto)
                        self.assertEqual(tensor, loaded)
                        np.testing.assert_array_equal(onh.to_array(tensor), onh.to_array(loaded))
                        direct = onnxl.TensorProto()
                        direct.ParseFromString(pathlib.Path(path).read_bytes())
                        self.assertEqual(tensor, direct)
                        self.assertEqual(tensor, onnxl.load_tensor(path, format="protobuf"))

    def test_tensor_helpers_accept_binary_streams_without_closing_them(self) -> None:
        tensor = _simple_tensor()
        stream = io.BytesIO()
        onnxl.save_tensor(tensor, stream)
        self.assertFalse(stream.closed)
        stream.seek(0)
        self.assertEqual(tensor, onnxl.load_tensor(stream))
        self.assertFalse(stream.closed)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir, "output_0.pb")
            with path.open("wb") as handle:
                onnxl.save_tensor(tensor, handle)
                self.assertFalse(handle.closed)
            with path.open("rb") as handle:
                self.assertEqual(tensor, onnxl.load_tensor(handle))
                self.assertFalse(handle.closed)

    def test_tensor_textproto_inference_and_explicit_format(self) -> None:
        tensor = _simple_tensor()
        with tempfile.TemporaryDirectory() as temp_dir:
            for filename, format in [("tensor.textproto", None), ("tensor.pb", "textproto")]:
                with self.subTest(filename=filename, format=format):
                    path = pathlib.Path(temp_dir, filename)
                    onnxl.save_tensor(tensor, path, format=format)
                    self.assertEqual(tensor, onnxl.load_tensor(path, format=format))
        stream = io.BytesIO()
        onnxl.save_tensor(tensor, stream, format="textproto")
        stream.seek(0)
        self.assertEqual(tensor, onnxl.load_tensor(stream, format="textproto"))

    def test_tensor_helpers_reject_invalid_inputs_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir, "tensor.pb")
            path.write_bytes(b"existing")
            with self.assertRaises(TypeError):
                onnxl.save_tensor(_simple_model(), path)
            with self.assertRaises(ValueError):
                onnxl.save_tensor(_simple_tensor(), path, format="ort")
            self.assertEqual(path.read_bytes(), b"existing")
            with self.assertRaises(ValueError):
                onnxl.load_tensor(path, format="ort")
            with self.assertRaises(TypeError):
                onnxl.load_tensor(1)
            with self.assertRaises(TypeError):
                onnxl.save_tensor(_simple_tensor(), 1)
            with self.assertRaises(FileNotFoundError):
                onnxl.load_tensor(pathlib.Path(temp_dir, "missing.pb"))
            path.write_bytes(b"\x0a\x05\x01")
            with self.assertRaises(RuntimeError):
                onnxl.load_tensor(path)

    def test_load_tensor_when_input_is_bytes(self) -> None:
        proto = _simple_tensor()
        proto_bytes = proto.SerializeToString()
        loaded_proto = onnxl.TensorProto()
        loaded_proto.ParseFromString(proto_bytes)
        self.assertEqual(proto.SerializeToString(), loaded_proto.SerializeToString())

    def test_save_and_load_tensor_when_input_is_file_descriptor(self) -> None:
        proto = _simple_tensor()
        with tempfile.TemporaryDirectory() as temp_dir:
            tensor_path = os.path.join(temp_dir, "tensor.data")
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            if hasattr(os, "O_BINARY"):
                flags |= os.O_BINARY
            fd = os.open(tensor_path, flags, 0o666)
            try:
                proto.SerializeToFileDescriptor(fd)
            finally:
                os.close(fd)
            loaded_proto = onnxl.TensorProto()
            loaded_proto.ParseFromString(pathlib.Path(tensor_path).read_bytes())
            self.assertEqual(proto.SerializeToString(), loaded_proto.SerializeToString())

    def test_serialize_tensor_to_ostream_and_parse(self) -> None:
        proto = _simple_tensor()
        buffer = io.BytesIO()
        proto.SerializeToOstream(buffer)
        loaded_proto = onnxl.TensorProto()
        loaded_proto.ParseFromString(buffer.getvalue())
        self.assertEqual(proto.SerializeToString(), loaded_proto.SerializeToString())


class TestIOPipeline(ExtTestCase):
    """Tests pipeline serialization behavior."""

    @staticmethod
    def _parse(serialized: bytes) -> onnxl.PipelineProto:
        pipeline = onnxl.PipelineProto()
        pipeline.ParseFromString(serialized)
        return pipeline

    def test_serialize_callbacks_apply_to_string_file_and_descriptor(self) -> None:
        """Checks callback edits of equal and different sizes on every output path."""
        for original, replacement in [("old", "new"), ("", "expanded")]:
            with self.subTest(original=original, replacement=replacement):
                pipeline = _pipeline_with_node(original)
                options = onnxl.SerializeOptions()
                options.node_callback = lambda node, graph: setattr(
                    node, "doc_string", replacement
                )

                restored = self._parse(pipeline.SerializeToString(options))
                self.assertEqual(restored.prefill.graph.node[0].doc_string, replacement)

                with tempfile.TemporaryDirectory() as temp_dir:
                    file_path = pathlib.Path(temp_dir, "pipeline.onnx")
                    pipeline.SerializeToFile(str(file_path), options)
                    restored = self._parse(file_path.read_bytes())
                    self.assertEqual(restored.prefill.graph.node[0].doc_string, replacement)

                    fd_path = pathlib.Path(temp_dir, "pipeline-fd.onnx")
                    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
                    if hasattr(os, "O_BINARY"):
                        flags |= os.O_BINARY
                    fd = os.open(fd_path, flags, 0o666)
                    try:
                        pipeline.SerializeToFileDescriptor(fd, options)
                    finally:
                        os.close(fd)
                    restored = self._parse(fd_path.read_bytes())
                    self.assertEqual(restored.prefill.graph.node[0].doc_string, replacement)

                self.assertEqual(pipeline.prefill.graph.node[0].doc_string, original)

    def test_external_data_rejection_preserves_existing_files(self) -> None:
        """Checks unsupported external output is rejected before files are opened."""
        pipeline = _pipeline_with_node("")
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = pathlib.Path(temp_dir, "pipeline.onnx")
            weights_path = pathlib.Path(temp_dir, "pipeline.data")
            file_path.write_bytes(b"existing pipeline")
            weights_path.write_bytes(b"existing weights")

            with self.assertRaises(RuntimeError):
                pipeline.SerializeToFile(str(file_path), external_data_file=str(weights_path))

            self.assertEqual(file_path.read_bytes(), b"existing pipeline")
            self.assertEqual(weights_path.read_bytes(), b"existing weights")

    def test_raw_data_callback_applies_to_string_file_and_descriptor(self) -> None:
        """Checks raw-data callback edits are retained on every output path."""
        for replacement in [b"WXYZ", b"expanded"]:
            with self.subTest(replacement=replacement):
                pipeline = _pipeline_with_node("")
                tensor = pipeline.prefill.graph.initializer.add()
                tensor.name = "weight"
                tensor.data_type = onnxl.TensorProto.UINT8
                tensor.dims.extend([4])
                tensor.raw_data = b"abcd"

                def callback(serialized_tensor, graph, buffer, size_only):
                    if size_only:
                        return len(replacement)
                    serialized_tensor.ClearField("dims")
                    serialized_tensor.dims.extend([len(replacement)])
                    np.copyto(np.asarray(buffer), np.frombuffer(replacement, dtype=np.uint8))
                    return len(replacement)

                options = onnxl.SerializeOptions()
                options.raw_data_callback = callback

                restored = self._parse(pipeline.SerializeToString(options))
                self.assertEqual(restored.prefill.graph.initializer[0].raw_data, replacement)

                with tempfile.TemporaryDirectory() as temp_dir:
                    file_path = pathlib.Path(temp_dir, "pipeline.onnx")
                    pipeline.SerializeToFile(str(file_path), options)
                    restored = self._parse(file_path.read_bytes())
                    self.assertEqual(restored.prefill.graph.initializer[0].raw_data, replacement)

                    fd_path = pathlib.Path(temp_dir, "pipeline-fd.onnx")
                    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
                    if hasattr(os, "O_BINARY"):
                        flags |= os.O_BINARY
                    fd = os.open(fd_path, flags, 0o666)
                    try:
                        pipeline.SerializeToFileDescriptor(fd, options)
                    finally:
                        os.close(fd)
                    restored = self._parse(fd_path.read_bytes())
                    self.assertEqual(restored.prefill.graph.initializer[0].raw_data, replacement)

                self.assertEqual(pipeline.prefill.graph.initializer[0].raw_data, b"abcd")


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

    def test_proto_bytesize_matches_serialized_length(self) -> None:
        model = _simple_model()
        self.assertEqual(model.ByteSize(), len(model.SerializeToString()))

        tensor = _simple_tensor()
        self.assertEqual(tensor.ByteSize(), len(tensor.SerializeToString()))

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

    def test_repeated_proto_field_survives_container_clear(self) -> None:
        # Regression test: a Python reference to an element of a repeated proto
        # field must keep the underlying C++ object alive even after the
        # container is cleared via __delitem__ / clear.
        import numpy as np

        graph = oh.make_graph(
            [
                oh.make_node("MatMul", ["X", "W"], ["XW"]),
                oh.make_node("Add", ["XW", "B"], ["Z"]),
                oh.make_node("Relu", ["Z"], ["Y"]),
            ],
            "shape_inference_demo",
            inputs=[oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, ["N", 4])],
            outputs=[oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, None)],
            initializer=[
                oh.make_tensor(
                    "W",
                    onnxl.TensorProto.FLOAT,
                    [4, 3],
                    np.zeros((4, 3), dtype=np.float32).flatten(),
                ),
                oh.make_tensor("B", onnxl.TensorProto.FLOAT, [3], np.zeros(3, dtype=np.float32)),
            ],
        )
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)], ir_version=8)

        # Take references via __getitem__, iteration and add() then clear the
        # underlying container; each held reference must remain usable.
        node_via_getitem = model.graph.node[0]
        iter_nodes = list(model.graph.node)
        new_node = model.graph.node.add()
        new_node.op_type = "Identity"

        del model.graph.node[:]

        self.assertEqual(node_via_getitem.op_type, "MatMul")
        self.assertEqual([n.op_type for n in iter_nodes], ["MatMul", "Add", "Relu"])
        self.assertEqual(new_node.op_type, "Identity")

    def test_persistence(self):
        graph = oh.make_graph(
            [
                oh.make_node("MatMul", ["X", "W"], ["XW"]),
                oh.make_node("Add", ["XW", "B"], ["Z"]),
                oh.make_node("Relu", ["Z"], ["Y"]),
            ],
            "shape_inference_demo",
            inputs=[oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, ["N", 4])],
            outputs=[oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, None)],
            initializer=[
                oh.make_tensor(
                    "W",
                    onnxl.TensorProto.FLOAT,
                    [4, 3],
                    np.zeros((4, 3), dtype=np.float32).flatten(),
                ),
                oh.make_tensor("B", onnxl.TensorProto.FLOAT, [3], np.zeros(3, dtype=np.float32)),
            ],
        )
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)], ir_version=8)
        node1 = model.graph.node[0]
        s1 = node1.SerializeToString()
        del model.graph.node[:]
        s2 = node1.SerializeToString()
        self.assertEqual(s1, s2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
