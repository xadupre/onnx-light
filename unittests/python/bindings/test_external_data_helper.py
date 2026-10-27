import ctypes
import gc
import mmap
import os
import tempfile
import unittest
import warnings
import weakref

import numpy as np

from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx_core.graph_builder import GraphBuilder
import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
from onnx_light.onnx_lib.external_data_helper import (
    convert_model_to_external_data,
    load_external_data_for_model,
    uses_external_data,
)
from onnx_light.onnx.external_data_helper import ExternalDataInfo, load_external_data_for_tensor
from onnx_light.onnx.numpy_helper import to_array


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


def _make_borrowed_model(values: np.ndarray, source_kind: str) -> onnxl.ModelProto:
    if source_kind == "numpy":
        builder = GraphBuilder()
        builder.init(values, name="W", copy=False)
        model = builder.to_onnx()
    else:
        model = onnxl.ModelProto()
        model.graph.initializer.append(onnxl.TensorProto.from_dlpack(values, name="W"))
    model.graph.node.append(oh.make_node("Identity", ["W"], ["Y"]))
    model.graph.output.append(
        oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, list(values.shape))
    )
    return model


def _iter_tensor_payloads(proto):
    """Yields tensors through bound properties independently of the native iterator."""
    if isinstance(proto, onnxl.TensorProto):
        yield proto
        return
    for name, descriptor in vars(type(proto)).items():
        if not isinstance(descriptor, property):
            continue
        presence = getattr(proto, "has_" + name, None)
        if presence is not None and not presence():
            continue
        value = getattr(proto, name)
        if hasattr(value, "SerializeToString"):
            yield from _iter_tensor_payloads(value)
        elif hasattr(value, "values"):
            for item in value.values():
                yield from _iter_tensor_payloads(item)
        elif not isinstance(value, (str, bytes)) and hasattr(value, "__iter__"):
            for item in value:
                if hasattr(item, "SerializeToString"):
                    yield from _iter_tensor_payloads(item)


class TestExternalDataHelper(ExtTestCase):
    def test_borrowed_external_save_shares_payload_and_preserves_source(self):
        for source_kind in ("numpy", "dlpack"):
            for num_threads in (1, 2):
                with (
                    self.subTest(source_kind=source_kind, num_threads=num_threads),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    values = np.arange(65536, dtype=np.float32)
                    address = values.ctypes.data
                    source = weakref.ref(values)
                    model = _make_borrowed_model(values, source_kind)
                    tensor = model.graph.initializer[0]
                    tensor.external_data.add(key="location", value="original.bin")
                    tensor.external_data.add(key="offset", value="0")
                    tensor.external_data.add(key="length", value=str(values.nbytes))
                    tensor.external_data.add(key="custom", value="preserved")
                    tensor.data_location = onnxl.TensorProto.EXTERNAL
                    before = model.SerializeToString()
                    del values
                    gc.collect()
                    self.assertIsNotNone(source())
                    calls = []

                    def callback(node, graph):
                        calls.append(node.op_type)
                        self.assertIsNot(graph, model.graph)
                        view = np.from_dlpack(graph.initializer[0])
                        self.assertEqual(view.ctypes.data, address)
                        node.doc_string = "saved only"
                        graph.doc_string = "working copy only"

                    options = onnxl.SerializeOptions()
                    options.num_threads = num_threads
                    options.min_parallel_block_size = 1
                    options.raw_data_threshold = 1024
                    options.node_callback = callback
                    path = os.path.join(tmp, "model.onnx")
                    for _ in range(2):
                        model.SerializeToFile(path, options, os.path.join(tmp, "weights.bin"))
                        self.assertEqual(model.SerializeToString(), before)
                        self.assertEqual(np.from_dlpack(tensor).ctypes.data, address)
                    self.assertEqual(calls, ["Identity", "Identity"])
                    loaded = onnxl.load(path, load_external_data=True)
                    np.testing.assert_array_equal(
                        to_array(loaded.graph.initializer[0]), np.arange(65536, dtype=np.float32)
                    )
                    self.assertEqual(loaded.graph.node[0].doc_string, "saved only")
                    self.assertEqual(loaded.graph.doc_string, "working copy only")
                    model = None
                    tensor = None
                    gc.collect()
                    self.assertIsNone(source())

    def test_owned_native_dlpack_external_save_shares_payload(self):
        values = np.arange(128, dtype=np.float32)
        source = _make_model(values)
        address = np.from_dlpack(source.graph.initializer[0]).ctypes.data
        model = _make_model(values)
        model.graph.initializer[0].CopyFrom(
            onnxl.TensorProto.from_dlpack(source.graph.initializer[0], name="W")
        )

        def callback(node, graph):
            self.assertEqual(np.from_dlpack(graph.initializer[0]).ctypes.data, address)

        options = onnxl.SerializeOptions()
        options.raw_data_threshold = 0
        options.node_callback = callback
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.onnx")
            model.SerializeToFile(path, options, os.path.join(tmp, "weights.bin"))
            loaded = onnxl.load(path, load_external_data=True)
            np.testing.assert_array_equal(to_array(loaded.graph.initializer[0]), values)

    def test_external_save_retains_borrowed_owner_after_source_clear(self):
        for source_kind in ("numpy", "dlpack"):
            with self.subTest(source_kind=source_kind), tempfile.TemporaryDirectory() as tmp:
                values = np.arange(65536, dtype=np.float32)
                address = values.ctypes.data
                source = weakref.ref(values)
                model = _make_borrowed_model(values, source_kind)
                del values
                calls = []

                def callback(node, graph):
                    self.assertEqual(np.from_dlpack(graph.initializer[0]).ctypes.data, address)
                    model.graph.initializer.clear()
                    gc.collect()
                    self.assertIsNotNone(source())
                    calls.append(node.op_type)

                options = onnxl.SerializeOptions()
                options.num_threads = 2
                options.min_parallel_block_size = 1
                options.raw_data_threshold = 0
                options.node_callback = callback
                path = os.path.join(tmp, "model.onnx")
                model.SerializeToFile(path, options, os.path.join(tmp, "weights.bin"))
                gc.collect()
                self.assertIsNone(source())
                self.assertEqual(calls, ["Identity"])
                loaded = onnxl.load(path, load_external_data=True)
                np.testing.assert_array_equal(
                    to_array(loaded.graph.initializer[0]), np.arange(65536, dtype=np.float32)
                )

    def test_borrowed_external_save_raw_callback_and_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as tmp:
                values = np.arange(64, dtype=np.float32)
                model = _make_borrowed_model(values, "dlpack")
                before = model.SerializeToString()
                calls = []

                def callback(tensor, graph, buffer, size_only):
                    self.assertEqual(np.from_dlpack(tensor).ctypes.data, values.ctypes.data)
                    calls.append(size_only)
                    graph.doc_string = "working copy only"
                    if not size_only:
                        if fail:
                            raise ValueError("callback failed")
                        buffer[:] = np.zeros_like(values).view(np.uint8)
                        tensor.doc_string = "rewritten"
                    return values.nbytes

                path = os.path.join(tmp, "model.onnx")
                kwargs = dict(
                    save_as_external_data=True,
                    location="weights.bin",
                    size_threshold=0,
                    raw_data_callback=callback,
                )
                if fail:
                    with self.assertRaisesRegex(ValueError, "callback failed"):
                        onnxl.save_model(model, path, **kwargs)
                else:
                    onnxl.save_model(model, path, **kwargs)
                    loaded = onnxl.load(path, load_external_data=True)
                    np.testing.assert_array_equal(
                        to_array(loaded.graph.initializer[0]), np.zeros_like(values)
                    )
                    self.assertEqual(loaded.graph.initializer[0].doc_string, "rewritten")
                self.assertEqual(calls, [True, False])
                self.assertEqual(model.SerializeToString(), before)
                self.assertEqual(
                    np.from_dlpack(model.graph.initializer[0]).ctypes.data, values.ctypes.data
                )

    def test_borrowed_external_save_size_limit_preserves_source(self):
        model = _make_borrowed_model(np.arange(64, dtype=np.float32), "numpy")
        before = model.SerializeToString()
        options = onnxl.SerializeOptions()
        options.raw_data_threshold = 0
        options.max_serialized_size_bytes = 1
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.onnx")
            weights = os.path.join(tmp, "weights.bin")
            with self.assertRaisesRegex(RuntimeError, "max_serialized_size_bytes"):
                model.SerializeToFile(path, options, weights)
            self.assertEqual(model.SerializeToString(), before)
            options.max_serialized_size_bytes = 0
            model.SerializeToFile(path, options, weights)
            np.testing.assert_array_equal(
                to_array(onnxl.load(path, load_external_data=True).graph.initializer[0]),
                np.arange(64, dtype=np.float32),
            )
            self.assertEqual(model.SerializeToString(), before)

    def test_mapped_external_save_failure_detaches_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.onnx")
            values = np.arange(128, dtype=np.float32)
            onnxl.save_model(_make_model(values), path)
            model = onnxl.load(path, file_load_mode="MMAP", no_copy=True)
            before = model.SerializeToString()
            borrowed = np.from_dlpack(model.graph.initializer[0])
            options = onnxl.SerializeOptions()
            options.raw_data_threshold = 0
            options.max_serialized_size_bytes = 1
            with self.assertRaisesRegex(RuntimeError, "max_serialized_size_bytes"):
                model.SerializeToFile(path, options, os.path.join(tmp, "weights.bin"))
            snapshot = np.from_dlpack(model.graph.initializer[0])
            self.assertNotEqual(snapshot.ctypes.data, borrowed.ctypes.data)
            np.testing.assert_array_equal(snapshot, values)
            self.assertEqual(model.SerializeToString(), before)

    def test_borrowed_external_save_validates_tensor_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = os.path.join(tmp, "model")
            os.mkdir(model_dir)
            outside = os.path.join(tmp, "outside.bin")
            with open(outside, "wb") as f:
                f.write(b"unchanged")
            model = _make_borrowed_model(np.arange(64, dtype=np.float32), "dlpack")
            tensor = model.graph.initializer[0]
            tensor.data_location = onnxl.TensorProto.EXTERNAL
            tensor.external_data.add(key="location", value="../outside.bin")
            tensor.external_data.add(key="offset", value="0")
            tensor.external_data.add(key="length", value="256")
            before = model.SerializeToString()
            with self.assertRaises(RuntimeError):
                model.SerializeToFile(
                    os.path.join(model_dir, "model.onnx"),
                    onnxl.SerializeOptions(),
                    os.path.join(model_dir, "weights.bin"),
                )
            self.assertEqual(model.SerializeToString(), before)
            with open(outside, "rb") as f:
                self.assertEqual(f.read(), b"unchanged")

    def test_mapped_external_resave_same_location(self):
        for max_file_size in (0, 256):
            with self.subTest(max_file_size=max_file_size), tempfile.TemporaryDirectory() as tmp:
                values = np.arange(128, dtype=np.float32)
                model = _make_model(values)
                extra = oh.make_tensor(
                    "other", onnxl.TensorProto.FLOAT, [128], (values + 1).tobytes(), raw=True
                )
                model.graph.initializer.append(extra)
                path = os.path.join(tmp, "model.onnx")
                weights = os.path.join(tmp, "weights.bin")
                onnxl.save_model(model, path, location=weights, size_threshold=0)
                borrowed = onnxl.load(path, load_external_data=True, no_copy=True)
                before = borrowed.SerializeToString()
                options = onnxl.SerializeOptions()
                options.alignment = 4096
                options.max_external_file_size = max_file_size
                borrowed.SerializeToFile(path, options, weights)
                self.assertEqual(borrowed.SerializeToString(), before)
                loaded = onnxl.load(path, load_external_data=True)
                for tensor, expected in zip(loaded.graph.initializer, (values, values + 1)):
                    np.testing.assert_array_equal(to_array(tensor), expected)

    def test_mapped_inline_resave_as_external_same_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = np.arange(128, dtype=np.float32)
            path = os.path.join(tmp, "model.onnx")
            onnxl.save_model(_make_model(values), path)
            model = onnxl.load(path, file_load_mode="MMAP", no_copy=True)
            before = model.SerializeToString()
            onnxl.save_model(model, path, save_as_external_data=True, size_threshold=0)
            self.assertEqual(model.SerializeToString(), before)
            loaded = onnxl.load(path, load_external_data=True)
            np.testing.assert_array_equal(to_array(loaded.graph.initializer[0]), values)

    def test_parse_cleanup_retaining_mapping_resaves_to_same_path(self):
        for external in (False, True):
            for reimport in (False, True):
                for num_threads in (1, 2):
                    with (
                        self.subTest(
                            external=external, reimport=reimport, num_threads=num_threads
                        ),
                        tempfile.TemporaryDirectory() as tmp,
                    ):
                        values = np.arange(2048, dtype=np.float32)
                        path = os.path.join(tmp, "model.onnx")
                        weights = os.path.join(tmp, "weights.bin")
                        onnxl.save_model(
                            _make_model(values),
                            path,
                            save_as_external_data=external,
                            location="weights.bin" if external else None,
                            size_threshold=0,
                        )
                        released = []

                        def callback(tensor, graph):
                            retained = onnxl.TensorProto()
                            retained.CopyFrom(tensor)
                            return lambda: released.append(retained.name)

                        parse_options = onnxl.ParseOptions()
                        parse_options.no_copy = True
                        parse_options.file_load_mode = onnxl.FileLoadMode.MMAP
                        parse_options.raw_data_callback = callback
                        model = onnxl.ModelProto()
                        model.ParseFromFile(
                            path, parse_options, external_data_file=weights if external else ""
                        )
                        if reimport:
                            tensor = onnxl.TensorProto.from_dlpack(
                                model.graph.initializer[0], name="W"
                            )
                            reimported = _make_model(values)
                            reimported.graph.initializer[0].CopyFrom(tensor)
                            del model, tensor
                            model = reimported
                        self.assertEqual(released, [])
                        before = model.SerializeToString()
                        options = onnxl.SerializeOptions()
                        options.raw_data_threshold = 0
                        options.num_threads = num_threads
                        options.min_parallel_block_size = 1
                        model.SerializeToFile(path, options, weights)
                        self.assertEqual(model.SerializeToString(), before)
                        self.assertEqual(released, ["W"])
                        loaded = onnxl.load(path, load_external_data=True)
                        np.testing.assert_array_equal(
                            to_array(loaded.graph.initializer[0]), values
                        )

    def test_mapped_mixed_initializers_resave_same_model_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = np.arange(128, dtype=np.float32)
            model = _make_model(values)
            encoded = onnxl.EncodedValueProto(name="encoded", raw_data=values.tobytes())
            model.graph.extended_initializer.append(
                onnxl.ExtendedInitializerProto(encoded_value=encoded)
            )
            path = os.path.join(tmp, "model.onnx")
            weights = os.path.join(tmp, "weights.bin")
            onnxl.save_model(model, path)
            mapped = onnxl.load(path, file_load_mode="MMAP", no_copy=True)
            options = onnxl.SerializeOptions()
            options.raw_data_threshold = 0
            mapped.SerializeToFile(path, options, weights)
            loaded = onnxl.load(path, load_external_data=True)
            np.testing.assert_array_equal(to_array(loaded.graph.initializer[0]), values)
            self.assertEqual(
                loaded.graph.extended_initializer[0].encoded_value.raw_data, values.tobytes()
            )

    def test_native_mapped_dlpack_reimports_resave_to_same_path(self):
        for external in (False, True):
            for depth in (1, 3):
                with (
                    self.subTest(external=external, depth=depth),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    values = np.arange(128, dtype=np.float32)
                    path = os.path.join(tmp, "model.onnx")
                    weights = os.path.join(tmp, "weights.bin")
                    onnxl.save_model(
                        _make_model(values),
                        path,
                        save_as_external_data=external,
                        location="weights.bin" if external else None,
                        size_threshold=0,
                    )
                    mapped = onnxl.load(
                        path, load_external_data=external, file_load_mode="MMAP", no_copy=True
                    )
                    tensor = mapped.graph.initializer[0]
                    for _ in range(depth):
                        tensor = onnxl.TensorProto.from_dlpack(tensor, name="W")
                    model = _make_model(values)
                    model.graph.initializer[0].CopyFrom(tensor)
                    del mapped, tensor
                    gc.collect()
                    before = model.SerializeToString()
                    options = onnxl.SerializeOptions()
                    options.raw_data_threshold = 0
                    options.num_threads = 2
                    options.min_parallel_block_size = 1
                    model.SerializeToFile(path, options, weights)
                    self.assertEqual(model.SerializeToString(), before)
                    loaded = onnxl.load(path, load_external_data=True)
                    np.testing.assert_array_equal(to_array(loaded.graph.initializer[0]), values)

    def test_mapped_layout_tensors_resave_same_model_path(self):
        for layout in ("affine", "inline", "catalogue", "nested", "array", "codec"):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as tmp:
                values = np.arange(2048, dtype=np.float32)
                model = _make_model(values)
                encoded = onnxl.EncodedValueProto(name="encoded", raw_data=bytes(range(128)))
                if layout == "affine":
                    encoded.affine.storage_type = onnxl.TensorProto.UINT8
                    encoded.affine.scale.CopyFrom(
                        oh.make_tensor(
                            "", onnxl.TensorProto.FLOAT, [], np.float32(0.25).tobytes(), raw=True
                        )
                    )
                    encoded.affine.zero_point.CopyFrom(
                        oh.make_tensor("", onnxl.TensorProto.UINT8, [], b"\x11", raw=True)
                    )
                    encoded.logical_type.CopyFrom(
                        oh.make_tensor_type_proto(onnxl.TensorProto.FLOAT, [128])
                    )
                else:
                    declaration = onnxl.StructTypeProto(
                        structure=onnxl.StructTypeProto.Structure(
                            field=[
                                onnxl.StructTypeProto.Structure.Field(
                                    name="code",
                                    type=oh.make_tensor_type_proto(onnxl.TensorProto.UINT8, []),
                                ),
                                onnxl.StructTypeProto.Structure.Field(
                                    name="constant",
                                    constant=oh.make_tensor(
                                        "",
                                        onnxl.TensorProto.FLOAT,
                                        [128],
                                        values[:128].tobytes(),
                                        raw=True,
                                    ),
                                ),
                            ]
                        )
                    )
                    if layout == "codec":
                        declaration.decoder.CopyFrom(
                            onnxl.FunctionProto(
                                name="Decode",
                                domain="test",
                                input=["X"],
                                output=["Y"],
                                node=[oh.make_node("Identity", ["X"], ["Y"])],
                                opset_import=[oh.make_opsetid("", 17)],
                                attribute_proto=[
                                    oh.make_attribute("body", _make_model(values[:128]).graph)
                                ],
                            )
                        )
                    if layout == "array":
                        encoded.struct_type.array.element_type.CopyFrom(
                            onnxl.TypeProto(struct_type=declaration)
                        )
                        encoded.struct_type.array.dimension = 128
                    elif layout == "catalogue":
                        declaration.type_id = 1
                        model.struct_types.append(declaration)
                        encoded.struct_type.type_ref = 1
                    elif layout == "nested":
                        element = onnxl.TypeProto(struct_type=declaration)
                        nested = onnxl.TypeProto(
                            optional_type=onnxl.TypeProto.Optional(
                                elem_type=onnxl.TypeProto(
                                    map_type=onnxl.TypeProto.Map(
                                        key_type=onnxl.TensorProto.INT64,
                                        value_type=onnxl.TypeProto(
                                            sequence_type=onnxl.TypeProto.Sequence(
                                                elem_type=element
                                            )
                                        ),
                                    )
                                )
                            )
                        )
                        model.graph.input.append(onnxl.ValueInfoProto(name="record", type=nested))
                    else:
                        encoded.struct_type.CopyFrom(declaration)
                model.graph.extended_initializer.append(
                    onnxl.ExtendedInitializerProto(encoded_value=encoded)
                )
                path = os.path.join(tmp, "model.onnx")
                onnxl.save_model(model, path)
                mapped = onnxl.load(path, file_load_mode="MMAP", no_copy=True)
                before = mapped.SerializeToString()
                tensors = [tensor for tensor in _iter_tensor_payloads(mapped) if tensor.raw_data]
                self.assertGreaterEqual(len(tensors), 2)
                borrowed_views = []
                for tensor in tensors:
                    # Byte-wise borrowed views also cover unaligned inline payloads.
                    byte_tensor = onnxl.TensorProto()
                    byte_tensor.CopyFrom(tensor)
                    byte_tensor.data_type = onnxl.TensorProto.UINT8
                    byte_tensor.dims.clear()
                    byte_tensor.dims.append(len(tensor.raw_data))
                    borrowed_views.append(np.from_dlpack(byte_tensor))
                mapped_addresses = {view.ctypes.data for view in borrowed_views}
                options = onnxl.SerializeOptions()
                options.raw_data_threshold = 1024
                options.num_threads = 2
                options.min_parallel_block_size = 1
                mapped.SerializeToFile(path, options, os.path.join(tmp, "weights.bin"))
                for tensor in _iter_tensor_payloads(mapped):
                    if tensor.raw_data:
                        self.assertNotIn(np.from_dlpack(tensor).ctypes.data, mapped_addresses)
                self.assertEqual(mapped.SerializeToString(), before)
                loaded = onnxl.load(path, load_external_data=True)
                loaded.graph.initializer[0].CopyFrom(model.graph.initializer[0])
                self.assertEqual(loaded.SerializeToString(), model.SerializeToString())

    def test_borrowed_mapped_views_resave_to_same_path(self):
        for source_kind in ("numpy", "dlpack"):
            for view_kind in ("direct", "array", "memoryview", "slice", "mmap"):
                with (
                    self.subTest(source_kind=source_kind, view_kind=view_kind),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    values = np.arange(128, dtype=np.float32)
                    weights = os.path.join(tmp, "weights.bin")
                    with open(weights, "wb") as stream:
                        stream.write(values.tobytes())
                    mapping = None
                    if view_kind == "mmap":
                        with open(weights, "r+b") as stream:
                            mapping = mmap.mmap(stream.fileno(), 0)
                        mapped_values = np.frombuffer(mapping, dtype=np.float32)
                    else:
                        mapped_values = np.memmap(
                            weights, dtype=np.float32, mode="r+", shape=values.shape
                        )
                        if view_kind == "array":
                            mapped_values = np.asarray(mapped_values)
                        elif view_kind == "memoryview":
                            mapped_values = np.asarray(memoryview(mapped_values))
                        elif view_kind == "slice":
                            mapped_values = np.asarray(mapped_values)[16:112]
                            values = values[16:112]
                    model = _make_borrowed_model(mapped_values, source_kind)
                    before = model.SerializeToString()
                    path = os.path.join(tmp, "model.onnx")
                    options = onnxl.SerializeOptions()
                    options.raw_data_threshold = 0
                    options.num_threads = 2
                    options.min_parallel_block_size = 1
                    model.SerializeToFile(path, options, weights)
                    self.assertEqual(model.SerializeToString(), before)
                    loaded = onnxl.load(path, load_external_data=True)
                    np.testing.assert_array_equal(to_array(loaded.graph.initializer[0]), values)
                    del model, mapped_values
                    gc.collect()
                    if mapping is not None:
                        mapping.close()

    def test_unrecognized_mapped_exporters_resave_to_same_path(self):
        class DLPackAdapter:
            def __init__(self, tensor):
                self.tensor = tensor

            def __dlpack_device__(self):
                return self.tensor.__dlpack_device__()

            def __dlpack__(self, **kwargs):
                return self.tensor.__dlpack__(**kwargs)

        for exporter_kind in ("buffer", "dlpack"):
            with self.subTest(exporter_kind=exporter_kind), tempfile.TemporaryDirectory() as tmp:
                values = np.arange(128, dtype=np.float32)
                weights = os.path.join(tmp, "weights.bin")
                with open(weights, "wb") as stream:
                    stream.write(values.tobytes())
                if exporter_kind == "buffer":
                    with open(weights, "r+b") as stream:
                        mapping = mmap.mmap(stream.fileno(), 0)
                    exporter = (ctypes.c_float * values.size).from_buffer(mapping)
                    tensor = onnxl.TensorProto(
                        name="W", data_type=onnxl.TensorProto.FLOAT, dims=[values.size]
                    )
                    tensor._set_raw_data_from_buffer(exporter)
                else:
                    exporter = DLPackAdapter(
                        np.memmap(weights, dtype=np.float32, mode="r+", shape=values.shape)
                    )
                    tensor = onnxl.TensorProto.from_dlpack(exporter, name="W")
                model = _make_model(values)
                model.graph.initializer[0].CopyFrom(tensor)
                before = model.SerializeToString()
                path = os.path.join(tmp, "model.onnx")
                options = onnxl.SerializeOptions()
                options.raw_data_threshold = 0
                options.num_threads = 2
                options.min_parallel_block_size = 1
                model.SerializeToFile(path, options, weights)
                self.assertEqual(model.SerializeToString(), before)
                loaded = onnxl.load(path, load_external_data=True)
                np.testing.assert_array_equal(to_array(loaded.graph.initializer[0]), values)
                del model, tensor, exporter
                gc.collect()
                if exporter_kind == "buffer":
                    mapping.close()

    def test_borrowed_buffer_rejects_cyclic_owner_chain(self):
        class CyclicArray(np.ndarray):
            @property
            def base(self):
                return self

            def __dlpack__(self, **kwargs):
                self.exported = True
                return super().__dlpack__(**kwargs)

        values = np.arange(8, dtype=np.float32).view(CyclicArray)
        values.exported = False
        source = weakref.ref(values)
        tensor = onnxl.TensorProto(raw_data=b"unchanged")
        with self.assertRaisesRegex(BufferError, "cyclic ownership chain"):
            tensor._set_raw_data_from_buffer(values)
        self.assertEqual(tensor.raw_data, b"unchanged")
        with self.assertRaisesRegex(BufferError, "cyclic ownership chain"):
            onnxl.TensorProto.from_dlpack(values)
        self.assertFalse(values.exported)
        del values
        gc.collect()
        self.assertIsNone(source())

    @unittest.skipIf(os.name == "nt", "Symlinks require elevated privileges on Windows")
    def test_auto_load_linked_models(self):
        values = np.arange(64, dtype=np.float32)
        for link_kind in ("symlink", "link"):
            for external in (False, True):
                with (
                    self.subTest(link_kind=link_kind, external=external),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    model_dir = os.path.join(tmp, "model")
                    os.mkdir(model_dir)
                    model_path = os.path.join(model_dir, "linked.onnx")
                    onnxl.save_model(
                        _make_model(values),
                        model_path,
                        save_as_external_data=external,
                        location="weights.bin" if external else None,
                        size_threshold=0,
                    )
                    original = os.path.join(tmp, "original.onnx")
                    os.rename(model_path, original)
                    getattr(os, link_kind)(original, model_path)
                    for options in (
                        {},
                        {"num_threads": 1},
                        {"no_copy": True},
                        {"file_load_mode": "MMAP"},
                        {"num_threads": 2, "min_block_size": 0},
                    ):
                        for path in (model_path, os.path.relpath(model_path)):
                            with self.subTest(options=options, path=path):
                                loaded = onnxl.load(path, load_external_data=True, **options)
                                np.testing.assert_array_equal(
                                    to_array(loaded.graph.initializer[0]), values
                                )

    @unittest.skipIf(os.name == "nt", "Symlinks require elevated privileges on Windows")
    def test_auto_load_validates_model_selected_as_weights_source(self):
        for link_kind, message in (
            ("symlink", "symbolic link"),
            ("link", "multiple hard links"),
            ("rename", None),
        ):
            with self.subTest(link_kind=link_kind), tempfile.TemporaryDirectory() as tmp:
                model = _make_model(np.arange(4, dtype=np.float32))
                tensor = model.graph.initializer[0]
                tensor.raw_data = b""
                tensor.data_location = onnxl.TensorProto.EXTERNAL
                for key, value in (
                    ("location", "linked.onnx"),
                    ("offset", "0"),
                    ("length", "16"),
                ):
                    entry = tensor.external_data.add()
                    entry.key = key
                    entry.value = value
                original = os.path.join(tmp, "original.onnx")
                serialized = model.SerializeToString()
                with open(original, "wb") as f:
                    f.write(serialized)
                model_path = os.path.join(tmp, "linked.onnx")
                getattr(os, link_kind)(original, model_path)
                for options in (
                    {},
                    {"num_threads": 1},
                    {"no_copy": True},
                    {"file_load_mode": "MMAP"},
                    {"num_threads": 2, "min_block_size": 0},
                ):
                    with self.subTest(options=options):
                        if message:
                            with self.assertRaisesRegex(RuntimeError, message):
                                onnxl.load(model_path, load_external_data=True, **options)
                        else:
                            loaded = onnxl.load(model_path, load_external_data=True, **options)
                            self.assertEqual(
                                bytes(loaded.graph.initializer[0].raw_data), serialized[:16]
                            )

    @unittest.skipIf(os.name == "nt", "Symlinks require elevated privileges on Windows")
    def test_auto_load_rejects_unsafe_weights_sources(self):
        values = np.arange(64, dtype=np.float32)
        for source_kind, message in (
            ("symlink", "symbolic link"),
            ("link", "multiple hard links"),
            ("parent_symlink", "outside the model directory"),
        ):
            with self.subTest(source_kind=source_kind), tempfile.TemporaryDirectory() as tmp:
                model_dir = os.path.join(tmp, "model")
                outside_dir = os.path.join(tmp, "outside")
                os.mkdir(model_dir)
                os.mkdir(outside_dir)
                location = "sub/weights.bin" if source_kind == "parent_symlink" else "weights.bin"
                if source_kind == "parent_symlink":
                    os.mkdir(os.path.join(model_dir, "sub"))
                model_path = os.path.join(model_dir, "model.onnx")
                onnxl.save_model(
                    _make_model(values),
                    model_path,
                    save_as_external_data=True,
                    location=location,
                    size_threshold=0,
                )
                source = os.path.join(model_dir, location)
                outside = os.path.join(outside_dir, "weights.bin")
                os.rename(source, outside)
                if source_kind == "parent_symlink":
                    os.rmdir(os.path.dirname(source))
                    os.symlink(outside_dir, os.path.dirname(source))
                else:
                    getattr(os, source_kind)(outside, source)
                for explicit in (False, True):
                    for options in (
                        {},
                        {"num_threads": 1},
                        {"no_copy": True},
                        {"file_load_mode": "MMAP"},
                        {"num_threads": 2, "min_block_size": 0},
                    ):
                        with (
                            self.subTest(explicit=explicit, options=options),
                            self.assertRaisesRegex(RuntimeError, message),
                        ):
                            onnxl.load(
                                model_path,
                                load_external_data=True,
                                location=source if explicit else None,
                                **options,
                            )

    def test_to_array_does_not_mutate_external_tensor(self):
        values = np.arange(64, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            onnxl.save_model(
                _make_model(values), model_path, save_as_external_data=True, size_threshold=0
            )
            model = onnxl.load(model_path, load_external_data=False)
            tensor = model.graph.initializer[0]
            self.assertEqual(tensor.data_location, onnxl.TensorProto.EXTERNAL)
            self.assertFalse(tensor.HasField("raw_data"))
            before = tensor.SerializeToString()

            for _ in range(2):
                np.testing.assert_array_equal(to_array(tensor, tmp), values)
                self.assertEqual(tensor.SerializeToString(), before)

            resaved_path = os.path.join(tmp, "resaved.onnx")
            onnxl.save_model(model, resaved_path)
            reloaded = onnxl.load(resaved_path, load_external_data=True)
            np.testing.assert_array_equal(to_array(reloaded.graph.initializer[0]), values)

    def test_external_tensor_read_and_load(self):
        for size in (0, 4):
            values = np.arange(size, dtype=np.float32)
            for length in (None, values.nbytes):
                for raw_data in (b"", b"stale"):
                    with (
                        self.subTest(size=size, length=length, raw_data=raw_data),
                        tempfile.TemporaryDirectory() as tmp,
                    ):
                        with open(os.path.join(tmp, "data.bin"), "wb") as f:
                            f.write(b"PAD" + values.tobytes())
                        tensor = onnxl.TensorProto()
                        tensor.name = "t"
                        tensor.data_type = onnxl.TensorProto.FLOAT
                        tensor.dims.extend([size])
                        tensor.data_location = onnxl.TensorProto.EXTERNAL
                        tensor.raw_data = raw_data
                        entries = [("location", "data.bin"), ("offset", "3")]
                        if length is not None:
                            entries.append(("length", str(length)))
                        for key, value in entries:
                            entry = tensor.external_data.add()
                            entry.key = key
                            entry.value = value
                        before = tensor.SerializeToString()

                        np.testing.assert_array_equal(to_array(tensor, tmp), values)
                        self.assertEqual(tensor.SerializeToString(), before)

                        load_external_data_for_tensor(tensor, tmp)
                        self.assertEqual(bytes(tensor.raw_data), values.tobytes())
                        self.assertEqual(tensor.data_location, onnxl.TensorProto.DEFAULT)
                        self.assertEqual(len(tensor.external_data), 0)
                        np.testing.assert_array_equal(to_array(tensor), values)

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
    # Security tests: GHSA-3jf9-582g-jjmq (CVE-2026-34445)
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

            before = tensor.SerializeToString()
            for read in (to_array, _load_external_data_for_tensor):
                with self.assertRaises(ValueError):
                    read(tensor, tmp)
                self.assertEqual(tensor.SerializeToString(), before)

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
            self.assertEqual(tensor.data_location, onnxl.TensorProto.DEFAULT)
            self.assertEqual(len(tensor.external_data), 0)
            got = np.frombuffer(tensor.raw_data, dtype=np.float32)
            np.testing.assert_array_equal(got, values[4:])


if __name__ == "__main__":
    unittest.main(verbosity=2)
