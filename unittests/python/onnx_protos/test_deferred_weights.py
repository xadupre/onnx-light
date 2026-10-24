import io
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh
from onnx_light.onnx import (
    ModelProto,
    ParseOptions,
    PipelineProto,
    SerializeFormat,
    SerializeOptions,
    load,
    save,
    save_tensor,
)


class TestDeferredWeights(unittest.TestCase):
    @staticmethod
    def make_model():
        """Returns a model with independently loadable raw tensors."""
        return oh.make_model(
            oh.make_graph(
                [],
                "weights",
                [],
                [],
                [
                    onh.from_array(np.arange(1024, dtype=np.float32), name="A"),
                    onh.from_array(np.array([7, 8], dtype=np.int64), name="B"),
                    onh.from_array(np.empty((0,), dtype=np.float32), name="empty"),
                ],
            )
        )

    def test_inline_and_external_cycles_copies_and_export(self):
        for external in (False, True):
            with self.subTest(external=external), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.onnx"
                save(self.make_model(), path, save_as_external_data=external, size_threshold=0)
                model = load(path, defer_weights=True)
                self.assertCountEqual(model.weight_names(), ["A", "B", "empty"])
                self.assertTrue(model.has_deferred_weights())
                self.assertFalse(model.is_weight_loaded("A"))
                with self.assertRaisesRegex(ValueError, "is not loaded"):
                    onh.to_array(model.graph.initializer[0])
                self.assertFalse(model.unload_weight("A"))
                self.assertTrue(model.load_weight("A"))
                self.assertFalse(model.load_weight("A"))
                self.assertFalse(model.is_weight_loaded("B"))
                np.testing.assert_array_equal(
                    onh.to_array(model.graph.initializer[0]), np.arange(1024, dtype=np.float32)
                )
                copy = ModelProto()
                copy.CopyFrom(model)
                self.assertTrue(model.unload_weight("A"))
                self.assertEqual(len(model.graph.initializer[0].raw_data), 0)
                self.assertTrue(copy.is_weight_loaded("A"))
                self.assertTrue(copy.load_weight("B"))
                self.assertFalse(model.is_weight_loaded("B"))
                with self.assertRaisesRegex(RuntimeError, "Unknown weight"):
                    model.load_weight("missing")
                with self.assertRaisesRegex(RuntimeError, "unloaded deferred"):
                    model.SerializeToString()
                self.assertEqual(model.load_weights(), 3)
                self.assertTrue(model.is_weight_loaded("empty"))
                self.assertEqual(model.load_weights(), 0)
                np.testing.assert_array_equal(
                    onh.to_array(model.graph.initializer[0]), np.arange(1024, dtype=np.float32)
                )
                backing = Path(str(path) + ".data") if external else path
                before = backing.read_bytes()
                with self.assertRaisesRegex(RuntimeError, "backing source"):
                    save(model, backing)
                self.assertEqual(backing.read_bytes(), before)
                output = Path(directory) / "export.onnx"
                save(model, output)
                eager = load(output)
                self.assertFalse(eager.has_deferred_weights())
                np.testing.assert_array_equal(
                    onh.to_array(eager.graph.initializer[1]), np.array([7, 8], dtype=np.int64)
                )
                self.assertEqual(len(eager.graph.initializer[0].external_data), 0)
                with self.assertRaisesRegex(RuntimeError, "no recoverable"):
                    eager.unload_weight("A")
                self.assertEqual(model.unload_weights(), 3)
                self.assertEqual(model.unload_weights(), 0)
                self.assertTrue(model.load_weight("empty"))

    def test_parse_options_mmap_limits_and_unsupported_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            save(self.make_model(), path)
            model = load(path, defer_weights=True, file_load_mode="MMAP")
            self.assertTrue(model.load_weight("B"))
            options = ParseOptions()
            options.defer_weights = True
            options.max_tensor_size_bytes = 4
            with self.assertRaisesRegex(RuntimeError, "max_tensor_size_bytes"):
                ModelProto().ParseFromFile(str(path), options)
            for kwargs in (
                {"no_copy": True},
                {"load_external_data": True},
                {"tiny_external_data_threshold": 4},
                {"format": "textproto"},
            ):
                with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "requires"):
                    load(path, defer_weights=True, **kwargs)
            with self.assertRaisesRegex(ValueError, "requires"):
                load(path.read_bytes(), defer_weights=True)
            typed = self.make_model()
            typed.graph.initializer[0].raw_data = b""
            typed.graph.initializer[0].float_data.extend([1.0])
            save(typed, path)
            with self.assertRaisesRegex(RuntimeError, "typed tensor payload"):
                load(path, defer_weights=True)

    def test_save_guards_preserve_backing_files_and_reject_unsupported_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            save(self.make_model(), path)
            model = load(path, defer_weights=True)
            model.load_weights()
            before = path.read_bytes()
            for alias in (() if os.name == "nt" else ("symlink.onnx", "hardlink.onnx")):
                target = Path(directory) / alias
                if alias.startswith("symlink"):
                    target.symlink_to(path)
                else:
                    os.link(path, target)
                with self.assertRaisesRegex(RuntimeError, "backing source"):
                    model.SerializeToFile(str(target))
                target.unlink()
            with path.open("r+b") as output:
                with self.assertRaisesRegex(RuntimeError, "backing source"):
                    model.SerializeToFileDescriptor(output.fileno())
                with self.assertRaisesRegex(RuntimeError, "backing source"):
                    save(model, output)
            self.assertEqual(path.read_bytes(), before)
            output = io.BytesIO()
            save(model, output)
            self.assertEqual(len(load(output.getvalue()).graph.initializer), 3)
            for kwargs in ({"save_as_external_data": True}, {"format": "textproto"}):
                with (
                    self.subTest(kwargs=kwargs),
                    self.assertRaisesRegex(ValueError, "single-file"),
                ):
                    save(model, Path(directory) / "unsupported.onnx", **kwargs)
            options = SerializeOptions()
            options.format = SerializeFormat.ORT_FLATBUFFERS
            with self.assertRaises(RuntimeError):
                model.SerializeToString(options)

    def test_changed_and_missing_sources_do_not_partially_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            save(self.make_model(), path)
            model = load(path, defer_weights=True)
            path.write_bytes(path.read_bytes() + b"x")
            for name in ("A", "empty"):
                with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, "changed"):
                    model.load_weight(name)
                self.assertFalse(model.is_weight_loaded(name))
            path.unlink()
            with self.assertRaises(RuntimeError):
                model.load_weight("B")

    def test_standalone_tensor_exports_do_not_overwrite_sources(self):
        for external in (False, True):
            with self.subTest(external=external), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.onnx"
                save(self.make_model(), path, save_as_external_data=external, size_threshold=0)
                model = load(path, defer_weights=True)
                backing = Path(str(path) + ".data") if external else path
                before = backing.read_bytes()
                for index in (0, 2):
                    tensor = model.graph.initializer[index]
                    for loaded in (False, True):
                        with self.subTest(name=tensor.name, loaded=loaded):
                            if loaded:
                                model.load_weight(tensor.name)
                            with self.assertRaisesRegex(RuntimeError, "deferred tensor"):
                                tensor.SerializeToFile(str(backing))
                            for format in ("protobuf", "textproto"):
                                with self.assertRaisesRegex(ValueError, "deferred tensor"):
                                    save_tensor(tensor, backing, format=format)
                            with backing.open("r+b") as output:
                                with self.assertRaisesRegex(RuntimeError, "deferred tensor"):
                                    tensor.SerializeToFileDescriptor(output.fileno())
                                with self.assertRaisesRegex(RuntimeError, "deferred tensor"):
                                    tensor.SerializeToOstream(output)
                                with self.assertRaisesRegex(ValueError, "deferred tensor"):
                                    save_tensor(tensor, output)
                            self.assertEqual(backing.read_bytes(), before)

    def test_pipeline_deferred_input_is_rejected_without_changing_eager_io(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = PipelineProto(prefill=self.make_model())
            for extension in (".pipe-onnx", ".pipeline"):
                path = Path(directory) / f"model{extension}"
                save(pipeline, path)
                before = path.read_bytes()
                with self.assertRaisesRegex(ValueError, "ModelProto"):
                    load(path, defer_weights=True)
                options = ParseOptions()
                options.defer_weights = True
                with self.assertRaisesRegex(RuntimeError, "PipelineProto"):
                    PipelineProto().ParseFromFile(str(path), options)
                self.assertEqual(path.read_bytes(), before)
                self.assertIsInstance(load(path), PipelineProto)


if __name__ == "__main__":
    unittest.main()
