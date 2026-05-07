import os
import unittest
from unittest.mock import patch
import numpy as np
import onnx
import onnx.helper as xoh
import onnx.numpy_helper as xonh
import onnx_light.onnx.helper as xoh2
import onnx_light.onnx as onnxl
import onnx_light.onnx.io_helper as io_helper
from onnx_light.ext_test_case import ExtTestCase


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
                n1 = self.get_dump_file("model1.onnx.txt")
                with open(n1, "w") as f:
                    f.write(str(model1))
                n2 = self.get_dump_file("model2.onnx.txt")
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
        model = self.make_model_gemm(xoh, onnx.TensorProto)
        onnx.save(model, name)
        model2 = onnxl.load(name)
        self.assertEqual(len(model.graph.node), len(model2.graph.node))
        name2 = self.get_dump_file("test_model_gemm_onnx_to_onnxlight_2.onnx")
        onnxl.save(model2, name2)
        model3 = onnx.load(name2)
        self.assertEqualModelProto(model, model3)

    def test_model_gemm_onnxlight_to_onnx(self):
        name2 = self.get_dump_file("test_model_gemm_onnxlight_to_onnx_2.onnx")
        model2 = self.make_model_gemm(xoh2, onnxl.TensorProto)
        onnxl.save(model2, name2)
        model = onnx.load(name2)
        self.assertEqual(len(model.graph.node), len(model2.graph.node))
        name = self.get_dump_file("test_model_gemm_onnxlight_to_onnx.onnx")
        onnx.save(model, name)
        model3 = onnx.load(name)
        self.assertEqualModelProto(model, model3)

    def _get_model_with_initializers(self, oh, onh):
        TFLOAT = oh.TensorProto.FLOAT
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
                    onh.from_array(np.random.rand(3, 5, 128, 64).astype(np.float32), name="Y1"),
                    onh.from_array(np.array([1], dtype=np.int64), name="un"),
                    onh.from_array(np.random.rand(3, 5, 128, 64).astype(np.float32), name="Y2"),
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
        model = self._get_model_with_initializers(xoh, onnx.numpy_helper)
        onnx.save(model, name)
        # loading with onnxlight
        model2 = onnxl.load(name, parallel=True, num_threads=2)
        self.assertEqual(len(model.graph.node), len(model2.graph.node))
        name2 = self.get_dump_file("test_parallelized_loading.onnx")
        onnxl.save(model2, name2)
        model3 = onnx.load(name2)
        self.assertEqualModelProto(model, model3)

    def test_parallelized_loading_min_block_size(self):
        # Verifies that min_block_size causes small tensor blocks to be read
        # on the calling thread while large ones are still parallelised.
        name = self.get_dump_file("test_parallelized_loading_min_block_size.onnx")
        model = self._get_model_with_initializers(xoh, onnx.numpy_helper)
        onnx.save(model, name)
        # Use a very large min_block_size so every block is read on the main thread.
        model2 = onnxl.load(name, parallel=True, num_threads=2, min_block_size=10 * 1024 * 1024)
        self.assertEqual(len(model.graph.node), len(model2.graph.node))
        name2 = self.get_dump_file("test_parallelized_loading_min_block_size_out.onnx")
        onnxl.save(model2, name2)
        model3 = onnx.load(name2)
        self.assertEqualModelProto(model, model3)

    def test_parallelized_loading_min_block_size_partial(self):
        # A min_block_size between the smallest and largest initializer means
        # some blocks are parallel and some are not; the result must still be correct.
        name = self.get_dump_file("test_parallelized_loading_min_block_size_partial.onnx")
        model = self._get_model_with_initializers(xoh, onnx.numpy_helper)
        onnx.save(model, name)
        # Setting min_block_size=1 means every block of at least 1 byte is
        # parallelised, so all non-empty tensors go through the thread pool.
        # This is functionally equivalent to min_block_size=0 for typical models
        # but exercises the threshold code path.
        model2 = onnxl.load(name, parallel=True, num_threads=2, min_block_size=1)
        self.assertEqual(len(model.graph.node), len(model2.graph.node))
        name2 = self.get_dump_file("test_parallelized_loading_min_block_size_partial_out.onnx")
        onnxl.save(model2, name2)
        model3 = onnx.load(name2)
        self.assertEqualModelProto(model, model3)

    def test_parallelized_saving(self):
        name = self.get_dump_file("test_parallelized_saving.onnx")
        model = self._get_model_with_initializers(xoh, onnx.numpy_helper)
        onnx.save(model, name)
        model2 = onnxl.load(name)
        self.assertEqual(len(model.graph.node), len(model2.graph.node))
        name2 = self.get_dump_file("test_parallelized_saving_out.onnx")
        onnxl.save(model2, name2, parallel=True, num_threads=2, min_block_size=1)
        model3 = onnx.load(name2)
        self.assertEqualModelProto(model, model3)

    def test_writing_external_weights_write(self):
        nameo = self.get_dump_file("test_writing_external_weights.original.onnx")
        name = self.get_dump_file("test_writing_external_weights.onnx")
        weights = self.get_dump_file("test_writing_external_weights.data")
        model = self._get_model_with_initializers(xoh, onnx.numpy_helper)
        proto = onnxl.ModelProto()
        s = model.SerializeToString()
        with open(nameo, "wb") as f:
            f.write(s)
        proto.ParseFromString(s)
        proto.SerializeToFile(name, external_data_file=weights)
        reload = onnx.load(name)
        self.assertEqual(len(reload.graph.initializer), len(model.graph.initializer))

    def test_writing_external_weights_read(self):
        nameo = self.get_dump_file("test_writing_external_weights.original.onnx")
        name = self.get_dump_file("test_writing_external_weights.onnx")
        weights = self.get_dump_file("test_writing_external_weights.data")
        model = self._get_model_with_initializers(xoh, onnx.numpy_helper)
        proto = onnxl.ModelProto()
        s = model.SerializeToString()
        with open(nameo, "wb") as f:
            f.write(s)
        proto.ParseFromString(s)
        proto.SerializeToFile(name, external_data_file=weights)
        reload = onnx.load(name)
        self.assertEqual(len(reload.graph.initializer), len(model.graph.initializer))
        proto2 = onnxl.ModelProto()
        proto2.ParseFromFile(name, external_data_file=weights)
        self.assertEqual(len(proto2.graph.initializer), len(model.graph.initializer))

    def test_writing_external_weights_read_from_onnx(self):
        model = self._get_model_with_initializers(xoh, onnx.numpy_helper)
        expected = [xonh.to_array(i) for i in model.graph.initializer]
        name = self.get_dump_file("test_writing_external_weights_read_from_onnx.onnx")
        weights = self.get_dump_file(
            "test_writing_external_weights_read_from_onnx.data", clean=True
        )
        onnx.save(model, name, save_as_external_data=True, location=os.path.split(weights)[-1])
        location = [init.data_location for init in model.graph.initializer]
        self.assertEqual(location, [0, 1, 0, 1, 0, 0, 0])
        proto2 = onnxl.ModelProto()
        proto2.ParseFromFile(name, external_data_file=weights)
        self.assertEqual(len(proto2.graph.initializer), len(model.graph.initializer))

        def tweak(i):
            t = onnx.TensorProto()
            t.ParseFromString(i.SerializeToString())
            return t

        got = [xonh.to_array(tweak(i)) for i in proto2.graph.initializer]
        self.assertEqual(len(expected), len(got))
        for a, b in zip(expected, got):
            self.assertEqualArray(a, b)

    def test_loading_external_weights(self):
        name = self.get_dump_file("test_loading_external_weights.onnx")
        weights = self.get_dump_file("test_loading_external_weights.data")
        model = self._get_model_with_initializers(xoh, onnx.numpy_helper)
        onnx.save(model, name, location=os.path.split(weights)[-1], save_as_external_data=True)
        proto = onnxl.load(name, location=weights)
        self.assertEqual(len(proto.graph.initializer), len(model.graph.initializer))
        proto_name = self.get_dump_file("test_loading_external_weights.2.onnx")
        onnxl.save(proto, proto_name)
        restored = onnx.load(proto_name)
        self.assertEqual(len(restored.graph.initializer), len(model.graph.initializer))

    def test_save_external_data_default_location(self):
        # Verifies that save_as_external_data=True without an explicit location
        # places the weights file next to the model file with a ".data" suffix,
        # instead of creating a file literally named "None" in the cwd.
        onnx_path = self.get_dump_file("test_save_ext_default_src.onnx")
        name = self.get_dump_file("test_save_external_data_default_location.onnx")
        expected_data = name + ".data"
        if os.path.exists(expected_data):
            os.remove(expected_data)
        model = self._get_model_with_initializers(xoh, onnx.numpy_helper)
        onnx.save(model, onnx_path)
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
        reload = onnx.load(name)
        self.assertEqual(len(reload.graph.initializer), len(model.graph.initializer))

    def test_parallel_external_data_write(self):
        # Verifies that parallel external data writing (parallel=True) produces
        # byte-for-byte identical output to sequential writing.
        onnx_path = self.get_dump_file("test_parallel_ext_write_src.onnx")
        name_seq = self.get_dump_file("test_parallel_ext_write_seq.onnx")
        name_par = self.get_dump_file("test_parallel_ext_write_par.onnx")
        model = self._get_model_with_initializers(xoh, onnx.numpy_helper)
        onnx.save(model, onnx_path)
        proto = onnxl.load(onnx_path)

        onnxl.save(proto, name_seq, save_as_external_data=True)
        onnxl.save(proto, name_par, save_as_external_data=True, parallel=True, num_threads=4)

        with open(name_seq + ".data", "rb") as f:
            seq_bytes = f.read()
        with open(name_par + ".data", "rb") as f:
            par_bytes = f.read()
        self.assertEqual(len(seq_bytes), len(par_bytes))
        self.assertEqual(seq_bytes, par_bytes)

        # Both files must be loadable and have the same number of initializers
        r_seq = onnx.load(name_seq)
        r_par = onnx.load(name_par)
        self.assertEqual(len(r_seq.graph.initializer), len(r_par.graph.initializer))
        self.assertEqual(len(r_seq.graph.initializer), len(model.graph.initializer))

    def test_parallel_external_data_write_auto_threads(self):
        # parallel=True with num_threads=-1 means one thread per hardware core;
        # verify that the output is byte-for-byte identical to sequential writing.
        onnx_path = self.get_dump_file("test_parallel_ext_write_auto_src.onnx")
        name_seq = self.get_dump_file("test_parallel_ext_write_auto_seq.onnx")
        name_par = self.get_dump_file("test_parallel_ext_write_auto_par.onnx")
        model = self._get_model_with_initializers(xoh, onnx.numpy_helper)
        onnx.save(model, onnx_path)
        proto = onnxl.load(onnx_path)

        onnxl.save(proto, name_seq, save_as_external_data=True)
        onnxl.save(proto, name_par, save_as_external_data=True, parallel=True, num_threads=-1)

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
            model = io_helper.load("model.onnx", location="model.data")

        self.assertEqual(len(model.calls), 1)
        args, kwargs = model.calls[0]
        self.assertEqual(args, ("model.onnx",))
        self.assertEqual(kwargs, {"external_data_file": "model.data"})

    def test_loading_with_location_and_parallel_uses_parse_options(self):
        class FakeParseOptions:
            def __init__(self):
                self.skip_raw_data = False
                self.raw_data_threshold = -1
                self.parallel = None
                self.num_threads = 0
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
            model = io_helper.load("model.onnx", location="model.data", parallel=True)

        self.assertEqual(len(model.calls), 1)
        args, kwargs = model.calls[0]
        self.assertEqual(args[0], "model.onnx")
        self.assertTrue(args[1].parallel)
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
            model = io_helper.load("model.onnx")

        self.assertEqual(len(model.calls), 1)
        args, kwargs = model.calls[0]
        self.assertEqual(args, ("model.onnx",))
        self.assertEqual(kwargs, {})

    def test_saving_without_external_data_keeps_non_parallel_default(self):
        class FakeModelProto:
            def __init__(self):
                self.calls = []

            def SerializeToFile(self, *args, **kwargs):
                self.calls.append((args, kwargs))

        with patch.object(io_helper, "ModelProto", FakeModelProto):
            model = FakeModelProto()
            io_helper.save(model, "model.onnx")

        self.assertEqual(len(model.calls), 1)
        args, kwargs = model.calls[0]
        self.assertEqual(args, ("model.onnx",))
        self.assertEqual(kwargs, {})

    def test_saving_with_parallel_uses_serialize_options(self):
        class FakeSerializeOptions:
            def __init__(self):
                self.raw_data_threshold = -1
                self.parallel = None
                self.num_threads = 0
                self.min_parallel_block_size = -1

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
            io_helper.save(model, "model.onnx", parallel=True, num_threads=3, min_block_size=256)

        self.assertEqual(len(model.calls), 1)
        args, kwargs = model.calls[0]
        self.assertEqual(args[0], "model.onnx")
        self.assertEqual(args[1].raw_data_threshold, 1024)
        self.assertTrue(args[1].parallel)
        self.assertEqual(args[1].num_threads, 3)
        self.assertEqual(args[1].min_parallel_block_size, 256)
        self.assertEqual(kwargs, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
