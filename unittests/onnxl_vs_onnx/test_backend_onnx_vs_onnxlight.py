import os
import re
import unittest
import onnx
import onnx_light.onnx as onnxl
from onnx.backend.test.loader import load_model_tests
from onnx_light.ext_test_case import ExtTestCase


class TestOnnxVsOnnxLight(ExtTestCase):
    regs = [
        (re.compile("(adagrad|adam)"), "training"),
        (re.compile("(if_opt)"), "attribute with a TypeProto"),
    ]

    @classmethod
    def filter_out(cls, model_name):
        for reg, reason in cls.regs:
            if reg.search(model_name):
                return reason
        return False

    @classmethod
    def add_test_methods(cls):
        tests = load_model_tests(kind="node")
        for test in tests:
            model_file = None
            if test.model_dir is not None:
                candidate = os.path.join(test.model_dir, "model.onnx")
                if os.path.exists(candidate):
                    model_file = candidate
            # Recent onnx releases ship the model only in memory (model_dir is
            # None); older releases ship it on disk. Skip an entry only when it
            # provides neither.
            if model_file is None and test.model is None:
                continue
            short_name = test.name.replace("test_", "", 1)
            reason = cls.filter_out(test.name)

            def _make(model_file, model, short_name):
                def _test_(self):
                    if model_file is not None:
                        onx = onnx.load(model_file)
                    else:
                        onx = model
                    self.run_test(onx, short_name, model_file=model_file)

                return _test_

            _test_ = _make(model_file, test.model, short_name)
            if reason:
                _test_ = unittest.skip(reason)(_test_)
            setattr(cls, f"test_vs_{short_name}", _test_)

    def break_into_pieces(self, onx, short_name):
        pieces = [onx, onx.graph, *onx.graph.node]
        for p in pieces:
            s = p.SerializeToString()
            st = p.__class__.__name__
            t2 = getattr(onnxl, st)
            o2 = t2()
            name = p.op_type if st == "NodeProto" else getattr(p, "name", "NONE")
            try:
                o2.ParseFromString(s)
            except Exception as e:
                print(f"-- {st}: FAIL due to {e} ({name!r})")
                filename = self.get_dump_file(f"{short_name}.{name}.onnx")
                with open(filename, "wb") as f:
                    f.write(s)
                with open(filename + ".txt", "w") as f:
                    f.write(f"{e}\n----\n{str(p)}")

    def look_into_pieces(self, model2: onnxl.ModelProto, short_name: str):
        assert isinstance(model2, onnxl.ModelProto), f"unexpected type ({type(model2)})"
        pieces = [model2, model2.graph, *model2.graph.node]
        for p in pieces:
            s = p.SerializeToString()
            st = p.__class__.__name__
            t2 = getattr(onnx, st)
            o2 = t2()
            name = p.op_type if st == "NodeProto" else getattr(p, "name", "NONE")
            try:
                o2.ParseFromString(s)
            except Exception as e:
                print(f"-- {st}: FAIL due to {e} ({name!r})")
                filename = self.get_dump_file(f"{short_name}.{name}.onnx")
                with open(filename, "wb") as f:
                    f.write(s)
                with open(filename + ".txt", "w") as f:
                    f.write(f"{e}\n----\n{str(p)}")

    def compare_pieces(self, model: onnx.ModelProto, short_name: str):
        assert isinstance(model, onnx.ModelProto), f"unexpected type ({type(model)})"
        pieces = [*model.graph.node, model.graph, model]
        for p in pieces:
            s = p.SerializeToString()
            st = p.__class__.__name__
            t2 = getattr(onnxl, st)
            o2 = t2()
            name = p.op_type if st == "NodeProto" else getattr(p, "name", "NONE")
            o2.ParseFromString(s)
            s2 = o2.SerializeToString()
            # back
            t3 = getattr(onnx, st)
            o3 = t3()
            o3.ParseFromString(s2)
            s3 = o3.SerializeToString()
            if s != s3:
                filename = self.get_dump_file(f"{short_name}.{name}.1.onnx")
                with open(filename, "wb") as f:
                    f.write(s)
                with open(filename + ".txt", "w") as f:
                    f.write(str(p))
                filename = self.get_dump_file(f"{short_name}.{name}.2.onnx")
                with open(filename + ".txt", "w") as f:
                    f.write(str(o2))
                filename = self.get_dump_file(f"{short_name}.{name}.3.onnx")
                with open(filename + ".txt", "w") as f:
                    f.write(str(o3))
            self.assertEqual(s, s3)

    def run_test(self, onx, short_name, model_file=None):
        """Compares the ``onnx`` and ``onnx_light`` serialization of one model.

        ``onx`` is the reference :class:`onnx.ModelProto`. ``model_file`` is the
        on-disk ``model.onnx`` path when the ONNX backend test ships one (older
        ``onnx`` releases) so ``onnx_light`` is exercised through its file loader;
        it is ``None`` when the backend test only provides the model in memory
        (recent ``onnx`` releases), in which case ``onnx_light`` parses the
        serialized bytes instead.
        """
        if onx.ir_version <= 3:
            raise unittest.SkipTest("ir_version={ir_version} too old")
        model_bytes = onx.SerializeToString()
        try:
            if model_file is not None:
                onx2 = onnxl.load(model_file)
            else:
                onx2 = onnxl.ModelProto()
                onx2.ParseFromString(model_bytes)
        except RuntimeError as e:
            name = self.get_dump_file(f"{short_name}.cannotload.onnx")
            with open(name, "wb") as f:
                f.write(model_bytes)
            self.break_into_pieces(onx, short_name)
            with open(name + ".txt", "w") as f:
                f.write(str(onx))
            content = model_bytes
            rows = []
            for i in range(0, len(content), 20):
                rows.append(f"{i:03d}: {content[i:min(i + 10, len(content))]}")
            if len(rows) >= 20:
                rows[20] = "..."
                del rows[21:-10]
            msg = "\n".join(rows)
            raise AssertionError(
                f"Unable to load {short_name!r} with onnxlight.\n---\n{msg}"
            ) from e
        self.assertEqual(len(onx.graph.node), len(onx2.graph.node))

        # compare the serialized string with onnxlight format
        with self.subTest(fmt="onnxlight"):
            s = model_bytes
            onx_onnxl = onnxl.ModelProto()
            onx_onnxl.ParseFromString(s)
            b = onx_onnxl.SerializeToString()
            a = onx2.SerializeToString()
            if a != b:
                f1 = self.get_dump_file(short_name + ".original2.onnx")
                with open(f1, "wb") as f:
                    f.write(a)
                with open(f1 + ".txt", "w") as f:
                    f.write(str(onx2))
                f2 = self.get_dump_file(short_name + ".original2_to_onnx.onnx")
                with open(f2, "wb") as f:
                    f.write(b)
                with open(f2 + ".txt", "w") as f:
                    f.write(str(onx_onnxl))
            self.assertEqual(a, b)

        # compare the serialized string with onnx format
        with self.subTest(fmt="onnx"):
            s2 = onx2.SerializeToString()
            onx2_onnx = onnx.ModelProto()
            onnxl.ModelProto().ParseFromString(s2)
            try:
                onx2_onnx.ParseFromString(s2)
            except Exception:
                rname = self.get_dump_file(f"{short_name}.onnx")
                with open(rname, "wb") as f:
                    f.write(model_bytes)
                with open(rname + ".txt", "w") as f:
                    f.write(str(onx))
                self.look_into_pieces(onx2, short_name + ".cannotparse")
                raise
            a = model_bytes
            b = onx2_onnx.SerializeToString()
            if a != b:
                self.compare_pieces(onx, short_name)
                f1 = self.get_dump_file(short_name + ".original.onnx")
                with open(f1, "wb") as f:
                    f.write(a)
                with open(f1 + ".txt", "w") as f:
                    f.write(str(onx))
                f2 = self.get_dump_file(short_name + ".original_to_onnxlight.onnx")
                with open(f2, "wb") as f:
                    f.write(b)
                with open(f2 + ".txt", "w") as f:
                    f.write(str(onx2_onnx))
            self.assertEqual(a, b)


TestOnnxVsOnnxLight.add_test_methods()

if __name__ == "__main__":
    unittest.main(verbosity=2, exit=False)
