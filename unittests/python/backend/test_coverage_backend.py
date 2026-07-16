import unittest
import onnx_light.onnx as onnxl
from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx.backend import collect_test_case


class TestCoverage(ExtTestCase):
    def _count_detadata(self, tag, kind, obj):
        d = {}
        for it in obj.metadata_props:
            d[it.key] = it.value
        if tag == "release":
            if set(d) & {"onnx_light.release_after", "onnx_light.not_used_after"}:
                return 1
        if tag == "inplace" and kind in {"node", "input"}:
            if "onnx_light.inplace_reuse" in d:
                return 1
        if tag == "shape_tag":
            self.assertNotEmpty(
                {"onnx_light.node_tag", "onnx_light.value_tag"} & set(d),
                msg=lambda: f"{tag=} {kind=} {d=} {obj=}",
            )
            self.assertIn("onnx_light.value_tag", d, msg=lambda: f"{tag=} {kind=} {d=} {obj=}")
            return 1
        return 0

    def _check_test(self, tc):
        model = tc.model
        found = 0
        for obj in model.graph.node:
            found += self._count_detadata(tc.tag, "node", obj)
        for obj in model.graph.initializer:
            found += self._count_detadata(tc.tag, "init", obj)
        for obj in model.graph.input:
            found += self._count_detadata(tc.tag, "input", obj)
        for obj in model.graph.output:
            found += self._count_detadata(tc.tag, "output", obj)
        self.assertGreater(found, 0)

    def test_check_backendtest(self):
        cases = collect_test_case(include_big=True)
        qwen3 = 0
        counts = {}
        for name, tc in cases.items():
            if "qwen3" in name:
                model = tc.model
                self.assertIsInstance(model, onnxl.ModelProto)
                self.assertGreater(len(model.graph.node), 300)
                qwen3 += 1
            if tc.tag not in counts:
                counts[tc.tag] = 0
            counts[tc.tag] += 1
            if tc.tag in {"shape_tag", "inplace", "release"}:
                with self.subTest(tag=tc.tag, name=name):
                    self._check_test(tc)
        self.assertGreaterEqual(qwen3, 1)
        self.assertEqual(
            {
                "",
                "inplace",
                "ai.onnx.preview.training",
                "inference",
                "release",
                "ai.onnx.preview",
                "ai.onnx.ml",
                "ai.rt",
                "nan_inf",
                "shape_tag",
                "empty_shape",
            },
            set(counts),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
