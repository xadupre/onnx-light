import unittest

from onnx_light.ext_test_case import ExtTestCase, import_or_skip

import onnx_light.onnx as onnxl

# The backend test registries are only available in the full build; skip this
# module on a reduced build (ONNX_LIGHT_BUILD_KERNELS=OFF).
collect_test_case = import_or_skip("onnx_light.onnx.backend", "collect_test_case")

# Feature metadata keys (inplace / shape-tag / release) that a backend test
# case may only carry when it is one of the cases specifically designed to
# pre-embed such metadata for the coverage checks (see
# ``test_backend_with_metadata_coverage.py``). Every other case must ship no
# such metadata in its model or graph.
_FEATURE_METADATA_KEYS = frozenset(
    {
        "onnx_light.inplace_reuse",
        "onnx_light.node_tag",
        "onnx_light.value_tag",
        "onnx_light.value_tags",
        "onnx_light.release_after",
        "onnx_light.not_used_after",
        "onnx_light.release_after_shape_tag",
    }
)

# Name prefixes of the backend test cases that intentionally pre-embed the
# feature metadata above. These mirror the include patterns consumed by
# ``TestBackendMetadataCoverage``.
_METADATA_CASE_PREFIXES = ("test_cc_shape_inference_", "test_cc_release_", "test_cc_shape_tag_")


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
                "peak_memory",
            },
            set(counts),
        )

    def _iter_metadata_objects(self, model):
        yield "graph", model.graph
        for obj in model.graph.node:
            yield "node", obj
        for obj in model.graph.initializer:
            yield "initializer", obj
        for obj in model.graph.input:
            yield "input", obj
        for obj in model.graph.output:
            yield "output", obj
        for obj in model.graph.value_info:
            yield "value_info", obj

    def test_no_feature_metadata_on_untagged_tests(self):
        # A backend test case that is not one of the cases specifically built to
        # pre-embed metadata (see ``_METADATA_CASE_PREFIXES``) must not store any
        # inplace/shape_tag/release information in its model or graph metadata.
        cases = collect_test_case(include_big=True)
        for name, tc in cases.items():
            if name.startswith(_METADATA_CASE_PREFIXES):
                continue
            with self.subTest(tag=tc.tag, name=name):
                for kind, obj in self._iter_metadata_objects(tc.model):
                    keys = {it.key for it in obj.metadata_props}
                    leaked = keys & _FEATURE_METADATA_KEYS
                    self.assertFalse(
                        leaked,
                        msg=lambda leaked=leaked, kind=kind: (
                            f"{name=} {tc.tag=} leaks metadata {sorted(leaked)} on {kind}"
                        ),
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
