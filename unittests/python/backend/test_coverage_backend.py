import unittest

from onnx_light.ext_test_case import ExtTestCase, import_or_skip

import onnx_light.onnx as onnxl

# The backend test registries are only available in the full build; skip this
# module on a reduced build (ONNX_LIGHT_BUILD_KERNELS=OFF).
collect_test_case = import_or_skip("onnx_light.onnx.backend", "collect_test_case")

# Metadata keys that must only appear on test cases explicitly tagged for the
# corresponding feature ("inplace", "shape_tag" or "release"). Any other test
# case that stores one of these keys leaks pre-computed information into the
# model or graph metadata and must be caught here.
_INPLACE_METADATA_KEYS = frozenset({"onnx_light.inplace_reuse"})
_SHAPE_TAG_METADATA_KEYS = frozenset(
    {"onnx_light.node_tag", "onnx_light.value_tag", "onnx_light.value_tags"}
)
_RELEASE_METADATA_KEYS = frozenset(
    {
        "onnx_light.release_after",
        "onnx_light.not_used_after",
        "onnx_light.release_after_shape_tag",
    }
)
_TAGGED_METADATA_KEYS = {
    "inplace": _INPLACE_METADATA_KEYS,
    "shape_tag": _SHAPE_TAG_METADATA_KEYS,
    "release": _RELEASE_METADATA_KEYS,
}
_ALL_FEATURE_METADATA_KEYS = (
    _INPLACE_METADATA_KEYS | _SHAPE_TAG_METADATA_KEYS | _RELEASE_METADATA_KEYS
)


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
        # A test case that is not explicitly tagged "inplace", "shape_tag" or
        # "release" must not store any inplace/shape_tag/release information in
        # its model or graph metadata. A test case tagged for one feature must
        # not leak the metadata keys of the *other* features either.
        cases = collect_test_case(include_big=True)
        for name, tc in cases.items():
            allowed = _TAGGED_METADATA_KEYS.get(tc.tag, frozenset())
            forbidden = _ALL_FEATURE_METADATA_KEYS - allowed
            with self.subTest(tag=tc.tag, name=name):
                for kind, obj in self._iter_metadata_objects(tc.model):
                    keys = {it.key for it in obj.metadata_props}
                    leaked = keys & forbidden
                    self.assertFalse(
                        leaked,
                        msg=lambda leaked=leaked, kind=kind: (
                            f"{name=} {tc.tag=} leaks metadata {sorted(leaked)} on {kind}"
                        ),
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
