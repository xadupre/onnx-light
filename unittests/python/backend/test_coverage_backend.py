import unittest

from onnx_light.ext_test_case import ExtTestCase, import_or_skip

import onnx_light.onnx as onnxl

# The backend test registries are only available in the full build; skip this
# module on a reduced build (ONNX_LIGHT_BUILD_KERNELS=OFF).
_backend = import_or_skip("onnx_light.onnx.backend")
collect_test_case = _backend.collect_test_case
_TestCaseTag = _backend.TestCaseTag

# Feature metadata keys (inplace / shape-tag / release) computed by the
# metadata passes. These belong on the NodeProto or ValueInfoProto/TensorProto
# they describe (node tags/lifetimes on nodes, value tags on value_info, inputs,
# outputs and initializers). They must never be attached to the graph-level
# ``metadata_props``.
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
_ALL_FEATURE_METADATA_KEYS = (
    _INPLACE_METADATA_KEYS | _SHAPE_TAG_METADATA_KEYS | _RELEASE_METADATA_KEYS
)


class TestCoverage(ExtTestCase):
    def _count_detadata(self, tag, kind, obj):
        d = {}
        for it in obj.metadata_props:
            d[it.key] = it.value
        if tag == _TestCaseTag.RELEASE:
            if set(d) & {"onnx_light.release_after", "onnx_light.not_used_after"}:
                return 1
        if tag == _TestCaseTag.INPLACE and kind in {"node", "input"}:
            if "onnx_light.inplace_reuse" in d:
                return 1
        if tag == _TestCaseTag.SHAPE_TAG:
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
                # The fused variant collapses each RMSNorm/attention subgraph
                # into a single ``RMSNormalization``/``Attention`` node, so it is
                # much smaller than the inlined variant.
                min_nodes = 150 if "fused" in name else 300
                self.assertGreater(len(model.graph.node), min_nodes)
                qwen3 += 1
            if tc.tag not in counts:
                counts[tc.tag] = 0
            counts[tc.tag] += 1
            if tc.tag in {_TestCaseTag.SHAPE_TAG, _TestCaseTag.INPLACE, _TestCaseTag.RELEASE}:
                with self.subTest(tag=tc.tag, name=name):
                    self._check_test(tc)
        self.assertGreaterEqual(qwen3, 1)
        self.assertEqual(
            {
                _TestCaseTag.NONE,
                _TestCaseTag.INPLACE,
                _TestCaseTag.AI_ONNX_PREVIEW_TRAINING,
                _TestCaseTag.INFERENCE,
                _TestCaseTag.RELEASE,
                _TestCaseTag.AI_ONNX_PREVIEW,
                _TestCaseTag.AI_ONNX_ML,
                _TestCaseTag.AI_RT,
                _TestCaseTag.NAN_INF,
                _TestCaseTag.SHAPE_TAG,
                _TestCaseTag.EMPTY_SHAPE,
                _TestCaseTag.PEAK_MEMORY,
                _TestCaseTag.CONSTANT,
            },
            set(counts),
        )

    def test_no_feature_metadata_on_model_or_graph(self):
        # Feature metadata (inplace / shape_tag / release) must be attached to
        # the NodeProto or ValueInfoProto/TensorProto it describes, never to the
        # model- or graph-level ``metadata_props``. Enforce this for every
        # backend case.
        cases = collect_test_case(include_big=True)
        for name, tc in cases.items():
            with self.subTest(tag=tc.tag, name=name):
                model_keys = {it.key for it in tc.model.metadata_props}
                graph_keys = {it.key for it in tc.model.graph.metadata_props}
                leaked = (model_keys | graph_keys) & _ALL_FEATURE_METADATA_KEYS
                self.assertFalse(
                    leaked,
                    msg=lambda leaked=leaked: (
                        f"{name=} {tc.tag=} stores feature metadata "
                        f"{sorted(leaked)} on model/graph metadata_props"
                    ),
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
