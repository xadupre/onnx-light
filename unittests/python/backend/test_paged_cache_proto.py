# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests paged cache serialization, graph defaults and native feedback conversion."""

import gc
import unittest

import numpy

from onnx_light import onnx
from onnx_light.onnx import compose, helper, numpy_helper
from onnx_light.onnx_proto import verify
from onnx_light.onnx_proto._text_format import serialize_to_textproto
from onnx_light.onnx_py._onnxpycore import builder
from onnx_light.onnx_py._onnxpykernels import runtime


def make_cache():
    """Returns a dense cache with one partially filled page."""
    cache = onnx.PagedCacheProto()
    cache.name = "past"
    block = cache.blocks.add()
    block.start = 0
    block.length = 1
    data = numpy.arange(4, dtype=numpy.float32).reshape(1, 1, 2, 2)
    block.key.CopyFrom(numpy_helper.from_array(data))
    block.value.CopyFrom(numpy_helper.from_array(-data))
    return cache


def make_model():
    """Returns a model forwarding its default paged cache through feedback."""
    inputs = []
    outputs = []
    for name, values in (("past", inputs), ("present", outputs)):
        info = onnx.ValueInfoProto()
        info.name = name
        info.type.CopyFrom(onnx.PagedCacheProto.CacheType())
        values.append(info)
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Identity", ["past"], ["present"])], "paged_cache", inputs, outputs
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )
    model.graph.paged_cache_initializer.add().CopyFrom(make_cache())
    binding = model.graph.persistent_bindings.add()
    binding.input_name = "past"
    binding.output_name = "present"
    return model


class TestPagedCacheProto(unittest.TestCase):
    def test_wire_roundtrip_and_presence(self):
        cache = make_cache()
        self.assertIsInstance(cache.blocks[0], onnx.PagedCacheBlockProto)
        self.assertTrue(cache.blocks[0].HasField("start"))
        self.assertEqual(cache.blocks[0].WhichOneof("key_payload"), "key")
        parsed = onnx.PagedCacheProto()
        parsed.ParseFromString(cache.SerializeToString())
        self.assertEqual(parsed.SerializeToString(), cache.SerializeToString())
        encoded = onnx.EncodedValueProto()
        parsed.blocks[0].encoded_key = encoded
        self.assertEqual(parsed.blocks[0].WhichOneof("key_payload"), "encoded_key")
        self.assertFalse(parsed.blocks[0].HasField("key"))

    def test_model_default_feedback_restore_and_lifetime(self):
        model = onnx.ModelProto()
        model.ParseFromString(make_model().SerializeToString())
        verify.verify_model(model)
        state = runtime.PersistentValueState(model, {})
        context = runtime.RuntimeContext(runtime.KernelContext(runtime.default_opset(18)))
        outputs = state.run(context, {})
        cache = outputs["present"]
        self.assertIsInstance(cache, onnx.PagedCacheProto)
        self.assertEqual(len(cache.blocks), 1)
        state.reset({"past": cache})
        state.close()
        del outputs, state, context, model
        gc.collect()
        numpy.testing.assert_array_equal(
            numpy_helper.to_array(cache.blocks[0].key),
            numpy.arange(4, dtype=numpy.float32).reshape(1, 1, 2, 2),
        )
        restored_model = make_model()
        restored_model.graph.paged_cache_initializer[0].CopyFrom(cache)
        restored_model.graph.paged_cache_initializer[0].name = "past"
        restored = runtime.PersistentValueState(restored_model, {})
        self.assertEqual(len(restored.values["past"].blocks), 1)
        restored.close()

    def test_invalid_cache_is_rejected_before_initialization(self):
        model = make_model()
        model.graph.paged_cache_initializer[0].blocks[0].length = 3
        with self.assertRaises(ValueError):
            verify.verify_model(model)
        with self.assertRaises(ValueError):
            runtime.PersistentValueState(model, {})

    def test_nonpersistent_cache_input_uses_default_and_allows_override(self):
        model = make_model()
        model.graph.persistent_bindings.clear()
        for name, values in (
            ("counter", model.graph.input),
            ("next_counter", model.graph.output),
        ):
            values.add().CopyFrom(
                helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1])
            )
        model.graph.node.add().CopyFrom(
            helper.make_node("Identity", ["counter"], ["next_counter"])
        )
        binding = model.graph.persistent_bindings.add()
        binding.input_name = "counter"
        binding.output_name = "next_counter"
        verify.verify_model(model)
        state = runtime.PersistentValueState(
            model, {"counter": numpy.ones(1, dtype=numpy.float32)}
        )
        context = runtime.RuntimeContext(runtime.KernelContext(runtime.default_opset(18)))
        self.assertEqual(len(state.run(context, {})["present"].blocks), 1)
        self.assertEqual(
            len(state.run(context, {"past": onnx.PagedCacheProto()})["present"].blocks), 0
        )
        self.assertEqual(len(state.run(context, {})["present"].blocks), 1)
        state.close()

    def test_text_export_rejects_cache_instead_of_dropping_it(self):
        with self.assertRaisesRegex(TypeError, "paged_cache_initializer"):
            serialize_to_textproto(make_model())

    def test_builder_preserves_cache_and_legacy_composition_rejects_it(self):
        model = make_model()
        graph_builder = builder.GraphBuilder(model)
        self.assertEqual(len(graph_builder.paged_cache_initializers()), 1)
        exported = graph_builder.to_model()
        self.assertEqual(
            exported.graph.paged_cache_initializer[0].SerializeToString(),
            model.graph.paged_cache_initializer[0].SerializeToString(),
        )
        unused = make_cache()
        unused.name = "unused"
        graph_builder.make_paged_cache_initializer(unused)
        self.assertEqual(len(graph_builder.paged_cache_initializers()), 2)
        with self.assertRaisesRegex(ValueError, "paged cache"):
            compose.add_prefix_graph(model.graph, "prefix_")
        with self.assertRaisesRegex(ValueError, "paged cache"):
            compose.merge_graphs(model.graph, model.graph, [])

    def test_ort_export_rejects_cache_instead_of_dropping_it(self):
        model = helper.make_model(helper.make_graph([], "cache", [], []))
        model.graph.paged_cache_initializer.add().CopyFrom(make_cache())
        options = onnx.SerializeOptions()
        options.format = onnx.SerializeFormat.ORT_FLATBUFFERS
        with self.assertRaisesRegex((ValueError, RuntimeError), "paged cache"):
            model.SerializeToString(options=options)


if __name__ == "__main__":
    unittest.main()
