# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy

from onnx_light import onnx
from onnx_light.ext_test_case import ExtTestCase, import_or_skip
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh
from onnx_light.onnx_lib import TensorProto, parser

runtime = import_or_skip("onnx_light.onnx_py._onnxpykernels", "runtime")
add_quantization_parameters = import_or_skip(
    "onnx_light.onnx_core.quantization", "add_quantization_parameters"
)
ReferenceEvaluator = import_or_skip("onnx_light.onnx.reference", "ReferenceEvaluator")


def transition_model(dtype=numpy.float32):
    """Builds a token transition model with a three-token vocabulary."""
    model = parser.parse_model(
        '<ir_version: 10, opset_import: ["" : 23]>'
        "transition (int64[B, S] input_ids) => (float[B, S, 3] logits)"
        "{ logits = Gather(table, input_ids) }"
    )
    table = numpy.array([[0, 4, 0], [0, 0, 4], [4, 0, 0]], dtype=dtype)
    model.graph.initializer.append(onh.from_array(table, name="table"))
    model.graph.output[0].type.tensor_type.elem_type = onh.from_array(table).data_type
    return model


def _add_shared_parameters(model, name, shape, scales, prefix):
    """Adds one shared parameter set with model-unique initializer names."""
    plan = runtime.make_quantization_plan(
        runtime.QuantizationFormat.INT8, int(numpy.prod(shape)), 1
    )
    storage = runtime.make_quantization_type(plan)
    logical = oh.make_tensor_type_proto(TensorProto.FLOAT, shape)
    compact = add_quantization_parameters(
        model, name, storage, logical, scales=numpy.asarray(scales, dtype=numpy.float64)
    )
    annotation = model.graph.quantization_annotation[-1]
    renamed = {
        entry.value: f"{prefix}_{entry.value}"
        for entry in annotation.quant_parameter_tensor_names
    }
    for initializer in model.graph.initializer:
        if initializer.name in renamed:
            initializer.name = renamed[initializer.name]
    for entry in annotation.quant_parameter_tensor_names:
        entry.value = renamed[entry.value]
    return compact


def _shared_value(model, name, values):
    """Quantizes values into a retained model-local shared runtime value."""
    array = numpy.asarray(values, dtype=numpy.float32)
    tensor = runtime.tensor_from_numpy(
        name,
        TensorProto.FLOAT,
        list(array.shape),
        numpy.frombuffer(array.tobytes(), dtype=numpy.uint8),
    )
    return runtime.quantize_tensor_shared(tensor, model, name)


def _shared_encoded(model, name, values):
    """Quantizes values through one model-local shared parameter set."""
    return _shared_value(model, name, values).encoded


class TestGeneration(ExtTestCase):
    def test_multiple_eos(self):
        model = transition_model()
        context = runtime.RuntimeContext(runtime.KernelContext(runtime.default_opset(23)))
        evaluator = ReferenceEvaluator(model)
        for eos_ids, prompt, expected in [
            ([1, 2], [[0], [1]], [[0, 1], [1, 2]]),
            ([1, 2, 2], [[0], [1]], [[0, 1], [1, 2]]),
            ([0, 2], [[0], [1]], [[0, 1, 2], [1, 2, 0]]),
            ([2, 0], [[0], [1]], [[0, 1, 2], [1, 2, 2]]),
            ([0, 2], [[2]], [[2, 0]]),
            ([0], [[2]], [[2, 0]]),
            ([], [[2]], [[2, 0, 1, 2, 0]]),
        ]:
            with self.subTest(eos_ids=eos_ids, prompt=prompt):
                options = runtime.GenerationOptions()
                options.max_new_tokens = 4
                options.eos_token_ids = eos_ids
                feeds = {"input_ids": numpy.array(prompt, dtype=numpy.int64)}
                result = runtime.generate(model, context, feeds, options)
                numpy.testing.assert_array_equal(numpy.from_dlpack(result), expected)
                numpy.testing.assert_array_equal(
                    evaluator.generate(feeds, max_new_tokens=4, eos_token_ids=eos_ids), expected
                )

    def test_attention_mask_gather_releases_after_last_use(self):
        model = parser.parse_model(
            '<ir_version: 10, opset_import: ["" : 23]>'
            "decoder (int64[B, S] input_ids, int64[B, P] past_ids)"
            " => (float[B, S, 3] logits, int64[B, T] present_ids)"
            "{ present_ids = Concat <axis: int = 1> (past_ids, input_ids)"
            " dimensions = Shape(present_ids)"
            " length = Gather(dimensions, index)"
            " unused = Identity(length)"
            " mask = Greater(length, zero)"
            " raw_logits = Gather(table, input_ids)"
            " logits = Where(mask, raw_logits, raw_logits) }"
        )
        length_name = "/model/attn_mask_reformat/attn_mask_subgraph/Gather/output_0"
        model.graph.node[2].output.clear()
        model.graph.node[2].output.append(length_name)
        model.graph.node[3].input.clear()
        model.graph.node[3].input.append(length_name)
        model.graph.node[4].input.clear()
        model.graph.node[4].input.extend([length_name, "zero"])
        model.graph.node[4].metadata_props.add(key="onnx_light.release_after", value=length_name)
        model.graph.initializer.extend(
            [
                onh.from_array(numpy.array(1, dtype=numpy.int64), name="index"),
                onh.from_array(numpy.array(0, dtype=numpy.int64), name="zero"),
                transition_model().graph.initializer[0],
            ]
        )
        binding = model.graph.persistent_bindings.add()
        binding.input_name = "past_ids"
        binding.output_name = "present_ids"
        evaluator = ReferenceEvaluator(model)
        feeds = {
            "input_ids": numpy.array([[0, 1]], dtype=numpy.int64),
            "past_ids": numpy.empty((1, 0), dtype=numpy.int64),
        }
        logits, present = evaluator.run(None, feeds)
        numpy.testing.assert_array_equal(logits.argmax(axis=-1), [[1, 2]])
        numpy.testing.assert_array_equal(present, [[0, 1]])
        for _ in range(2):
            numpy.testing.assert_array_equal(
                evaluator.generate(feeds, max_new_tokens=3), [[0, 1, 2, 0, 1]]
            )

    def test_split_embedding_pipeline(self):
        embedding_model = parser.parse_model(
            '<ir_version: 10, opset_import: ["" : 23]>'
            "embedding (int64[B, S] input_ids) => (float[B, S, 3] inputs_embeds)"
            "{ raw = Gather(table, input_ids) inputs_embeds = Abs(raw) }"
        )
        embedding_model.graph.initializer.append(transition_model().graph.initializer[0])
        decoder_model = parser.parse_model(
            '<ir_version: 10, opset_import: ["" : 23]>'
            "decoder (float[B, S, 3] inputs_embeds, seq(float[1, 1, 3]) vision)"
            " => (float[B, S, 3] logits)"
            "<int64 index = {0}>"
            "{ bias = SequenceAt(vision, index) logits = Add(inputs_embeds, bias) }"
        )
        policy = {
            "num_threads": 2,
            "affinity_policy": "none",
            "spin_policy": "park_immediately",
            "allow_nested_parallelism": True,
        }
        decoder = ReferenceEvaluator(
            decoder_model, cpu_execution=policy, cpu_execution_counters=True, events_enabled=True
        )
        embedding = ReferenceEvaluator(embedding_model, cpu_execution=policy)
        embedding_calls = []

        def embedding_abs(node, value):
            embedding_calls.append(node.op_type)
            return numpy.abs(value)

        feeds = {
            "input_ids": numpy.array([[0, 1]], dtype=numpy.int64),
            "vision": [numpy.zeros((1, 1, 3), dtype=numpy.float32)],
        }

        def generate():
            return decoder.generate(feeds, max_new_tokens=3, embedding=embedding)

        before = decoder.cpu_execution_counters.dispatches
        expected = ReferenceEvaluator(transition_model()).generate(
            {"input_ids": feeds["input_ids"]}, max_new_tokens=3
        )
        for result in [generate(), generate()]:
            numpy.testing.assert_array_equal(result, expected)
        with ThreadPoolExecutor(max_workers=2) as pool:
            for result in pool.map(lambda _: generate(), range(2)):
                numpy.testing.assert_array_equal(result, expected)
        self.assertEqual(decoder.cpu_executor_instance_id, embedding.cpu_executor_instance_id)
        self.assertGreaterEqual(decoder.cpu_execution_counters.dispatches - before, 4 * 3 * 2)
        self.assertTrue({"Gather", "Abs", "Add"} <= {event.op_type for event in decoder.events()})
        decoder.register_custom_kernel("", "Abs", embedding_abs)
        numpy.testing.assert_array_equal(generate(), expected)
        self.assertEqual(len(embedding_calls), 3)
        numpy.testing.assert_array_equal(feeds["input_ids"], [[0, 1]])
        numpy.testing.assert_array_equal(
            decoder.generate(
                feeds,
                max_new_tokens=3,
                embedding=ReferenceEvaluator(
                    embedding_model, cpu_execution={"num_threads": 1, "affinity_policy": "none"}
                ),
            ),
            expected,
        )
        with self.assertRaisesRegex((ValueError, RuntimeError), "vision"):
            decoder.generate({"input_ids": feeds["input_ids"]}, embedding=embedding)
        overwriting_model = parser.parse_model(
            '<ir_version: 10, opset_import: ["" : 23]>'
            "overwriting (int64[B, S] input_ids)"
            " => (int64[B, S] input_ids)"
            "{ }"
        )
        overwriting = ReferenceEvaluator(overwriting_model, cpu_execution=policy)
        with self.assertRaisesRegex(ValueError, "graph output cannot replace generated inputs"):
            decoder.generate(feeds, embedding=overwriting)

        def failing_add(node, left, right):
            raise ValueError("generation kernel failed")

        decoder.register_custom_kernel("", "Add", failing_add)
        before_failure = len(decoder.events())
        with self.assertRaisesRegex((ValueError, RuntimeError), "generation kernel failed"):
            generate()
        failure_events = decoder.events()[before_failure:]
        self.assertTrue({"Gather", "Abs"} <= {event.op_type for event in failure_events})

    def test_split_embedding_model_scoped_nested_parameter_references(self):
        """Resolves embedding-only nested values against the embedding catalogue."""
        embedding_model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Gather", ["table", "input_ids"], ["inputs_embeds"])],
                "embedding",
                [oh.make_tensor_value_info("input_ids", TensorProto.INT64, [None, None])],
                [oh.make_tensor_value_info("inputs_embeds", TensorProto.FLOAT, [None, None, 3])],
                [onh.from_array(numpy.eye(3, dtype=numpy.float32) * 4, name="table")],
            ),
            opset_imports=[oh.make_opsetid("", 23), oh.make_opsetid("ai.rt", 1)],
            ir_version=10,
        )
        compact = _add_shared_parameters(
            embedding_model, "embedding", [3], [1, 1, 1], "embedding"
        )
        direct_type = onnx.TypeProto(struct_type=compact)
        nested_type = onnx.TypeProto(
            struct_type=onnx.StructTypeProto(
                structure=onnx.StructTypeProto.Structure(
                    field=[onnx.StructTypeProto.Structure.Field(name="payload", type=direct_type)]
                )
            )
        )
        sequence_type = onnx.TypeProto(
            sequence_type=onnx.TypeProto.Sequence(elem_type=direct_type)
        )
        for name, value_type in (
            ("direct", direct_type),
            ("nested", nested_type),
            ("sequence", sequence_type),
        ):
            embedding_model.graph.input.append(oh.make_value_info(name, value_type))

        _add_shared_parameters(embedding_model, "cache", [1, 1, 2, 2], [1, 1, 1, 1], "cache")
        cache_info = onnx.ValueInfoProto()
        cache_info.name = "cache"
        cache_info.type.struct_type = onnx.StructTypeProto()
        embedding_model.graph.input.append(cache_info)

        direct = _shared_value(embedding_model, "embedding", [1, 2, 3])
        cache_value = _shared_encoded(
            embedding_model, "cache", numpy.arange(4, dtype=numpy.float32).reshape(1, 1, 2, 2)
        )
        cache = onnx.PagedCacheProto(name="cache")
        block = cache.blocks.add()
        block.start = 0
        block.length = 2
        block.encoded_key = cache_value
        block.encoded_value = cache_value
        decoder_model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Identity", ["inputs_embeds"], ["logits"])],
                "decoder",
                [oh.make_tensor_value_info("inputs_embeds", TensorProto.FLOAT, [None, None, 3])],
                [oh.make_tensor_value_info("logits", TensorProto.FLOAT, [None, None, 3])],
            ),
            opset_imports=[oh.make_opsetid("", 23)],
            ir_version=10,
        )
        feeds = {
            "input_ids": numpy.array([[0]], dtype=numpy.int64),
            "direct": direct,
            "nested": {"payload": direct},
            "sequence": [direct],
            "cache": cache,
        }
        originals = {
            "direct": direct.encoded.SerializeToString(),
            "cache": cache.SerializeToString(),
        }
        numpy.testing.assert_array_equal(
            ReferenceEvaluator(decoder_model).generate(
                feeds, max_new_tokens=2, embedding=ReferenceEvaluator(embedding_model)
            ),
            [[0, 0, 0]],
        )
        self.assertEqual(direct.encoded.SerializeToString(), originals["direct"])
        self.assertEqual(cache.SerializeToString(), originals["cache"])

        missing = onnx.EncodedValueProto()
        missing.CopyFrom(direct.encoded)
        missing.parameter_ref = "missing"
        with self.assertRaisesRegex(ValueError, "parameter_ref"):
            ReferenceEvaluator(decoder_model).generate(
                {**feeds, "direct": missing},
                max_new_tokens=1,
                embedding=ReferenceEvaluator(embedding_model),
            )

    def test_split_embedding_shared_feed_uses_each_model_catalogue(self):
        """Binds one shared input name independently in both model scopes."""

        def make_model(embedding, shape, scales):
            prefix = "embedding" if embedding else "decoder"
            inputs = (
                [oh.make_tensor_value_info("input_ids", TensorProto.INT64, [None, None])]
                if embedding
                else []
            )
            nodes = [
                oh.make_node(
                    "Dequantize",
                    ["shared"],
                    [f"{prefix}_bias"],
                    domain="ai.rt",
                    dtype=TensorProto.FLOAT,
                ),
                oh.make_node(
                    "Unsqueeze", [f"{prefix}_bias", f"{prefix}_axes"], [f"{prefix}_expanded"]
                ),
            ]
            if embedding:
                outputs = [
                    oh.make_tensor_value_info("inputs_embeds", TensorProto.FLOAT, [1, 1, 3])
                ]
                nodes.append(oh.make_node("Identity", [f"{prefix}_expanded"], ["inputs_embeds"]))
            else:
                inputs.append(
                    oh.make_tensor_value_info("inputs_embeds", TensorProto.FLOAT, [1, 1, 3])
                )
                outputs = [oh.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 1, 3])]
                nodes.append(
                    oh.make_node("Add", ["inputs_embeds", f"{prefix}_expanded"], ["logits"])
                )
            model = oh.make_model(
                oh.make_graph(
                    nodes,
                    prefix,
                    inputs,
                    outputs,
                    [
                        onh.from_array(
                            numpy.array([0, 1], dtype=numpy.int64), name=f"{prefix}_axes"
                        )
                    ],
                ),
                opset_imports=[oh.make_opsetid("", 23), oh.make_opsetid("ai.rt", 1)],
                ir_version=10,
            )
            compact = _add_shared_parameters(model, "common", shape, scales, prefix)
            model.graph.input.append(
                oh.make_value_info("shared", onnx.TypeProto(struct_type=compact))
            )
            return model

        embedding_model = make_model(True, [3], [5, 1, 1])
        decoder_model = make_model(False, [3], [1, 5, 1])
        shared = _shared_value(embedding_model, "common", [5, 1, 4])
        feeds = {"input_ids": numpy.array([[0]], dtype=numpy.int64), "shared": shared}
        numpy.testing.assert_array_equal(
            ReferenceEvaluator(decoder_model).generate(
                feeds, max_new_tokens=1, embedding=ReferenceEvaluator(embedding_model)
            ),
            [[0, 2]],
        )

        incompatible_decoder = make_model(False, [4], [1, 1, 1, 1])
        with self.assertRaisesRegex(ValueError, "incompatible type"):
            ReferenceEvaluator(incompatible_decoder).generate(
                feeds, max_new_tokens=1, embedding=ReferenceEvaluator(embedding_model)
            )

    def test_split_embedding_persistent_state(self):
        embedding_model = parser.parse_model(
            '<ir_version: 10, opset_import: ["" : 23]>'
            "embedding (int64[B, S] input_ids) => (float[B, S, 3] inputs_embeds)"
            "{ inputs_embeds = Gather(table, input_ids) }"
        )
        embedding_model.graph.initializer.append(transition_model().graph.initializer[0])
        decoder_model = parser.parse_model(
            '<ir_version: 10, opset_import: ["" : 23]>'
            "decoder (float[B, S, 3] inputs_embeds, float[B, P, 3] past)"
            " => (float[B, T, 3] logits, float[B, T, 3] present)"
            "{ present = Concat <axis: int = 1> (past, inputs_embeds)"
            " logits = Identity(present) }"
        )
        binding = decoder_model.graph.persistent_bindings.add()
        binding.input_name = "past"
        binding.output_name = "present"
        decoder = ReferenceEvaluator(decoder_model)
        embedding = ReferenceEvaluator(embedding_model)
        feeds = {
            "input_ids": numpy.array([[0]], dtype=numpy.int64),
            "past": numpy.empty((1, 0, 3), dtype=numpy.float32),
        }

        def generate():
            return decoder.generate(feeds, max_new_tokens=4, embedding=embedding)

        expected = ReferenceEvaluator(transition_model()).generate(
            {"input_ids": feeds["input_ids"]}, max_new_tokens=4
        )
        numpy.testing.assert_array_equal(generate(), expected)
        numpy.testing.assert_array_equal(generate(), expected)
        with ThreadPoolExecutor(max_workers=2) as pool:
            for result in pool.map(lambda _: generate(), range(2)):
                numpy.testing.assert_array_equal(result, expected)
        self.assertEqual(feeds["past"].shape, (1, 0, 3))

    def test_greedy_batch_and_eos(self):
        evaluator = ReferenceEvaluator(transition_model())
        feeds = {"input_ids": numpy.array([[0], [1]], dtype=numpy.int64)}
        numpy.testing.assert_array_equal(
            evaluator.generate(feeds, max_new_tokens=3), [[0, 1, 2, 0], [1, 2, 0, 1]]
        )
        numpy.testing.assert_array_equal(
            evaluator.generate(feeds, max_new_tokens=5, eos_token_ids=[2], pad_token_id=0),
            [[0, 1, 2], [1, 2, 0]],
        )
        numpy.testing.assert_array_equal(feeds["input_ids"], [[0], [1]])
        numpy.testing.assert_array_equal(
            evaluator.generate(feeds, max_new_tokens=0), feeds["input_ids"]
        )
        numpy.testing.assert_array_equal(
            evaluator.generate(feeds, max_new_tokens=1), [[0, 1], [1, 2]]
        )

    def test_temperature(self):
        evaluator = ReferenceEvaluator(transition_model())
        feeds = {"input_ids": numpy.zeros((128, 1), dtype=numpy.int64)}
        cold = evaluator.generate(feeds, max_new_tokens=1, temperature=0.01, seed=42)
        hot = evaluator.generate(feeds, max_new_tokens=1, temperature=100, seed=42)
        numpy.testing.assert_array_equal(cold[:, -1], numpy.ones(128, dtype=numpy.int64))
        self.assertGreater(numpy.count_nonzero(hot[:, -1] != 1), 30)
        numpy.testing.assert_array_equal(
            hot, evaluator.generate(feeds, max_new_tokens=1, temperature=100, seed=42)
        )
        other = evaluator.generate(feeds, max_new_tokens=1, temperature=100, seed=43)
        self.assertFalse(numpy.array_equal(hot, other))

    def test_persistent_operator_without_append_optimization(self):
        model = parser.parse_model(
            '<ir_version: 10, opset_import: ["" : 23]>'
            "cached (int64[B, S] input_ids, int64[B, P] past_ids)"
            " => (float[B, T, 3] logits, int64[B, T] present_ids)"
            "{ present_ids = Concat <axis: int = 1> (past_ids, input_ids)"
            " logits = Gather(table, present_ids) }"
        )
        model.graph.initializer.append(transition_model().graph.initializer[0])
        binding = model.graph.persistent_bindings.add()
        binding.input_name = "past_ids"
        binding.output_name = "present_ids"
        evaluator = ReferenceEvaluator(model)
        feeds = {
            "input_ids": numpy.array([[0]], dtype=numpy.int64),
            "past_ids": numpy.empty((1, 0), dtype=numpy.int64),
        }
        for _ in range(2):
            numpy.testing.assert_array_equal(
                evaluator.generate(feeds, max_new_tokens=4), [[0, 1, 2, 0, 1]]
            )
        self.assertEqual(feeds["past_ids"].shape, (1, 0))
        with self.assertRaisesRegex(ValueError, "missing initial"):
            evaluator.generate({"input_ids": feeds["input_ids"]})
        with self.assertRaisesRegex(ValueError, "dtype mismatch"):
            evaluator.generate({**feeds, "past_ids": numpy.empty((1, 0), dtype=numpy.float32)})

    def test_persistent_state_with_multiple_consumers(self):
        model = parser.parse_model(
            '<ir_version: 10, opset_import: ["" : 23]>'
            "cached (int64[B, S] input_ids, int64[B, 1] past_state)"
            " => (float[B, 1, 3] logits, int64[B, 1] present_state)"
            "{ present_state = Add(past_state, one)"
            " logits = Gather(table, past_state) }"
        )
        model.graph.initializer.append(
            onh.from_array(numpy.ones((1, 1), dtype=numpy.int64), name="one")
        )
        model.graph.initializer.append(transition_model().graph.initializer[0])
        binding = model.graph.persistent_bindings.add()
        binding.input_name = "past_state"
        binding.output_name = "present_state"
        evaluator = ReferenceEvaluator(model)
        feeds = {
            "input_ids": numpy.array([[0]], dtype=numpy.int64),
            "past_state": numpy.array([[0]], dtype=numpy.int64),
        }
        for _ in range(2):
            numpy.testing.assert_array_equal(
                evaluator.generate(feeds, max_new_tokens=3), [[0, 1, 2, 0]]
            )
        numpy.testing.assert_array_equal(feeds["past_state"], [[0]])

    def test_floating_logits(self):
        import ml_dtypes

        for dtype in (numpy.float32, numpy.float64, numpy.float16, ml_dtypes.bfloat16):
            with self.subTest(dtype=dtype):
                result = ReferenceEvaluator(transition_model(dtype)).generate(
                    {"input_ids": numpy.array([[0, 1]], dtype=numpy.int64)}, max_new_tokens=2
                )
                numpy.testing.assert_array_equal(result, [[0, 1, 2, 0]])

    def test_bfloat16_key_value_feedback(self):
        import ml_dtypes

        model = transition_model()
        for suffix in ("key", "value"):
            past = f"past_key_values.0.{suffix}"
            present = f"present.0.{suffix}"
            model.graph.input.append(
                oh.make_tensor_value_info(past, TensorProto.BFLOAT16, [1, 4, None, 64])
            )
            model.graph.output.append(
                oh.make_tensor_value_info(present, TensorProto.BFLOAT16, [1, 4, None, 64])
            )
            model.graph.node.append(
                oh.make_node("Concat", [past, "new_cache"], [present], axis=2)
            )
            binding = model.graph.persistent_bindings.add()
            binding.input_name = past
            binding.output_name = present
        model.graph.initializer.append(
            onh.from_array(numpy.ones((1, 4, 1, 64), dtype=ml_dtypes.bfloat16), name="new_cache")
        )
        feeds = {
            "input_ids": numpy.array([[0]], dtype=numpy.int64),
            "past_key_values.0.key": numpy.empty((1, 4, 0, 64), dtype=ml_dtypes.bfloat16),
            "past_key_values.0.value": numpy.empty((1, 4, 0, 64), dtype=ml_dtypes.bfloat16),
        }
        numpy.testing.assert_array_equal(
            ReferenceEvaluator(model).generate(feeds, max_new_tokens=3), [[0, 1, 2, 0]]
        )
        self.assertEqual(feeds["past_key_values.0.key"].shape, (1, 4, 0, 64))

    def test_fixed_tensor_sequence_feed(self):
        model = parser.parse_model(
            '<ir_version: 10, opset_import: ["" : 23]>'
            "sequence_feed (int64[B, S] input_ids, seq(float[3, 3]) tables)"
            " => (float[B, S, 3] logits)"
            "<int64 index = {0}>"
            "{ table = SequenceAt(tables, index) logits = Gather(table, input_ids) }"
        )
        table = numpy.array([[0, 4, 0], [0, 0, 4], [4, 0, 0]], dtype=numpy.float32)
        result = ReferenceEvaluator(model).generate(
            {"input_ids": numpy.array([[0]], dtype=numpy.int64), "tables": [table]},
            max_new_tokens=3,
        )
        numpy.testing.assert_array_equal(result, [[0, 1, 2, 0]])

    def test_masks_positions_and_custom_names(self):
        model = parser.parse_model(
            '<ir_version: 10, opset_import: ["" : 23]>'
            "positions (int64[B, S] tokens, int64[B, S] mask, int64[B, S] positions)"
            " => (float[B, S, 3] scores)"
            "{ index = Mul(positions, mask) scores = Gather(table, index) }"
        )
        model.graph.initializer.append(
            onh.from_array(
                numpy.array([[0, 4, 0], [0, 0, 4], [4, 0, 0], [0, 4, 0]], dtype=numpy.float32),
                name="table",
            )
        )
        context = runtime.RuntimeContext(runtime.KernelContext(runtime.default_opset(23)))
        options = runtime.GenerationOptions()
        options.max_new_tokens = 3
        options.input_ids_name = "tokens"
        options.logits_name = "scores"
        options.attention_mask_name = "mask"
        options.position_ids_name = "positions"
        feeds = {
            "tokens": numpy.array([[0, 0, 0]], dtype=numpy.int64),
            "mask": numpy.array([[0, 0, 1]], dtype=numpy.int64),
        }
        numpy.testing.assert_array_equal(
            numpy.from_dlpack(runtime.generate(model, context, feeds, options)),
            [[0, 0, 0, 1, 2, 0]],
        )
        feeds["mask"] = numpy.array([[1, 0, 1]], dtype=numpy.int64)
        with self.assertRaisesRegex(ValueError, "left-padded"):
            runtime.generate(model, context, feeds, options)
        feeds["mask"] = numpy.zeros((1, 3), dtype=numpy.int64)
        with self.assertRaisesRegex(ValueError, "unmasked"):
            runtime.generate(model, context, feeds, options)

    def test_multi_axis_positions(self):
        model = parser.parse_model(
            '<ir_version: 10, opset_import: ["" : 23]>'
            "mrope (int64[B, S] input_ids, int64[3, B, S] position_ids)"
            " => (float[B, S, 3] logits, int64[3, B, S] observed)"
            "{ logits = Gather(table, input_ids) observed = Identity(position_ids) }"
        )
        model.graph.initializer.append(transition_model().graph.initializer[0])
        evaluator = ReferenceEvaluator(model)
        feeds = {"input_ids": numpy.array([[0, 1], [1, 2]], dtype=numpy.int64)}
        expected = [[0, 1, 2, 0], [1, 2, 0, 1]]
        numpy.testing.assert_array_equal(evaluator.generate(feeds, max_new_tokens=2), expected)
        coordinates = numpy.array(
            [[[1, 3], [2, 4]], [[1, 9], [2, 5]], [[1, 6], [2, 8]]], dtype=numpy.int64
        )
        feeds["position_ids"] = coordinates
        numpy.testing.assert_array_equal(evaluator.generate(feeds, max_new_tokens=2), expected)
        numpy.testing.assert_array_equal(feeds["position_ids"], coordinates)
        with self.assertRaisesRegex(ValueError, "axis count mismatch"):
            evaluator.generate(feeds, max_new_tokens=2, position_axes=2)
        with self.assertRaisesRegex(ValueError, "shape or type mismatch"):
            evaluator.generate({**feeds, "position_ids": coordinates[0]}, max_new_tokens=2)

    def test_invalid_options_and_inputs(self):
        evaluator = ReferenceEvaluator(transition_model())
        feeds = {"input_ids": numpy.array([[0]], dtype=numpy.int64)}
        for temperature in (-1, float("nan"), float("inf")):
            with (
                self.subTest(temperature=temperature),
                self.assertRaisesRegex(ValueError, "temperature"),
            ):
                evaluator.generate(feeds, temperature=temperature)
        for options in (
            {"max_new_tokens": -1},
            {"max_new_tokens": 2**63 - 1},
            {"eos_token_ids": [-1]},
            {"eos_token_ids": [2, 3]},
            {"eos_token_ids": [2, -1], "max_new_tokens": 0},
            {"pad_token_id": -1},
            {"pad_token_id": 3},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                evaluator.generate(feeds, **options)
        for tokens in (
            numpy.array([[0]], dtype=numpy.int32),
            numpy.array([0], dtype=numpy.int64),
            numpy.empty((1, 0), dtype=numpy.int64),
            numpy.array([[-1]], dtype=numpy.int64),
        ):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                evaluator.generate({"input_ids": tokens})
        with self.assertRaisesRegex(ValueError, "unknown feed"):
            evaluator.generate({**feeds, "typo": feeds["input_ids"]})
        with self.assertRaisesRegex(ValueError, "missing tensor"):
            evaluator.generate({})
        with self.assertRaisesRegex(ValueError, "ModelProto"):
            ReferenceEvaluator(transition_model().graph).generate(feeds)

    def test_invalid_logits_and_negative_infinity(self):
        for logits, message in (
            (numpy.array([[numpy.nan, 1]], dtype=numpy.float32), "NaN"),
            (numpy.array([[numpy.inf, 1]], dtype=numpy.float32), "infinity"),
            (numpy.array([[-numpy.inf, -numpy.inf]], dtype=numpy.float32), "negative infinity"),
            (numpy.array([[1, 2]], dtype=numpy.int64), "floating-point"),
            (numpy.array([1, 2], dtype=numpy.float32), "shape"),
            (numpy.empty((1, 0), dtype=numpy.float32), "shape"),
        ):
            with self.subTest(message=message):
                model = oh.make_model(
                    oh.make_graph(
                        [],
                        "constant_logits",
                        [oh.make_tensor_value_info("input_ids", TensorProto.INT64, [1, None])],
                        [oh.make_tensor_value_info("logits", TensorProto.FLOAT, None)],
                        [onh.from_array(logits, name="logits")],
                    ),
                    opset_imports=[oh.make_opsetid("", 23)],
                )
                with self.assertRaisesRegex(ValueError, message):
                    ReferenceEvaluator(model).generate(
                        {"input_ids": numpy.array([[0]], dtype=numpy.int64)}, max_new_tokens=1
                    )
        model = transition_model()
        model.graph.initializer[0].CopyFrom(
            onh.from_array(
                numpy.array([[-numpy.inf, 0, -numpy.inf]] * 3, dtype=numpy.float32), name="table"
            )
        )
        numpy.testing.assert_array_equal(
            ReferenceEvaluator(model).generate(
                {"input_ids": numpy.array([[0]], dtype=numpy.int64)},
                max_new_tokens=2,
                temperature=1,
            ),
            [[0, 1, 1]],
        )


if __name__ == "__main__":
    unittest.main()
