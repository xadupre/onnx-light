# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy

from onnx_light.ext_test_case import import_or_skip
from onnx_light.onnx_lib import TensorProto, helper, numpy_helper, parser

runtime = import_or_skip("onnx_light.onnx_py._onnxpykernels", "runtime")
ReferenceEvaluator = import_or_skip("onnx_light.onnx.reference", "ReferenceEvaluator")


def transition_model(dtype=numpy.float32):
    """Builds a token transition model with a three-token vocabulary."""
    model = parser.parse_model(
        '<ir_version: 10, opset_import: ["" : 23]>'
        "transition (int64[B, S] input_ids) => (float[B, S, 3] logits)"
        "{ logits = Gather(table, input_ids) }"
    )
    table = numpy.array([[0, 4, 0], [0, 0, 4], [4, 0, 0]], dtype=dtype)
    model.graph.initializer.append(numpy_helper.from_array(table, name="table"))
    model.graph.output[0].type.tensor_type.elem_type = numpy_helper.from_array(table).data_type
    return model


class TestGeneration(unittest.TestCase):
    def test_split_embedding_pipeline(self):
        embedding_model = parser.parse_model(
            '<ir_version: 10, opset_import: ["" : 23]>'
            "embedding (int64[B, S] tokens) => (float[B, S, 3] vectors)"
            "{ raw = Gather(table, tokens) vectors = Abs(raw) }"
        )
        embedding_model.graph.initializer.append(transition_model().graph.initializer[0])
        decoder_model = parser.parse_model(
            '<ir_version: 10, opset_import: ["" : 23]>'
            "decoder (float[B, S, 3] inputs_embeds, float[1, 1, 3] vision)"
            " => (float[B, S, 3] logits)"
            "{ logits = Add(inputs_embeds, vision) }"
        )
        vision_model = parser.parse_model(
            '<ir_version: 10, opset_import: ["" : 23]>'
            "vision (float[1, 1, 3] pixels) => (float[1, 1, 3] features)"
            "{ features = Abs(pixels) }"
        )
        policy = {"num_threads": 2, "affinity_policy": "none", "spin_policy": "park_immediately"}
        decoder = ReferenceEvaluator(
            decoder_model, cpu_execution=policy, cpu_execution_counters=True
        )
        embedding = ReferenceEvaluator(embedding_model, cpu_execution=policy)
        vision = ReferenceEvaluator(vision_model, cpu_execution=policy)
        feeds = {
            "input_ids": numpy.array([[0, 1]], dtype=numpy.int64),
            "image": numpy.zeros((1, 1, 3), dtype=numpy.float32),
        }

        def generate():
            return decoder.generate(
                feeds,
                max_new_tokens=3,
                embedding=embedding,
                embedding_token_input="tokens",
                embedding_outputs={"vectors": "inputs_embeds"},
                prefill=[(vision, {"pixels": "image"}, {"features": "vision"})],
            )

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
        self.assertEqual(decoder.cpu_executor_instance_id, vision.cpu_executor_instance_id)
        self.assertGreaterEqual(
            decoder.cpu_execution_counters.dispatches - before, 4 * (3 * 2 + 1)
        )
        numpy.testing.assert_array_equal(feeds["input_ids"], [[0, 1]])
        with self.assertRaisesRegex(ValueError, "compatible CPU"):
            decoder.generate(
                feeds,
                embedding=ReferenceEvaluator(
                    embedding_model, cpu_execution={"num_threads": 1, "affinity_policy": "none"}
                ),
                embedding_token_input="tokens",
                embedding_outputs={"vectors": "inputs_embeds"},
            )
        with self.assertRaisesRegex(ValueError, "missing component feed"):
            decoder.generate(
                {"input_ids": feeds["input_ids"]},
                embedding=embedding,
                embedding_token_input="tokens",
                embedding_outputs={"vectors": "inputs_embeds"},
                prefill=[(vision, {"pixels": "image"}, {"features": "vision"})],
            )

    def test_split_embedding_persistent_state(self):
        embedding_model = parser.parse_model(
            '<ir_version: 10, opset_import: ["" : 23]>'
            "embedding (int64[B, S] tokens) => (float[B, S, 3] vectors)"
            "{ vectors = Gather(table, tokens) }"
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
            return decoder.generate(
                feeds,
                max_new_tokens=4,
                embedding=embedding,
                embedding_token_input="tokens",
                embedding_outputs={"vectors": "inputs_embeds"},
            )

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
            evaluator.generate(feeds, max_new_tokens=5, eos_token_id=2, pad_token_id=0),
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
                helper.make_tensor_value_info(past, TensorProto.BFLOAT16, [1, 4, None, 64])
            )
            model.graph.output.append(
                helper.make_tensor_value_info(present, TensorProto.BFLOAT16, [1, 4, None, 64])
            )
            model.graph.node.append(
                helper.make_node("Concat", [past, "new_cache"], [present], axis=2)
            )
            binding = model.graph.persistent_bindings.add()
            binding.input_name = past
            binding.output_name = present
        model.graph.initializer.append(
            numpy_helper.from_array(
                numpy.ones((1, 4, 1, 64), dtype=ml_dtypes.bfloat16), name="new_cache"
            )
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
            numpy_helper.from_array(
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
            {"eos_token_id": -1},
            {"eos_token_id": 3},
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
                model = helper.make_model(
                    helper.make_graph(
                        [],
                        "constant_logits",
                        [
                            helper.make_tensor_value_info(
                                "input_ids", TensorProto.INT64, [1, None]
                            )
                        ],
                        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, None)],
                        [numpy_helper.from_array(logits, name="logits")],
                    ),
                    opset_imports=[helper.make_opsetid("", 23)],
                )
                with self.assertRaisesRegex(ValueError, message):
                    ReferenceEvaluator(model).generate(
                        {"input_ids": numpy.array([[0]], dtype=numpy.int64)}, max_new_tokens=1
                    )
        model = transition_model()
        model.graph.initializer[0].CopyFrom(
            numpy_helper.from_array(
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
