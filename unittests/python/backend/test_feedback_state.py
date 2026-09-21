# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests persistent model feedback through the native bindings."""

import gc
import subprocess
import sys
import threading
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy

from onnx_light import onnx
from onnx_light.ext_test_case import import_or_skip
from onnx_light.onnx import helper, numpy_helper
from onnx_light.onnx_lib import parser
from onnx_light.onnx_proto import verify

runtime = import_or_skip("onnx_light.onnx_py._onnxpykernels", "runtime")


def make_model():
    """Returns an accumulator with graph-declared feedback."""
    model = parser.parse_model(
        '<ir_version: 10, opset_import: ["" : 18]>'
        "accumulate (float[2] delta, float[2] past) => (float[2] present) "
        "{ present = Add(delta, past) }"
    )
    add_binding(model, "past", "present")
    return model


def add_binding(model, input_name, output_name):
    """Declares a persistent input/output binding."""
    binding = model.graph.persistent_bindings.add()
    binding.input_name = input_name
    binding.output_name = output_name


def make_context():
    """Returns an independent runtime context."""
    return runtime.RuntimeContext(runtime.KernelContext(runtime.default_opset(18)))


def array(tensor):
    """Returns a zero-copy read-only view of a runtime tensor."""
    return numpy.from_dlpack(tensor)


class TestFeedbackState(unittest.TestCase):
    def test_context_retention_is_bounded(self):
        state = runtime.FeedbackState(make_model(), {"past": numpy.zeros(2, dtype=numpy.float32)})
        context = make_context()
        feeds = {"delta": numpy.ones(2, dtype=numpy.float32)}
        state.run(context, feeds)
        references = sys.getrefcount(context)
        for _ in range(100):
            state.run(context, feeds)
        self.assertEqual(sys.getrefcount(context), references)

    def test_shared_string_computation(self):
        model = parser.parse_model(
            '<ir_version: 10, opset_import: ["" : 20]>'
            "strings (string[2] past, string[2] suffix) => (string[2] present)"
            "{ present = StringConcat(past, suffix) }"
        )
        add_binding(model, "past", "present")
        initial = runtime.tensor_from_proto(
            numpy_helper.from_array(numpy.array(["a", "b"], dtype=object))
        )
        suffix = runtime.tensor_from_proto(
            numpy_helper.from_array(numpy.array(["!", "?"], dtype=object))
        )
        state = runtime.FeedbackState(model, {"past": initial})
        context = runtime.RuntimeContext(runtime.KernelContext(runtime.default_opset(20)))
        for expected in (["a!", "b?"], ["a!!", "b??"]):
            result = state.run(context, {"suffix": suffix})["present"]
            numpy.testing.assert_array_equal(
                numpy_helper.to_array(runtime.tensor_to_proto(result)), expected
            )

    def test_global_callback_interpreter_shutdown(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from onnx_light.onnx_py._onnxpykernels import runtime; "
                    "runtime.register_custom_kernel('feedback.test', 'Shutdown', "
                    "lambda n, c: None)"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_matches_manual_loop_and_reset(self):
        model = make_model()
        initial = numpy.zeros(2, dtype=numpy.float32)
        state = runtime.FeedbackState(model, {"past": initial})
        context = make_context()
        manual = runtime.tensor_from_proto(numpy_helper.from_array(initial, name="past"))
        for index in range(4):
            delta = numpy.array([index, 1], dtype=numpy.float32)
            expected = runtime.run_model(
                model,
                [runtime.tensor_from_proto(numpy_helper.from_array(delta, name="delta")), manual],
            )[0]
            actual = state.run(context, {"delta": delta})["present"]
            numpy.testing.assert_array_equal(array(expected), array(actual))
            manual = expected
            manual.name = "past"
        state.reset({"past": initial})
        numpy.testing.assert_array_equal(array(state.values["past"]), initial)
        state.close()
        with self.assertRaises((ValueError, RuntimeError)):
            state.run(context, {"delta": initial})

    def test_ownership_and_request_isolation(self):
        model = make_model()
        initial = numpy.ones(2, dtype=numpy.float32)
        first = runtime.FeedbackState(model, {"past": initial})
        second = runtime.FeedbackState(model, {"past": initial})
        self.assertEqual(array(first.values["past"]).ctypes.data, initial.ctypes.data)
        self.assertEqual(array(second.values["past"]).ctypes.data, initial.ctypes.data)
        del model
        gc.collect()
        output = first.run(make_context(), {"delta": numpy.ones(2, dtype=numpy.float32)})
        self.assertEqual(
            array(output["present"]).ctypes.data, array(first.values["past"]).ctypes.data
        )
        output["present"].shape = [1, 2]
        snapshot = first.values
        snapshot["past"].data_type = int(onnx.TensorProto.INT32)
        numpy.testing.assert_array_equal(array(first.values["past"]), [2, 2])
        numpy.testing.assert_array_equal(array(second.values["past"]), [1, 1])
        first.close()
        numpy.testing.assert_array_equal(array(output["present"]), [[2, 2]])

    def test_validation_and_rewrites(self):
        initial = numpy.zeros(2, dtype=numpy.float32)
        model = make_model()
        model.graph.persistent_bindings[0].input_name = "absent"
        with self.assertRaises(ValueError):
            runtime.FeedbackState(model, {"absent": initial})
        model = make_model()
        with self.assertRaises(ValueError):
            runtime.FeedbackState(model, {})
        with self.assertRaises(ValueError):
            runtime.FeedbackState(model, {"past": numpy.zeros(3, dtype=numpy.float32)})
        state = runtime.FeedbackState(model, {"past": initial})
        with self.assertRaises(ValueError):
            state.run(make_context(), {})
        with self.assertRaises(ValueError):
            state.run(make_context(), {"past": initial, "delta": initial})
        rewritten = make_model()
        rewritten.graph.output[0].name = "rewritten"
        with self.assertRaises(ValueError):
            runtime.FeedbackState(rewritten, {"past": initial})

    def test_cancelled_call_preserves_state(self):
        state = runtime.FeedbackState(make_model(), {"past": numpy.zeros(2, dtype=numpy.float32)})
        completion = runtime.TaskCompletion()
        completion.cancel("request cancelled")
        with self.assertRaises((ValueError, RuntimeError)):
            state.run(make_context(), {"delta": numpy.ones(2, dtype=numpy.float32)}, completion)
        numpy.testing.assert_array_equal(array(state.values["past"]), [0, 0])
        self.assertTrue(completion.is_ready())

    def test_inflight_cancellation_and_concurrent_operation_rejection(self):
        model = parser.parse_model(
            '<ir_version: 10, opset_import: ["" : 18, "feedback.test" : 1]>'
            "blocked (float[2] past) => (float[2] present)"
            "{ present = feedback.test.Block(past) }"
        )
        add_binding(model, "past", "present")
        for global_registration in (False, True):
            with self.subTest(global_registration=global_registration):
                initial = numpy.zeros(2, dtype=numpy.float32)
                state = runtime.FeedbackState(model, {"past": initial})
                context = make_context()
                entered = threading.Event()
                resume = threading.Event()
                completion = runtime.TaskCompletion()

                def block(node, context):
                    """Waits for cancellation before producing an output."""
                    entered.set()
                    if not resume.wait(10):
                        raise RuntimeError("test timed out waiting for cancellation")
                    context.put_value(str(node.output[0]), numpy.ones(2, dtype=numpy.float32))

                registry = runtime if global_registration else context
                registry.register_custom_kernel("feedback.test", "Block", block)
                try:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        pending = executor.submit(state.run, context, {}, completion)
                        try:
                            self.assertTrue(entered.wait(10))
                            for operation in (
                                lambda: state.run(context, {}),
                                lambda: state.reset({"past": initial}),
                                state.close,
                            ):
                                with self.assertRaises(ValueError):
                                    operation()
                            completion.cancel("cancel during execution")
                        finally:
                            resume.set()
                        with self.assertRaises((ValueError, RuntimeError)):
                            pending.result(timeout=10)
                    numpy.testing.assert_array_equal(array(state.values["past"]), initial)
                    state.close()
                finally:
                    registry.unregister_custom_kernel("feedback.test", "Block")

    def test_whole_structured_feedback_and_kernel_failure(self):
        tensor_type = helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, [2])

        def struct_type(*names):
            """Returns a declaration containing named tensor fields."""
            return onnx.TypeProto(
                struct_type=onnx.StructTypeProto(
                    structure=onnx.StructTypeProto.Structure(
                        field=[
                            onnx.StructTypeProto.Structure.Field(name=name, type=tensor_type)
                            for name in names
                        ]
                    )
                )
            )

        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node(
                        "Step", ["request", "tokens"], ["response"], domain="feedback.test"
                    )
                ],
                "structured_feedback",
                [
                    onnx.ValueInfoProto(name="request", type=struct_type("logits", "cache")),
                    onnx.ValueInfoProto(name="tokens", type=tensor_type),
                ],
                [onnx.ValueInfoProto(name="response", type=struct_type("logits", "cache"))],
            ),
            opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("feedback.test", 1)],
        )
        add_binding(model, "request", "response")
        verify.verify_model(model)
        initial_cache = numpy.zeros(2, dtype=numpy.float32)
        initial_logits = numpy.zeros(2, dtype=numpy.float32)
        cache_ref, logits_ref = weakref.ref(initial_cache), weakref.ref(initial_logits)
        state = runtime.FeedbackState(
            model, {"request": {"cache": initial_cache, "logits": initial_logits}}
        )
        snapshot = state.values
        self.assertEqual(
            array(snapshot["request"]["cache"]).ctypes.data, initial_cache.ctypes.data
        )
        self.assertEqual(
            array(snapshot["request"]["logits"]).ctypes.data, initial_logits.ctypes.data
        )
        del initial_cache, initial_logits
        gc.collect()
        self.assertIsNotNone(cache_ref())
        self.assertIsNotNone(logits_ref())
        context = make_context()
        fail = False

        def step(node, context):
            """Produces structured outputs through the ordinary custom-kernel API."""
            request = context.get_value(str(node.input[0]))
            cache = array(request["cache"]) + array(context.get_value(str(node.input[1])))
            context.put_value(str(node.output[0]), {"cache": cache, "logits": cache * 2})
            if fail:
                raise RuntimeError("failed after producing outputs")

        context.register_custom_kernel("feedback.test", "Step", step)
        for expected in (1, 2):
            output = state.run(context, {"tokens": numpy.ones(2, dtype=numpy.float32)})
            numpy.testing.assert_array_equal(array(output["response"]["cache"]), [expected] * 2)
            for field in ("cache", "logits"):
                self.assertEqual(
                    array(state.values["request"][field]).ctypes.data,
                    array(output["response"][field]).ctypes.data,
                )
        self.assertEqual(set(state.values), {"request"})
        for replacement in (
            {},
            {"cache": numpy.ones(2, dtype=numpy.float32)},
            state.values["request"],
        ):
            with self.assertRaisesRegex(ValueError, "override retained input"):
                state.run(
                    context,
                    {"request": replacement, "tokens": numpy.ones(2, dtype=numpy.float32)},
                )
        with self.assertRaisesRegex(ValueError, "unknown graph input"):
            state.run(context, {"request.cache": numpy.ones(2, dtype=numpy.float32)})
        with self.assertRaisesRegex(ValueError, "missing field"):
            state.reset({"request": {"cache": numpy.zeros(2, dtype=numpy.float32)}})
        fail = True
        with self.assertRaisesRegex(RuntimeError, "failed after producing outputs"):
            state.run(context, {"tokens": numpy.ones(2, dtype=numpy.float32)})
        numpy.testing.assert_array_equal(array(state.values["request"]["cache"]), [2, 2])
        numpy.testing.assert_array_equal(array(state.values["request"]["logits"]), [4, 4])
        fail = False
        output = state.run(context, {"tokens": numpy.ones(2, dtype=numpy.float32)})
        numpy.testing.assert_array_equal(array(output["response"]["cache"]), [3, 3])
        state.close()
        numpy.testing.assert_array_equal(array(snapshot["request"]["cache"]), [0, 0])
        del snapshot
        gc.collect()
        self.assertIsNone(cache_ref())
        self.assertIsNone(logits_ref())

    def test_source_and_output_lifetimes(self):
        initial = numpy.zeros(2, dtype=numpy.float32)
        source_ref = weakref.ref(initial)
        address = initial.ctypes.data
        state = runtime.FeedbackState(make_model(), {"past": initial})
        retained = state.values["past"]
        self.assertEqual(array(retained).ctypes.data, address)
        del initial
        gc.collect()
        self.assertIsNotNone(source_ref())
        output = state.run(make_context(), {"delta": numpy.ones(2, dtype=numpy.float32)})
        saved = array(output["present"])
        state.reset({"past": numpy.zeros(2, dtype=numpy.float32)})
        state.close()
        del state, output
        gc.collect()
        numpy.testing.assert_array_equal(saved, [1, 1])
        numpy.testing.assert_array_equal(array(retained), [0, 0])
        del retained
        gc.collect()
        self.assertIsNone(source_ref())

    def test_next_invocation_receives_same_output_buffer(self):
        model = parser.parse_model(
            '<ir_version: 10, opset_import: ["" : 18, "feedback.test" : 1]>'
            "observe (float[2] past) => (float[2] present)"
            "{ present = feedback.test.Observe(past) }"
        )
        add_binding(model, "past", "present")
        initial = numpy.zeros(2, dtype=numpy.float32)
        state = runtime.FeedbackState(model, {"past": initial})
        context = make_context()
        addresses = []

        def observe(node, context):
            """Forwards the same storage through the custom-kernel boundary."""
            value = context.get_value(str(node.input[0]))
            addresses.append(array(value).ctypes.data)
            context.put_value(str(node.output[0]), value)

        context.register_custom_kernel("feedback.test", "Observe", observe)
        for _ in range(3):
            output = state.run(context, {})
            self.assertEqual(array(output["present"]).ctypes.data, initial.ctypes.data)
            self.assertEqual(array(state.values["past"]).ctypes.data, initial.ctypes.data)
        self.assertEqual(addresses, [initial.ctypes.data] * 3)

    def test_runtime_tensor_input_without_copy(self):
        source = runtime.tensor_from_proto(
            numpy_helper.from_array(numpy.ones(2, dtype=numpy.float32))
        )
        address = array(source).ctypes.data
        state = runtime.FeedbackState(make_model(), {"past": source})
        self.assertEqual(array(state.values["past"]).ctypes.data, address)
        del source
        gc.collect()
        numpy.testing.assert_array_equal(array(state.values["past"]), [1, 1])

    def test_scalar_numpy_input_without_copy(self):
        model = parser.parse_model(
            '<ir_version: 10, opset_import: ["" : 18]>'
            "scalar (float past) => (float present) { present = Identity(past) }"
        )
        add_binding(model, "past", "present")
        initial = numpy.array(3.0, dtype=numpy.float32)
        state = runtime.FeedbackState(model, {"past": initial})
        self.assertEqual(array(state.values["past"]).ctypes.data, initial.ctypes.data)
        output = state.run(make_context(), {})
        self.assertEqual(array(output["present"]).ctypes.data, initial.ctypes.data)
        del initial, state
        gc.collect()
        numpy.testing.assert_array_equal(array(output["present"]), numpy.float32(3))

    def test_pytorch_input_without_copy(self):
        torch = import_or_skip("torch")
        source = torch.zeros(2, dtype=torch.float32)
        address = source.data_ptr()
        state = runtime.FeedbackState(make_model(), {"past": source})
        self.assertEqual(array(state.values["past"]).ctypes.data, address)
        del source
        gc.collect()
        delta = torch.ones(2, dtype=torch.float32)
        output = state.run(make_context(), {"delta": delta})
        numpy.testing.assert_array_equal(array(output["present"]), [1, 1])
        with self.assertRaises(TypeError):
            state.reset({"past": torch.zeros(4, dtype=torch.float32)[::2]})

    def test_rejects_inputs_requiring_copy(self):
        for initial in (
            numpy.zeros(4, dtype=numpy.float32)[::2],
            numpy.zeros(2, dtype=numpy.float32)[::-1],
            numpy.zeros(2, dtype=numpy.dtype(numpy.float32).newbyteorder("S")),
            [0.0, 0.0],
        ):
            with self.subTest(initial=initial), self.assertRaises((TypeError, ValueError)):
                runtime.FeedbackState(make_model(), {"past": initial})

    def test_literal_dots_in_whole_structured_state(self):
        tensor_type = helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, [2])

        def structure(fields):
            """Returns a named structure declaration."""
            return onnx.TypeProto(
                struct_type=onnx.StructTypeProto(
                    structure=onnx.StructTypeProto.Structure(
                        field=[
                            onnx.StructTypeProto.Structure.Field(name=name, type=field_type)
                            for name, field_type in fields
                        ]
                    )
                )
            )

        value_type = structure([("a.b", tensor_type), ("a", structure([("b", tensor_type)]))])
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node("Echo", ["x"], ["y"], domain="feedback.test")],
                "literal_fields",
                [onnx.ValueInfoProto(name="x", type=value_type)],
                [onnx.ValueInfoProto(name="y", type=value_type)],
            ),
            opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("feedback.test", 1)],
        )
        add_binding(model, "x", "y")
        verify.verify_model(model)
        first = numpy.ones(2, dtype=numpy.float32)
        second = numpy.full(2, 2, dtype=numpy.float32)
        state = runtime.FeedbackState(model, {"x": {"a.b": first, "a": {"b": second}}})
        context = make_context()

        def echo(node, context):
            """Forwards named fields without flattening their names."""
            context.put_value(str(node.output[0]), context.get_value(str(node.input[0])))

        context.register_custom_kernel("feedback.test", "Echo", echo)
        output = state.run(context, {})
        self.assertEqual(set(state.values), {"x"})
        self.assertEqual(array(output["y"]["a.b"]).ctypes.data, first.ctypes.data)
        self.assertEqual(array(output["y"]["a"]["b"]).ctypes.data, second.ctypes.data)
        state.close()
        for field, name in (("input_name", "x.a.b"), ("output_name", "y.a.b")):
            setattr(model.graph.persistent_bindings[0], field, name)
            with self.assertRaisesRegex(ValueError, "exact.*input/output"):
                verify.verify_model(model)
            with self.assertRaisesRegex(ValueError, "exact.*input/output"):
                runtime.FeedbackState(model, {"x": {"a.b": first, "a": {"b": second}}})
            setattr(
                model.graph.persistent_bindings[0], field, "x" if field == "input_name" else "y"
            )

    def test_literal_dotted_graph_name(self):
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node("Identity", ["past.part"], ["present.part"])],
                "literal_root",
                [helper.make_tensor_value_info("past.part", onnx.TensorProto.FLOAT, [2])],
                [helper.make_tensor_value_info("present.part", onnx.TensorProto.FLOAT, [2])],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
        )
        add_binding(model, "past.part", "present.part")
        verify.verify_model(model)
        initial = numpy.ones(2, dtype=numpy.float32)
        state = runtime.FeedbackState(model, {"past.part": initial})
        output = state.run(make_context(), {})
        self.assertEqual(set(state.values), {"past.part"})
        self.assertEqual(array(output["present.part"]).ctypes.data, initial.ctypes.data)
        with self.assertRaisesRegex(ValueError, "override retained input"):
            state.run(make_context(), {"past.part": initial})
        with self.assertRaisesRegex(ValueError, "missing initial whole input"):
            state.reset({r"past\.part": initial})

    def test_graph_name_prefixes_are_independent_inputs(self):
        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node("Identity", ["past"], ["present"]),
                    helper.make_node("Identity", ["past.part"], ["present.part"]),
                ],
                "literal_prefixes",
                [
                    helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [2])
                    for name in ("past", "past.part")
                ],
                [
                    helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [2])
                    for name in ("present", "present.part")
                ],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
        )
        add_binding(model, "past", "present")
        verify.verify_model(model)
        first = numpy.ones(2, dtype=numpy.float32)
        second = numpy.full(2, 2, dtype=numpy.float32)
        state = runtime.FeedbackState(model, {"past": first})
        output = state.run(make_context(), {"past.part": second})
        self.assertEqual(array(output["present.part"]).ctypes.data, second.ctypes.data)
        self.assertEqual(set(state.values), {"past"})
        with self.assertRaisesRegex(ValueError, "unknown graph input"):
            state.run(make_context(), {r"past\.part": second})
        state.close()
        add_binding(model, "past.part", "present.part")
        verify.verify_model(model)
        state = runtime.FeedbackState(model, {"past": first, "past.part": second})
        output = state.run(make_context(), {})
        self.assertEqual(set(state.values), {"past", "past.part"})
        self.assertEqual(array(output["present"]).ctypes.data, first.ctypes.data)
        self.assertEqual(array(output["present.part"]).ctypes.data, second.ctypes.data)

    def test_initializer_output_retains_model(self):
        weights = numpy.ones(2, dtype=numpy.float32)
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node("Identity", ["W"], ["present"])],
                "initializer_feedback",
                [helper.make_tensor_value_info("past", onnx.TensorProto.FLOAT, [2])],
                [helper.make_tensor_value_info("present", onnx.TensorProto.FLOAT, [2])],
                [numpy_helper.from_array(weights, name="W")],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
        )
        add_binding(model, "past", "present")
        state = runtime.FeedbackState(model, {"past": numpy.zeros(2, dtype=numpy.float32)})
        output = state.run(make_context(), {})
        self.assertEqual(
            array(output["present"]).ctypes.data, array(state.values["past"]).ctypes.data
        )
        state.close()
        del model, state
        gc.collect()
        numpy.testing.assert_array_equal(array(output["present"]), weights)


if __name__ == "__main__":
    unittest.main(verbosity=2)
