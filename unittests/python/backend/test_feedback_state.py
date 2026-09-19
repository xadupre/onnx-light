# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests persistent model feedback through the native bindings."""

import gc
import subprocess
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy

from onnx_light import onnx
from onnx_light.ext_test_case import import_or_skip
from onnx_light.onnx import helper, numpy_helper
from onnx_light.onnx_lib import parser

runtime = import_or_skip("onnx_light.onnx_py._onnxpykernels", "runtime")


def make_model():
    """Returns a stateless accumulator."""
    return parser.parse_model(
        '<ir_version: 10, opset_import: ["" : 18]>'
        "accumulate (float[2] delta, float[2] past) => (float[2] present) "
        "{ present = Add(delta, past) }"
    )


def make_context():
    """Returns an independent runtime context."""
    return runtime.RuntimeContext(runtime.KernelContext(runtime.default_opset(18)))


def array(tensor):
    """Returns the float values of an owned runtime tensor."""
    return numpy.frombuffer(tensor.raw_data(), dtype=numpy.float32).reshape(tensor.shape)


class TestFeedbackState(unittest.TestCase):
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
        state = runtime.FeedbackState(model, {"past": "present"}, {"past": initial})
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
        first = runtime.FeedbackState(model, {"past": "present"}, {"past": initial})
        second = runtime.FeedbackState(model, {"past": "present"}, {"past": initial})
        initial[:] = 100
        del model
        gc.collect()
        output = first.run(make_context(), {"delta": numpy.ones(2, dtype=numpy.float32)})
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
        with self.assertRaises(ValueError):
            runtime.FeedbackState(model, {"absent": "present"}, {"absent": initial})
        with self.assertRaises(ValueError):
            runtime.FeedbackState(model, {"past": "present"}, {})
        with self.assertRaises(ValueError):
            runtime.FeedbackState(
                model, {"past": "present"}, {"past": numpy.zeros(3, dtype=numpy.float32)}
            )
        state = runtime.FeedbackState(model, {"past": "present"}, {"past": initial})
        with self.assertRaises(ValueError):
            state.run(make_context(), {})
        with self.assertRaises(ValueError):
            state.run(make_context(), {"past": initial, "delta": initial})
        model.graph.output[0].name = "rewritten"
        with self.assertRaises(ValueError):
            state.run(make_context(), {"delta": initial})

    def test_cancelled_call_preserves_state(self):
        state = runtime.FeedbackState(
            make_model(), {"past": "present"}, {"past": numpy.zeros(2, dtype=numpy.float32)}
        )
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
        for global_registration in (False, True):
            with self.subTest(global_registration=global_registration):
                initial = numpy.zeros(2, dtype=numpy.float32)
                state = runtime.FeedbackState(model, {"past": "present"}, {"past": initial})
                context = make_context()
                entered = threading.Event()
                resume = threading.Event()
                completion = runtime.TaskCompletion()

                def block(node, context):
                    """Waits for cancellation before producing an output."""
                    entered.set()
                    if not resume.wait(10):
                        raise RuntimeError("test timed out waiting for cancellation")
                    context.put_value(node.output[0], numpy.ones(2, dtype=numpy.float32))

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

    def test_structured_field_feedback_and_kernel_failure(self):
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
                [helper.make_node("Step", ["request"], ["response"], domain="feedback.test")],
                "structured_feedback",
                [onnx.ValueInfoProto(name="request", type=struct_type("tokens", "cache"))],
                [onnx.ValueInfoProto(name="response", type=struct_type("logits", "cache"))],
            ),
            opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("feedback.test", 1)],
        )
        state = runtime.FeedbackState(
            model,
            {"request.cache": "response.cache"},
            {"request.cache": numpy.zeros(2, dtype=numpy.float32)},
        )
        context = make_context()
        fail = False

        def step(node, context):
            """Produces structured outputs through the ordinary custom-kernel API."""
            request = context.get_value(node.input[0])
            cache = array(request["cache"]) + array(request["tokens"])
            context.put_value(node.output[0], {"cache": cache, "logits": cache * 2})
            if fail:
                raise RuntimeError("failed after producing outputs")

        context.register_custom_kernel("feedback.test", "Step", step)
        for expected in (1, 2):
            output = state.run(context, {"request.tokens": numpy.ones(2, dtype=numpy.float32)})
            numpy.testing.assert_array_equal(array(output["response"]["cache"]), [expected] * 2)
        self.assertEqual(set(state.values), {"request.cache"})
        fail = True
        with self.assertRaisesRegex(RuntimeError, "failed after producing outputs"):
            state.run(context, {"request.tokens": numpy.ones(2, dtype=numpy.float32)})
        numpy.testing.assert_array_equal(array(state.values["request.cache"]), [2, 2])
        fail = False
        output = state.run(context, {"request.tokens": numpy.ones(2, dtype=numpy.float32)})
        numpy.testing.assert_array_equal(array(output["response"]["cache"]), [3, 3])


if __name__ == "__main__":
    unittest.main(verbosity=2)
