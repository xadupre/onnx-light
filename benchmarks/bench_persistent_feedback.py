"""Measures graph-declared Attention feedback against explicit stateless feedback.

Run ``git fetch origin main`` before comparing measurements across revisions.
Run ``PYTHONPATH=. python benchmarks/bench_persistent_feedback.py --tokens 16`` from the
repository root. The JSON contains raw per-token times and kernel storage
events; Python and session setup are timed separately from decode calls.
"""

import argparse
import json
import platform
import subprocess
import time

import numpy

import onnx_light.onnx.helper as oh
from onnx_light.onnx import TensorProto
from onnx_light.onnx_py._onnxpykernels import runtime


def make_model():
    """Returns an Attention model with graph-declared key/value feedback."""
    graph = oh.make_graph(
        [
            oh.make_node(
                "Attention",
                ["Q", "K", "V", "", "past_key", "past_value"],
                ["Y", "present_key", "present_value"],
                is_causal=1,
            )
        ],
        "contiguous_decode",
        [
            oh.make_tensor_value_info(name, TensorProto.FLOAT, [1, 1, 1, 2])
            for name in ("Q", "K", "V")
        ]
        + [
            oh.make_tensor_value_info(name, TensorProto.FLOAT, [1, 1, None, 2])
            for name in ("past_key", "past_value")
        ],
        [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 1, 2])]
        + [
            oh.make_tensor_value_info(name, TensorProto.FLOAT, [1, 1, None, 2])
            for name in ("present_key", "present_value")
        ],
    )
    for past, present in (("past_key", "present_key"), ("past_value", "present_value")):
        binding = graph.persistent_bindings.add()
        binding.input_name = past
        binding.output_name = present
    return oh.make_model(graph, opset_imports=[oh.make_opsetid("", 23)], ir_version=10)


def main():
    """Measures decode latency and reports separate kernel storage costs."""
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--tokens", type=int, default=16)
    args = args.parse_args()
    if args.tokens < 1:
        args.error("--tokens must be positive")

    model = make_model()
    bindings = {
        binding.input_name: binding.output_name for binding in model.graph.persistent_bindings
    }
    initial = {name: numpy.empty((1, 1, 0, 2), dtype=numpy.float32) for name in bindings}
    options = runtime.RuntimeSessionOptions(persistent_tensor_initial_capacity=args.tokens)
    started = time.perf_counter_ns()
    state = runtime.PersistentValueState(model, initial, options)
    state_setup_ns = time.perf_counter_ns() - started
    started = time.perf_counter_ns()
    stateless = runtime.RuntimeSession(model)
    stateless_setup_ns = time.perf_counter_ns() - started
    context = runtime.RuntimeContext(
        runtime.KernelContext(runtime.default_opset(23)), events_enabled=True
    )
    manual = initial.copy()
    records = []
    for index in range(args.tokens):
        token = numpy.array([index / 8, index / 4], dtype=numpy.float32).reshape(1, 1, 1, 2)
        feeds = {"Q": token, "K": token, "V": -token}
        manual_context = runtime.RuntimeContext(runtime.KernelContext(runtime.default_opset(23)))
        for name, value in {**manual, **feeds}.items():
            manual_context.put_value(name, value)
        started = time.perf_counter_ns()
        stateless.run(manual_context)
        stateless_ns = time.perf_counter_ns() - started
        context.clear_events()
        started = time.perf_counter_ns()
        outputs = state.run(context, feeds)
        stateful_ns = time.perf_counter_ns() - started
        storage = [
            event
            for event in context.events()
            if event.action == runtime.RuntimeEventAction.kPersistentStorage
        ]
        for name, source in bindings.items():
            expected = manual_context.get_value(source)
            actual = numpy.from_dlpack(outputs[source])
            numpy.testing.assert_array_equal(actual, numpy.from_dlpack(expected))
            assert actual.ctypes.data == numpy.from_dlpack(state.values[name]).ctypes.data
            manual[name] = expected
        numpy.testing.assert_allclose(
            numpy.from_dlpack(outputs["Y"]),
            numpy.from_dlpack(manual_context.get_value("Y")),
            rtol=1e-6,
            atol=1e-6,
        )
        records.append(
            {
                "token": index + 1,
                "stateful_ns": stateful_ns,
                "stateless_ns": stateless_ns,
                "retained_logical_bytes": sum(
                    numpy.from_dlpack(value).nbytes for value in state.values.values()
                ),
                "event_workspace_peak_bytes": max(
                    (event.peak_bytes for event in context.events()), default=0
                ),
                "kernel_storage_allocations": sum(event.storage_allocations for event in storage),
                "kernel_storage_allocated_bytes": sum(
                    event.storage_allocated_bytes for event in storage
                ),
                "kernel_prefix_copied_bytes": sum(
                    event.storage_prefix_copied_bytes for event in storage
                ),
                "kernel_append_copied_bytes": sum(
                    event.storage_append_copied_bytes for event in storage
                ),
                "kernel_reuse_count": sum(event.storage_reuse_count for event in storage),
            }
        )
        del actual, expected, outputs
    state.close()
    print(
        json.dumps(
            {
                "model": "synthetic single-head FLOAT Attention, head width 2, opset 23",
                "policy": "CPU, one token/call, no held cache views, fixed capacity",
                "python": platform.python_version(),
                "revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "target_revision": subprocess.check_output(
                    ["git", "rev-parse", "origin/main"], text=True
                ).strip(),
                "state_setup_ns": state_setup_ns,
                "stateless_setup_ns": stateless_setup_ns,
                "measurements": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
