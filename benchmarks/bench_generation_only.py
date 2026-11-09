"""Benchmarks explicit last-token preparation, not all-token TinyLLM inference.

Runs from the repository root with ``PYTHONPATH=. python
benchmarks/bench_generation_only.py --model model.onnx --feeds prefill.npz``.
Use ``--synthetic`` only for a labeled smoke test, never TinyLLM measurements.
"""

import argparse
import gc
import json
import os
import platform
import statistics
import time
from functools import partial

import numpy

from onnx_light import __version__, onnx
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh
from onnx_light.onnx import TensorProto
from onnx_light.onnx.reference import ReferenceEvaluator, prepare_generation_model


def clone_proto(proto):
    """Copies protobuf graph metadata through the native message API."""
    result = type(proto)()
    result.CopyFrom(proto)
    return result


def synthetic_model():
    """Returns a small deterministic projection smoke test, not a TinyLLM."""
    rng = numpy.random.default_rng(42)
    graph = oh.make_graph(
        [
            oh.make_node("Gather", ["embedding", "input_ids"], ["hidden"]),
            oh.make_node("Unsqueeze", ["hidden", "cache_axis"], ["cache_tokens"]),
            oh.make_node(
                "Concat", ["past_key_values.0.key", "cache_tokens"], ["present.0.key"], axis=2
            ),
            oh.make_node(
                "Concat", ["past_key_values.0.value", "cache_tokens"], ["present.0.value"], axis=2
            ),
            oh.make_node("MatMul", ["hidden", "weight"], ["raw_logits"]),
            oh.make_node("Cast", ["raw_logits"], ["logits"], to=TensorProto.FLOAT),
        ],
        "synthetic_projection_only",
        [oh.make_tensor_value_info("input_ids", TensorProto.INT64, [1, "sequence"])]
        + [
            oh.make_tensor_value_info(name, TensorProto.FLOAT, [1, 1, "past", 64])
            for name in ("past_key_values.0.key", "past_key_values.0.value")
        ],
        [oh.make_tensor_value_info("logits", TensorProto.FLOAT, [1, "sequence", 256])]
        + [
            oh.make_tensor_value_info(name, TensorProto.FLOAT, [1, 1, "total", 64])
            for name in ("present.0.key", "present.0.value")
        ],
        [
            onh.from_array(rng.standard_normal((256, 64)).astype(numpy.float32), "embedding"),
            onh.from_array(rng.standard_normal((64, 256)).astype(numpy.float32), "weight"),
            onh.from_array(numpy.array([1], dtype=numpy.int64), "cache_axis"),
        ],
        value_info=[
            oh.make_tensor_value_info("hidden", TensorProto.FLOAT, [1, "sequence", 64]),
            oh.make_tensor_value_info("raw_logits", TensorProto.FLOAT, [1, "sequence", 256]),
        ],
    )
    return (
        oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)], ir_version=10),
        {
            "input_ids": numpy.arange(128, dtype=numpy.int64).reshape(1, -1),
            "past_key_values.0.key": numpy.empty((1, 1, 0, 64), dtype=numpy.float32),
            "past_key_values.0.value": numpy.empty((1, 1, 0, 64), dtype=numpy.float32),
        },
    )


def load_artifacts(model_path, feeds_path):
    """Loads an exported model and explicit, pickle-free prefill inputs."""
    model = onnx.load(model_path)
    with numpy.load(feeds_path, allow_pickle=False) as archive:
        feeds = {name: archive[name] for name in archive.files}
    expected = {value.name for value in model.graph.input}
    if set(feeds) != expected:
        raise ValueError(
            f"Feed names differ: missing={expected - set(feeds)}, extra={set(feeds) - expected}"
        )
    ids = feeds.get("input_ids")
    if ids is None or ids.dtype != numpy.int64 or ids.ndim != 2 or ids.shape[1] < 2:
        raise ValueError("input_ids must be INT64 [batch, sequence] with sequence >= 2")
    for name, value in feeds.items():
        if name.startswith("past_key_values.") and (value.ndim < 3 or value.shape[-2] != 0):
            raise ValueError(f"{name!r} must contain an explicit empty prefill cache")
    return model, feeds


def native_session(model, threads):
    """Returns a CPU-only native evaluator with an explicit worker count."""
    return ReferenceEvaluator(
        model, cpu_execution={"num_threads": threads, "affinity_policy": "none"}
    )


def make_session(model, backend, threads):
    """Returns one runtime's session without constructing the other runtime."""
    if backend == "native":
        return native_session(model, threads)
    import onnxruntime

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return onnxruntime.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def head_cases(model, feeds, threads):
    """Extracts the actual terminal projection and its untimed hidden states."""
    producers = {output: node for node in model.graph.node for output in node.output}
    terminal = producers.get("logits")
    trailing = []
    if terminal is not None and terminal.op_type == "Cast" and not terminal.domain:
        trailing = [terminal]
        terminal = producers.get(terminal.input[0])
    if terminal is None or terminal.op_type != "MatMul" or terminal.domain:
        raise ValueError(
            "Expected terminal MatMul, optionally followed by Cast, producing logits"
        )
    probe = clone_proto(model)
    infos = {
        value.name: value
        for value in (*model.graph.input, *model.graph.value_info, *model.graph.output)
    }
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    logits = next(output for output in model.graph.output if output.name == "logits")
    head_dtype = (
        initializers[terminal.input[1]].data_type
        if terminal.input[1] in initializers
        else logits.type.tensor_type.elem_type
    )
    for name in terminal.input:
        if name in infos:
            info = clone_proto(infos[name])
        elif name in initializers:
            tensor = initializers[name]
            info = oh.make_tensor_value_info(name, tensor.data_type, list(tensor.dims))
        else:
            info = oh.make_tensor_value_info(name, head_dtype, None)
        probe.graph.output.append(info)
    evaluator = native_session(probe, threads)
    hidden, weight = evaluator.run(list(terminal.input), feeds)
    hidden, weight = hidden.copy(), weight.copy()
    del evaluator, probe
    if hidden.ndim != 3 or weight.ndim != 2:
        raise ValueError("LM head benchmark requires rank-3 hidden states and rank-2 weights")
    head_weight = onh.from_array(weight, terminal.input[1])
    cases = {}
    for variant, values in (
        ("full", hidden),
        ("reduced", numpy.ascontiguousarray(hidden[:, -1:, :])),
    ):
        graph = oh.make_graph(
            [clone_proto(terminal), *[clone_proto(node) for node in trailing]],
            "isolated_lm_head",
            [
                oh.make_tensor_value_info(
                    terminal.input[0], oh.np_dtype_to_tensor_dtype(values.dtype), values.shape
                )
            ],
            [clone_proto(logits)],
            [head_weight],
        )
        graph.output[0].type.tensor_type.shape.dim[1].dim_value = values.shape[1]
        head = oh.make_model(
            graph, opset_imports=list(model.opset_import), ir_version=model.ir_version
        )
        cases[variant] = (head, {terminal.input[0]: values})
    return cases


def generation_model(model):
    """Adds standard exported cache feedback only to a generation-local copy."""
    result = clone_proto(model)
    outputs = {value.name for value in result.graph.output}
    bindings = {binding.input_name for binding in result.graph.persistent_bindings}
    for value in result.graph.input:
        if not value.name.startswith("past_key_values.") or value.name in bindings:
            continue
        suffix = value.name.removeprefix("past_key_values.")
        candidates = [
            name
            for name in (f"present.{suffix}", f"present_key_values.{suffix}")
            if name in outputs
        ]
        if len(candidates) != 1:
            raise ValueError(f"Cannot identify unique present cache for {value.name!r}")
        binding = result.graph.persistent_bindings.add()
        binding.input_name = value.name
        binding.output_name = candidates[0]
    return result


def validate_generation(models, feeds, threads, tokens):
    """Checks native prefill/decode parity and fresh state across repeated calls."""
    expected = None
    for variant in ("full", "reduced"):
        evaluator = native_session(generation_model(models[variant]), threads)
        for count in (1, tokens):
            first = evaluator.generate(feeds, max_new_tokens=count, temperature=0.0)
            second = evaluator.generate(feeds, max_new_tokens=count, temperature=0.0)
            numpy.testing.assert_array_equal(first, second)
            if variant == "full":
                if expected is None:
                    expected = {}
                expected[count] = first.copy()
            else:
                numpy.testing.assert_array_equal(first, expected[count])
        del evaluator


def prepare_cases(model, feeds, threads, tokens):
    """Prepares full/reduced graphs and validates native generation outside timing."""
    reduced = clone_proto(model)
    prepare_generation_model(reduced, logits_name="logits")
    models = {"full": model, "reduced": reduced}
    heads = head_cases(model, feeds, threads)
    validate_generation(models, feeds, threads, tokens)
    return models, heads


def measure(call, repeat):
    """Returns warmed raw latency samples bounded by two cumulative seconds."""
    call()
    samples = []
    while len(samples) < repeat and sum(samples) < 2:
        started = time.perf_counter()
        call()
        samples.append(time.perf_counter() - started)
    return {"median_seconds": statistics.median(samples), "samples_seconds": samples}


def check_outputs(full, reduced, names):
    """Checks all outputs, reducing only the logits sequence axis."""
    for name, expected, actual in zip(names, full, reduced):
        if name == "logits":
            if expected.ndim != 3 or actual.shape != (expected.shape[0], 1, expected.shape[2]):
                raise AssertionError(
                    f"Unexpected full/reduced logits shapes: {expected.shape}, {actual.shape}"
                )
            expected = expected[:, -1:, :]
        numpy.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


def run_benchmark(model, feeds, threads=1, repeat=5, tokens=4):
    """Measures separate native/ORT phases on identical full and reduced models."""
    models, heads = prepare_cases(model, feeds, threads, tokens)
    names = [output.name for output in model.graph.output]
    results = {}
    reference = {}
    for backend in ("native", "ort"):
        backend_outputs = {}
        for variant in ("full", "reduced"):
            session = make_session(models[variant], backend, threads)
            outputs = [array.copy() for array in session.run(None, feeds)]
            backend_outputs[variant] = outputs
            if backend == "native":
                reference[variant] = outputs
            else:
                for actual, expected in zip(outputs, reference[variant]):
                    numpy.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
            prefill = measure(partial(session.run, None, feeds), repeat)
            del session
            head, head_feeds = heads[variant]
            session = make_session(head, backend, threads)
            projected = session.run(None, head_feeds)[0]
            numpy.testing.assert_allclose(
                projected, outputs[names.index("logits")], rtol=1e-4, atol=1e-5
            )
            projection = measure(partial(session.run, None, head_feeds), repeat)
            del session
            results[f"{backend}_{variant}"] = {
                "prefill": prefill,
                "isolated_lm_head_projection": projection,
                "logits_bytes": outputs[names.index("logits")].nbytes,
            }
        check_outputs(backend_outputs["full"], backend_outputs["reduced"], names)
        gc.collect()
    return results


class TinyLLMGenerationOnly:
    """Benchmarks supplied TinyLLM artifacts without altering upstream defaults."""

    params = (["native", "ort"], ["full", "reduced"])
    param_names = ["backend", "variant"]
    timeout = 300

    def setup(self, backend, variant):
        """Loads explicitly supplied artifacts and prepares only the selected runtime."""
        model_path = os.environ.get("ONNX_LIGHT_TINYLLM_MODEL")
        feeds_path = os.environ.get("ONNX_LIGHT_TINYLLM_FEEDS")
        if not model_path or not feeds_path:
            raise NotImplementedError("Set ONNX_LIGHT_TINYLLM_MODEL and ONNX_LIGHT_TINYLLM_FEEDS")
        threads = int(os.environ.get("ONNX_LIGHT_TINYLLM_THREADS", "1"))
        if threads < 1:
            raise ValueError("ONNX_LIGHT_TINYLLM_THREADS must be positive")
        model, self.feeds = load_artifacts(model_path, feeds_path)
        models, heads = prepare_cases(model, self.feeds, threads, 4)
        gc.collect()
        self.prefill = make_session(models[variant], backend, threads)
        head, self.head_feeds = heads[variant]
        self.head = make_session(head, backend, threads)
        self.logits_bytes = self.prefill.run(["logits"], self.feeds)[0].nbytes
        self.head.run(None, self.head_feeds)

    def time_prefill(self, backend, variant):
        """Measures one already-prepared model prefill."""
        self.prefill.run(None, self.feeds)

    def time_lm_head_projection(self, backend, variant):
        """Measures the isolated MatMul and optional output Cast."""
        self.head.run(None, self.head_feeds)

    def track_logits_bytes(self, backend, variant):
        """Reports output tensor bytes, not allocator or process peak memory."""
        return self.logits_bytes

    track_logits_bytes.unit = "bytes"

    def teardown(self, backend, variant):
        """Releases workers before another runtime's benchmark starts."""
        del self.prefill, self.head
        gc.collect()


def main():
    """Runs the separately named generation-only benchmark and emits JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--feeds", help="NPZ containing every input, including empty caches")
    parser.add_argument("--synthetic", action="store_true", help="Runs a smoke test, not TinyLLM")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--tokens", type=int, default=4)
    args = parser.parse_args()
    if min(args.threads, args.repeat) < 1 or args.tokens < 2:
        parser.error("--threads/--repeat must be positive and --tokens must be >= 2")
    if args.synthetic:
        if args.model or args.feeds:
            parser.error("--synthetic cannot be combined with --model/--feeds")
        model, feeds = synthetic_model()
    else:
        if not args.model or not args.feeds:
            parser.error("Supply both --model and --feeds, or explicitly select --synthetic")
        model, feeds = load_artifacts(args.model, args.feeds)
    import onnxruntime

    report = {
        "benchmark": "TinyLLMGenerationOnly",
        "artifact": (
            "SYNTHETIC_PROJECTION_SMOKE_TEST_NOT_TINYLLM" if args.synthetic else args.model
        ),
        "onnx_light": __version__,
        "onnxruntime": onnxruntime.__version__,
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "threads": args.threads,
        "input_shapes": {name: list(value.shape) for name, value in feeds.items()},
        "native_greedy_parity": {
            "prefill_tokens": 1,
            "prefill_and_decode_tokens": args.tokens,
            "repeated_calls": 2,
        },
        "results": run_benchmark(model, feeds, args.threads, args.repeat, args.tokens),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
