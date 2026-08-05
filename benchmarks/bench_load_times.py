"""
bench_load_times.py

Standalone Python benchmark that measures the load time of a single ONNX
model across several loaders and configurations.  It takes exactly one
positional argument: the path to the ``.onnx`` model to benchmark.

The following load configurations are timed:

* ``onnx`` — the reference :mod:`onnx` loader (protobuf based).
* ``onnxruntime (opt off)`` — :class:`onnxruntime.InferenceSession` with
  graph optimizations disabled (``ORT_DISABLE_ALL``).
* ``onnxruntime (opt on)`` — :class:`onnxruntime.InferenceSession` with all
  graph optimizations enabled (``ORT_ENABLE_ALL``).
* ``onnx_light 1 thread`` — :func:`onnx_light.onnx.load` with ``num_threads=1``.
* ``onnx_light 2 threads`` — :func:`onnx_light.onnx.load` with ``num_threads=2``.
* ``onnx_light 4 threads`` — :func:`onnx_light.onnx.load` with ``num_threads=4``.

Usage::

    python benchmarks/bench_load_times.py path/to/model.onnx [-n ITERS]

``onnxruntime`` is optional: when it is not installed, its rows are skipped
and a warning is printed instead.
"""

import argparse
import statistics
import time

import onnx
import onnx_light.onnx as onnxl

try:
    import onnxruntime as ort
except ImportError:
    ort = None


def _measure(fn, n_iters: int) -> float:
    """Runs *fn* *n_iters* times and returns the median wall-clock time (seconds)."""
    timings = []
    for _ in range(n_iters):
        begin = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - begin)
    return statistics.median(timings)


def _build_cases(model_path: str) -> list[tuple[str, object]]:
    """Returns the list of ``(label, callable)`` load cases for *model_path*."""
    cases: list[tuple[str, object]] = []  # ("onnx", lambda: onnx.load(model_path))]

    if ort is not None:
        opt_off = ort.SessionOptions()
        opt_off.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        opt_on = ort.SessionOptions()
        opt_on.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        cases.append(
            (
                "onnxruntime (opt off)",
                lambda: ort.InferenceSession(
                    model_path, sess_options=opt_off, providers=["CPUExecutionProvider"]
                ),
            )
        )
        cases.append(
            (
                "onnxruntime (opt on)",
                lambda: ort.InferenceSession(
                    model_path, sess_options=opt_on, providers=["CPUExecutionProvider"]
                ),
            )
        )

    for n_threads in (1, 2, 4, 8):
        cases.append(
            (
                f"onnx_light {n_threads} thread(s)",
                lambda threads=n_threads: onnxl.load(
                    model_path, num_threads=threads, load_external_data=True
                ),
            )
        )

    return cases


def main(argv: list[str] | None = None) -> None:
    """Parses arguments and runs the load-time benchmark for a single model path."""
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("model", help="Path to the .onnx model to benchmark.")
    parser.add_argument(
        "-n",
        "--iters",
        type=int,
        default=10,
        help="Number of load iterations per case (default: 10).",
    )
    args = parser.parse_args(argv)

    if ort is None:
        print("WARNING: onnxruntime is not installed, skipping onnxruntime cases.")

    print(f"model    = {args.model}")
    print(f"n_iters  = {args.iters}\n")

    for label, fn in _build_cases(args.model):
        load_s = _measure(fn, args.iters)
        print(f"{label:<24} {load_s * 1e3:9.3f} ms/iter")


if __name__ == "__main__":
    main()
