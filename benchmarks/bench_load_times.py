"""
bench_load_times.py

Standalone Python benchmark that measures the load time of a single ONNX
model across several loaders and configurations.  It takes exactly one
positional argument: the path to the ``.onnx`` model to benchmark.

The following load configurations are timed:

* ``onnx`` — the reference :mod:`onnx` loader (protobuf based), disabled by
  default (uncomment the case in ``_build_cases`` to enable it).
* ``onnxruntime (opt off)`` — :class:`onnxruntime.InferenceSession` with
  graph optimizations disabled (``ORT_DISABLE_ALL``).
* ``onnxruntime (opt on)`` — :class:`onnxruntime.InferenceSession` with all
  graph optimizations enabled (``ORT_ENABLE_ALL``).
* ``onnx_light ifstream`` — :func:`onnx_light.onnx.load` with
  ``file_load_mode="IFSTREAM"`` and ``num_threads`` = 1, 2, 4.
* ``onnx_light memmap`` — :func:`onnx_light.onnx.load` with
  ``file_load_mode="MMAP"`` and ``num_threads`` = 1, 2, 4.
* ``onnx_light reference`` — :class:`onnx_light.onnx.reference.ReferenceEvaluator`
  built from a model loaded with ``num_threads`` = 1, 2, 4, 8.

Usage::

    python benchmarks/bench_load_times.py path/to/model.onnx [-n ITERS] [-o OUT.xlsx]

The results are printed and also exported to a single ``.xlsx`` file (via
:mod:`pandas`).  Each row includes the benchmarked ``model`` path and the
``machine`` (processor name) so results from different runs can be combined.

``onnxruntime`` is optional: when it is not installed, its rows are skipped
and a warning is printed instead.
"""

import argparse
import statistics
import time

import pandas

import onnx_light.onnx as onnxl
from onnx_light.doc import get_processor_name
from onnx_light.onnx.reference import ReferenceEvaluator

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
    cases: list[tuple[str, object]] = []
    # The reference ``onnx`` loader is disabled by default; uncomment to enable it.
    # cases.append(("onnx", lambda: onnx.load(model_path)))

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

    for file_load_mode in ("IFSTREAM", "MMAP"):
        for n_threads in (1, 2, 4, 8):
            cases.append(
                (
                    f"onnx_light {file_load_mode.lower()} {n_threads} thread(s)",
                    lambda mode=file_load_mode, threads=n_threads: onnxl.load(
                        model_path,
                        num_threads=threads,
                        load_external_data=True,
                        file_load_mode=mode,
                    ),
                )
            )

    for n_threads in (1, 2, 4, 8):
        cases.append(
            (
                f"onnx_light reference {n_threads} thread(s)",
                lambda threads=n_threads: ReferenceEvaluator(
                    onnxl.load(model_path, num_threads=threads, load_external_data=True)
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
    parser.add_argument(
        "-o",
        "--output",
        default="bench_load_times.xlsx",
        help="Path to the .xlsx file to write the results to "
        "(default: bench_load_times.xlsx).",
    )
    args = parser.parse_args(argv)

    if ort is None:
        print("WARNING: onnxruntime is not installed, skipping onnxruntime cases.")

    machine = get_processor_name()
    print(f"model    = {args.model}")
    print(f"machine  = {machine}")
    print(f"n_iters  = {args.iters}\n")

    rows = []
    for label, fn in _build_cases(args.model):
        load_s = _measure(fn, args.iters)
        load_ms = load_s * 1e3
        print(f"{label:<32} {load_ms:9.3f} ms/iter")
        rows.append(
            {
                "model": args.model,
                "machine": machine,
                "case": label,
                "iters": args.iters,
                "load_ms": load_ms,
            }
        )

    df = pandas.DataFrame(rows)
    df.to_excel(args.output, index=False)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
