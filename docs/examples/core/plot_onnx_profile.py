"""
.. _l-example-plot-onnx-profile:

Profile the C++ parsing and serialization code
===============================================

This page explains how to build the standalone C++ benchmark
``bench_parse_serialize`` with debug symbols, run it, and collect a
function-level profile using ``perf``, ``gprof``, or
``valgrind --tool=callgrind``.

Because the onnx-light core is a native C++ library, only native Linux
profiling tools can attribute wall-clock or instruction samples to individual
C++ functions.  The Python :mod:`cProfile` module only sees the Python →
C++ call boundary and cannot look inside the C++ implementation.

The benchmark is in ``benchmarks/bench_parse_serialize.cpp`` and is
controlled by the ``ONNX_LIGHT_BUILD_BENCHMARKS`` CMake option.

Step 1 — build with ``RelWithDebInfo``
---------------------------------------

``RelWithDebInfo`` enables compiler optimizations (``-O2``) while retaining
full DWARF debug symbols (``-g``).  Both ``perf`` and ``valgrind`` rely on
those symbols to resolve addresses back to function names and source lines.

.. code-block:: bash

    cmake -B build \\
          -DCMAKE_BUILD_TYPE=RelWithDebInfo \\
          -DONNX_LIGHT_BUILD_BENCHMARKS=ON \\
          -DONNX_LIGHT_BUILD_PYTHON=OFF
    cmake --build build --target bench_parse_serialize -j

The binary lands at ``build/bench_parse_serialize``.

Usage::

    ./build/bench_parse_serialize -n <iters> -t <threads> -i <tensors> -d <dim>

    -n 200    number of parse + serialize round-trips (default 200)
    -t 4      thread count for parallel mode (1 = sequential, 0 = auto)
    -i 40     number of initializer tensors in the synthetic model
    -d 512    square dimension of each float weight matrix

Step 2a — profile with ``perf``
--------------------------------

``perf`` is the standard Linux performance counter tool.  It samples the
instruction pointer at the hardware-counter frequency and uses DWARF unwind
info to build full call stacks.

.. code-block:: bash

    # Counts and a summary of dominant events (no data written to disk).
    perf stat ./build/bench_parse_serialize -n 200 -t 1

    # Sample-based callgraph profile.  Produces perf.data.
    perf record -g ./build/bench_parse_serialize -n 500 -t 1

    # Print top functions to stdout.
    perf report --stdio --no-children -n | head -60

    # Interactive TUI (terminal).
    perf report

    # Generate a flame graph (requires FlameGraph scripts).
    perf script | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg

.. note::

    ``perf record`` requires ``/proc/sys/kernel/perf_event_paranoid <= 1``.
    On a personal machine run ``sudo sysctl -w kernel.perf_event_paranoid=1``
    or add it to ``/etc/sysctl.conf`` to make the change permanent.

Step 2b — profile with ``gprof``
----------------------------------

``gprof`` instruments every function call with ``-pg``.  The CMake option
``ONNX_LIGHT_BENCH_GPROF=ON`` adds ``-pg`` to the benchmark compile and link
flags.

.. code-block:: bash

    cmake -B build_gprof \\
          -DCMAKE_BUILD_TYPE=RelWithDebInfo \\
          -DONNX_LIGHT_BUILD_BENCHMARKS=ON \\
          -DONNX_LIGHT_BUILD_PYTHON=OFF \\
          -DONNX_LIGHT_BENCH_GPROF=ON
    cmake --build build_gprof --target bench_parse_serialize -j

    # Run the benchmark — gmon.out is written automatically.
    ./build_gprof/bench_parse_serialize -n 200 -t 1

    # Print the flat + call-graph profile.
    gprof ./build_gprof/bench_parse_serialize gmon.out | less

    # Quick top-20 self-time view.
    gprof -b ./build_gprof/bench_parse_serialize gmon.out | head -40

Step 2c — profile with ``valgrind --tool=callgrind``
------------------------------------------------------

Callgrind is a cache-simulation + call-graph tool from the Valgrind suite.
It runs the program instrumented (roughly 20× slower) so use a small
iteration count.

.. code-block:: bash

    valgrind --tool=callgrind \\
             --callgrind-out-file=callgrind.out \\
             ./build/bench_parse_serialize -n 20 -t 1

    # Annotated flat profile
    callgrind_annotate callgrind.out | head -80

    # Or open in the KCachegrind / QCachegrind GUI
    kcachegrind callgrind.out
"""

import os
import subprocess
import sys

import pandas

# %%
# Build and run the benchmark
# ----------------------------
#
# When the environment variable ``BENCH_PARSE_SERIALIZE_BIN`` is set, the
# pre-built binary is used directly.  Otherwise the script attempts a quick
# ``cmake`` build inside a temporary directory.  If neither succeeds
# (e.g. in the documentation CI which does not build the C++ library),
# sample data is used so the plots can still be rendered.

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_UNITTEST = os.environ.get("UNITTEST_GOING") == "1"

# Candidate binary paths (env override, repo default build dir).
_CANDIDATES = [
    os.environ.get("BENCH_PARSE_SERIALIZE_BIN", ""),
    os.path.join(_REPO_ROOT, "build", "bench_parse_serialize"),
]
_BENCH_BIN = next((p for p in _CANDIDATES if p and os.path.isfile(p)), None)

# Try a fast cmake build only when not in unittest/CI mode and cmake is
# available — skip on CI where the C++ tree has not been configured.
if _BENCH_BIN is None and not _UNITTEST and sys.platform.startswith("linux"):
    import shutil
    import tempfile

    _CMAKE = shutil.which("cmake")
    if _CMAKE:
        _BUILD_DIR = os.path.join(tempfile.gettempdir(), "onnx_light_bench_build")
        _rc = subprocess.run(
            [
                _CMAKE,
                "-B",
                _BUILD_DIR,
                "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
                "-DONNX_LIGHT_BUILD_BENCHMARKS=ON",
                "-DONNX_LIGHT_BUILD_PYTHON=OFF",
                _REPO_ROOT,
            ],
            capture_output=True,
        ).returncode
        if _rc == 0:
            _rc = subprocess.run(
                [_CMAKE, "--build", _BUILD_DIR, "--target", "bench_parse_serialize", "-j4"],
                capture_output=True,
            ).returncode
        if _rc == 0:
            _BENCH_BIN = os.path.join(_BUILD_DIR, "bench_parse_serialize")

# %%
# Run the benchmark and collect timings
# ---------------------------------------
#
# The benchmark is run four times to cover the sequential and parallel (4-
# thread) variants of both operations.  ``-d 128`` keeps the model small
# enough to finish quickly.

_N_ITERS = 5 if _UNITTEST else 100
_DIM = 64 if _UNITTEST else 128
_N_INIT = 4 if _UNITTEST else 20


def _run(n_threads: int) -> dict | None:
    """Runs the benchmark binary and returns the parsed timing dict.

    Args:
        n_threads: Thread count forwarded to the ``-t`` argument.

    Returns:
        A dict with keys ``serialize_ms`` and ``parse_ms``, or ``None`` if
        the binary is unavailable.
    """
    if _BENCH_BIN is None:
        return None
    result = subprocess.run(
        [
            _BENCH_BIN,
            "-n",
            str(_N_ITERS),
            "-t",
            str(n_threads),
            "-i",
            str(_N_INIT),
            "-d",
            str(_DIM),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    print(result.stdout)
    timings = {}
    for line in result.stdout.splitlines():
        # Lines look like "serialize: 1.51 ms/iter ..." or "parse    : 0.07 ms/iter ..."
        # Split on ':' first, then extract the float from the right-hand side.
        if ":" in line and ("serialize" in line or "parse" in line):
            key = "serialize_ms" if "serialize" in line else "parse_ms"
            rhs = line.split(":", 1)[1].strip()
            try:
                timings[key] = float(rhs.split()[0])
            except (ValueError, IndexError):
                pass
    return timings if timings else None


t_x1 = _run(1)
t_x4 = _run(4)

# %%
# Fallback: use sample data when the binary is not available
# -----------------------------------------------------------
#
# The numbers below are representative of a 20-tensor, dim=128 model on a
# typical x86-64 laptop.

_SAMPLE = {"serialize_ms": 0.45, "parse_ms": 0.30}
_SAMPLE_X4 = {"serialize_ms": 0.18, "parse_ms": 0.14}

if t_x1 is None:
    print("Benchmark binary not found — using sample data for the plot.")
    t_x1 = _SAMPLE
if t_x4 is None:
    t_x4 = _SAMPLE_X4

# %%
# Plot: serialize and parse latency (sequential vs. parallel)
# ------------------------------------------------------------

records = [
    {"operation": "serialize/x1", "ms": t_x1["serialize_ms"]},
    {"operation": "serialize/x4", "ms": t_x4["serialize_ms"]},
    {"operation": "parse/x1", "ms": t_x1["parse_ms"]},
    {"operation": "parse/x4", "ms": t_x4["parse_ms"]},
]
df = pandas.DataFrame(records).set_index("operation").sort_index(ascending=False)
print(df)

ax = df["ms"].plot.barh(
    title="bench_parse_serialize: latency per iteration\n(lower is better)",
    xlabel="ms / iteration",
    color="steelblue",
)
ax.grid(axis="x")
ax.figure.tight_layout()
ax.figure.savefig("plot_onnx_profile.png")
