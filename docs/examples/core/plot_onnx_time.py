"""
.. _l-example-plot-onnx-time:

Measures loading and saving time for an ONNX model
====================================================

This script builds a small ONNX model and benchmarks the time to load
and save it using :mod:`onnx`, :mod:`onnx_light.onnx`, and
:mod:`onnxruntime`.
When the standalone C++ example executables ``load_onnx_time``,
``load_onnx_light_time``, and
``save_onnx_light_time`` are available, it also includes their timing output.
The model structure is identical in all cases.

The ``onnx_light.onnx`` implementation does not depend on protobuf and
therefore avoids the overhead of the protobuf serialization layer.
It also supports parallel loading of tensor weights through the
``parallel`` keyword and loading models stored with external data.

File loading in ``onnx_light.onnx`` uses **memory-mapped I/O** (``mmap``
on POSIX, ``CreateFileMapping`` on Windows).  The file is mapped directly
into the virtual address space so that the OS page cache is exposed as
contiguous memory; no extra system-call-per-byte buffering is required.
This makes loading from a file nearly as fast as parsing from an
already-in-memory bytes object.

One key advantage over the ``onnx`` package is zero-copy parsing:
when ``no_copy=True`` is passed to :func:`onnx_light.onnx.load` (or via
:class:`~onnx_light.onnx.ParseOptions`), tensor ``raw_data`` blobs are
**not** copied into new buffers.  Instead each ``TensorProto`` stores a
direct pointer into the serialized bytes.  This eliminates one
``malloc + memcpy`` per tensor initializer and is therefore especially
beneficial for models with many large weight tensors.

.. warning::
   When ``no_copy=True`` is used the caller must keep the original bytes
   object alive for as long as the parsed model is in use.  This
   constraint does not apply to the ``onnx`` package.

For ``onnxruntime``, the session is created with all graph optimizations
disabled (``ORT_DISABLE_ALL``) so that the measurement reflects only the
model loading overhead rather than compilation or fusion costs.

* ``onnx``, ``onnxlight``, ``ort``: use ``onnx``, ``onnx-light``, or ``onnxruntime``
* ``1filex1``: saves in a single file with 1 thread
* ``1filex4``: saves in a single file with 4 threads
* ``2filex1``: saves in a file and another for external data with 1 thread
* ``2filex4``: saves in a file and another for external data with 4 threads

Selectable benchmark scenarios (via ``--scenario``):
``load``, ``save``, ``serialize``, ``parse``, ``cpp``, ``all``.
The ``cpp`` scenario runs the standalone C++ timing executables
(``load_onnx_time``, ``load_onnx_light_time``, ``save_onnx_light_time``)
when they are available. The executable discovery automatically skips
them when the ``CI`` environment variable is set, so no results are
produced in CI environments where the executables have not been built.
"""

import argparse
import os
import pathlib
import re
import shutil
import tempfile
import time

import numpy as np
import pandas
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh

import onnxruntime as ort

_ort_sess_opts = ort.SessionOptions()
_ort_sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

import onnx_light.onnx as onnxl
from onnx_light.doc import find_standalone_executable, measure_cpp_with_example

# %%
# Setup
# -----
#
# Build a small synthetic ONNX model with several ``Gemm`` nodes and large
# initializers so that the load/save times are measurable.

N_INIT = 40
DIM = 256 if os.environ.get("UNITTEST_GOING") == "1" else 2048
BENCHMARK_SCENARIOS = ("load", "save", "serialize", "parse", "cpp")


def _parse_benchmark_scenarios(args=None) -> set[str]:
    """Parses command-line arguments and returns the selected benchmark scenarios."""
    parser = argparse.ArgumentParser(
        description="Runs one or several benchmark scenarios for plot_onnx_time.py."
    )
    parser.add_argument(
        "--scenario",
        dest="scenarios",
        action="append",
        choices=(*BENCHMARK_SCENARIOS, "all"),
        help=(
            "Scenario to execute. May be specified multiple times. "
            "Supported values: load, save, serialize, parse, cpp, all."
        ),
    )
    parsed, _ = parser.parse_known_args(args=args)
    values = parsed.scenarios or ["all"]
    if "all" in values:
        return set(BENCHMARK_SCENARIOS)
    return set(values)


SELECTED_SCENARIOS = _parse_benchmark_scenarios()


def _run_scenario(name: str) -> bool:
    """Checks whether the given scenario name is selected for execution."""
    return name in SELECTED_SCENARIOS


def make_model(n_init: int = N_INIT, dim: int = DIM) -> onnx.ModelProto:
    """Returns a synthetic ONNX model with *n_init* Gemm initializers of size *dim*."""
    initializers = []
    nodes = []
    inputs = [oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [None, dim])]

    prev = "X"
    for i in range(n_init):
        weight_name = f"W{i}"
        out_name = f"Y{i}"
        w = np.random.randn(dim, dim).astype(np.float32)
        initializers.append(onh.from_array(w, name=weight_name))
        nodes.append(oh.make_node("Gemm", [prev, weight_name], [out_name], transB=1))
        prev = out_name

    outputs = [oh.make_tensor_value_info(prev, onnx.TensorProto.FLOAT, [None, dim])]
    graph = oh.make_graph(nodes, "bench_graph", inputs, outputs, initializer=initializers)
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)], ir_version=9)
    return model


model = make_model()
size_bytes = model.ByteSize()
print(f"Model size: {size_bytes / 2 ** 20:.3f} MB")

# %%
# Write the model to a temporary file.

tmp_dir = "temp_plot_onnx_time"
if not os.path.exists(tmp_dir):
    os.mkdir(tmp_dir)
onnx_path = os.path.join(tmp_dir, "bench.onnx")
onnx.save(model, onnx_path)
file_size = os.path.getsize(onnx_path)
print(f"File size : {file_size / 2 ** 20:.3f} MB")

onx = onnx.load(onnx_path)
onxl = onnxl.load(onnx_path)
onxl_x4 = onnxl.load(onnx_path, parallel=True, num_threads=4)

ext_load_onnx = os.path.abspath(os.path.join(tmp_dir, "ext_load.onnx"))
ext_load_data = os.path.abspath(os.path.join(tmp_dir, "ext_load.onnx.data"))
onnxl.save(onxl, ext_load_onnx, location=ext_load_data)


# %%
# Benchmark helper.

MIN_TIME_THRESHOLD = 1e-9
CPP_LOAD_METRIC_PATTERN = re.compile(
    r"^\s*(Average|Median|Min|Max|Std|Standard deviation) load \(ms\)\s*:\s*([0-9.eE+-]+)\s*$"
)
CPP_SAVE_METRIC_PATTERN = re.compile(
    r"^\s*(Average|Median|Min|Max|Std|Standard deviation) save \(ms\)\s*:\s*([0-9.eE+-]+)\s*$"
)
WINDOWS_BUILD_CONFIGS = ("Release", "RelWithDebInfo", "Debug", "MinSizeRel")


def measure(name: str, fn, n: int = 5, warmup: int = 1) -> dict:
    """
    Executes *fn* with warm-up iterations and records timing statistics.

    Args:
        name: Benchmark name.
        fn: Callable to execute.
        n: Number of measured iterations.
        warmup: Number of non-measured warm-up iterations.

    Returns:
        A dictionary containing name, median, avg, min, max, and std.
    """
    for _ in range(max(0, warmup)):
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    arr = np.array(times)
    return {
        "name": name,
        "median": float(np.median(arr)),
        "avg": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }


def _flush_file(path: str) -> None:
    """Flushes one file descriptor so benchmark timing includes write-back."""
    with open(path, "r+b") as stream:
        stream.flush()
        os.fsync(stream.fileno())


def print_stats(name: str, stats: dict) -> None:
    """Prints timing statistics (average, median, max, and standard deviation) in milliseconds."""
    print(
        f"{name:<35} avg={stats['avg'] * 1e3:.1f} ms"
        f" median={stats['median'] * 1e3:.1f} ms"
        f" max={stats['max'] * 1e3:.1f} ms"
        f" std={stats['std'] * 1e3:.1f} ms"
    )


def _find_load_onnx_time_executable() -> str | None:
    """Locates the standalone C++ timing executable.

    Returns:
        The path to ``load_onnx_time`` if available, otherwise ``None``.
    """
    return find_standalone_executable(
        "load_onnx_time",
        [
            pathlib.Path("build/load-onnx-time-example/load_onnx_time"),
            pathlib.Path("build/examples/load_onnx_time/load_onnx_time"),
            pathlib.Path("build-load-onnx-time/load_onnx_time"),
        ],
        script_file=globals().get("__file__"),
        windows_build_configs=WINDOWS_BUILD_CONFIGS,
    )


def _find_load_onnx_light_time_executable() -> str | None:
    """Locates the standalone ``load_onnx_light_time`` executable.

    Returns:
        The path to ``load_onnx_light_time`` if available, otherwise ``None``.
    """
    return find_standalone_executable(
        "load_onnx_light_time",
        [
            pathlib.Path("build/load-onnx-light-time-example/load_onnx_light_time"),
            pathlib.Path("build/examples/load_onnx_light_time/load_onnx_light_time"),
            pathlib.Path("build-load-onnx-light-time/load_onnx_light_time"),
        ],
        script_file=globals().get("__file__"),
        windows_build_configs=WINDOWS_BUILD_CONFIGS,
    )


def _measure_cpp_load_with_example(
    onnx_file: str,
    n: int = 5,
    num_threads: int = 1,
    executable_name: str = "load_onnx_light_time",
) -> dict | None:
    """Measures C++ loading performance through a standalone executable.

    Args:
        onnx_file: Model path to pass to the standalone executable.
        n: Number of iterations to pass to the standalone executable.
        num_threads: Number of loading threads to pass to the standalone executable.
        executable_name: Executable selector to use:
            ``"load_onnx_time"`` or ``"load_onnx_light_time"``.

    Returns:
        A benchmark dictionary matching :func:`measure` output keys if successful,
        otherwise ``None``.
    """
    if executable_name == "load_onnx_time":
        executable = _find_load_onnx_time_executable()
        result_name = f"load/1filex{num_threads}/onnx-cpp"
    elif executable_name == "load_onnx_light_time":
        executable = _find_load_onnx_light_time_executable()
        result_name = f"load/1filex{num_threads}/onnxlight-cpp"
    else:
        raise ValueError(
            "executable_name must be 'load_onnx_time' or "
            f"'load_onnx_light_time', got {executable_name!r}"
        )
    return measure_cpp_with_example(
        executable=executable,
        args=[onnx_file, str(n), str(num_threads)],
        metric_pattern=CPP_LOAD_METRIC_PATTERN,
        result_name=result_name,
        executable_name=executable_name,
    )


def _find_save_onnx_light_time_executable() -> str | None:
    """Locates the standalone C++ save-timing executable.

    Returns:
        The path to ``save_onnx_light_time`` if available, otherwise ``None``.
    """
    return find_standalone_executable(
        "save_onnx_light_time",
        [
            pathlib.Path("build/save-onnx-light-time-example/save_onnx_light_time"),
            pathlib.Path("build/examples/save_onnx_light_time/save_onnx_light_time"),
            pathlib.Path("build-save-onnx-light-time/save_onnx_light_time"),
        ],
        script_file=globals().get("__file__"),
        windows_build_configs=WINDOWS_BUILD_CONFIGS,
    )


def _measure_cpp_save_with_example(
    onnx_file: str, n: int = 5, num_threads: int = 1
) -> dict | None:
    """Measures C++ one-file save performance through ``save_onnx_light_time``.

    Returns:
        A benchmark dictionary matching :func:`measure` output keys if successful,
        otherwise ``None``.
    """
    executable = _find_save_onnx_light_time_executable()
    if executable is None:
        return None
    with tempfile.TemporaryDirectory() as tmp_save_dir:
        return measure_cpp_with_example(
            executable=executable,
            args=[onnx_file, tmp_save_dir, str(n), str(num_threads), "onefile"],
            metric_pattern=CPP_SAVE_METRIC_PATTERN,
            result_name=f"save/1filex{num_threads}/onnxlight-cpp",
            executable_name="save_onnx_light_time",
        )


# Load scenarios
# --------------

data = []
if _run_scenario("load"):
    # %%
    # Load with onnx.

    data.append(measure("load/1filex1/onnx", lambda: onnx.load(onnx_path)))
    print_stats("load/1filex1/onnx", data[-1])

    # %%
    # Load with ``onnx_light.onnx``.

    data.append(measure("load/1filex1/onnxlight", lambda: onnxl.load(onnx_path)))
    print_stats("load/1filex1/onnxlight", data[-1])

    # %%
    # Load with ``onnx_light.onnx`` using parallel tensor loading.

    data.append(
        measure(
            "load/1filex4/onnxlight", lambda: onnxl.load(onnx_path, parallel=True, num_threads=4)
        )
    )
    print_stats("load/1filex4/onnxlight", data[-1])

    # %%
    # Load with ``onnxruntime`` (all optimizations disabled).
    # ``InferenceSession`` is created with ``ORT_DISABLE_ALL`` so the
    # measurement captures only model loading overhead, not graph optimization.

    data.append(
        measure(
            "load/1filex1/ort",
            lambda: ort.InferenceSession(onnx_path, sess_options=_ort_sess_opts),
        )
    )
    print_stats("load/1filex1/ort", data[-1])

# %%
# Serialize and Parse benchmarks
# ------------------------------


def _serialize_onnx() -> bytes:
    """Serializes the ONNX model to bytes."""
    return onx.SerializeToString()


def _serialize_onnxlight() -> bytes:
    """Serializes the onnx_light model to bytes."""
    return onxl.SerializeToString()


def _serialize_onnxlight_x4() -> bytes:
    """Serializes the onnx_light model in parallel to bytes."""
    return onxl.SerializeToString(opts_serial_x4)


if _run_scenario("serialize"):
    opts_serial_x4 = onnxl.SerializeOptions()
    opts_serial_x4.parallel = True
    opts_serial_x4.num_threads = 4

    assert len(_serialize_onnx()) > 0
    assert len(_serialize_onnxlight()) > 0
    assert len(_serialize_onnxlight_x4()) > 0

    data.append(measure("serialize/x1/onnx", _serialize_onnx))
    print_stats("serialize/x1/onnx", data[-1])
    data.append(measure("serialize/x1/onnxlight", _serialize_onnxlight))
    print_stats("serialize/x1/onnxlight", data[-1])
    data.append(measure("serialize/x4/onnxlight", _serialize_onnxlight_x4))
    print_stats("serialize/x4/onnxlight", data[-1])


# %%
# ParseFromString comparison between ``onnx`` and ``onnx_light.onnx``.


def _parse_onnx() -> onnx.ModelProto:
    """Parses ONNX bytes into a ModelProto."""
    parsed = onnx.ModelProto()
    parsed.ParseFromString(serialized_onnx)
    return parsed


def _parse_onnxlight() -> onnxl.ModelProto:
    """Parses onnx_light bytes into a ModelProto."""
    parsed = onnxl.ModelProto()
    parsed.ParseFromString(serialized_onnxlight)
    return parsed


def _parse_onnxlight_x4() -> onnxl.ModelProto:
    """Parses onnx_light bytes in parallel into a ModelProto."""
    parsed = onnxl.ModelProto()
    parsed.ParseFromString(serialized_onnxlight, opts_parse_x4)
    return parsed


def _parse_onnxlight_nc() -> onnxl.ModelProto:
    """Parses onnx_light bytes without copying raw tensor data (zero-copy)."""
    parsed = onnxl.ModelProto()
    parsed.ParseFromString(serialized_onnxlight, opts_parse_nc)
    return parsed


def _parse_onnxlight_nc_x4() -> onnxl.ModelProto:
    """Parses onnx_light bytes in parallel without copying raw tensor data (zero-copy, 4 t)."""
    parsed = onnxl.ModelProto()
    parsed.ParseFromString(serialized_onnxlight, opts_parse_nc_x4)
    return parsed


if _run_scenario("parse"):
    serialized_onnx = onx.SerializeToString()
    serialized_onnxlight = onxl.SerializeToString()
    opts_parse_x4 = onnxl.ParseOptions()
    opts_parse_x4.parallel = True
    opts_parse_x4.num_threads = 4
    opts_parse_nc = onnxl.ParseOptions()
    opts_parse_nc.no_copy = True
    opts_parse_nc_x4 = onnxl.ParseOptions()
    opts_parse_nc_x4.no_copy = True
    opts_parse_nc_x4.parallel = True
    opts_parse_nc_x4.num_threads = 4

    parsed_onnx = _parse_onnx()
    assert parsed_onnx.ir_version == onx.ir_version
    assert len(parsed_onnx.graph.node) == len(onx.graph.node)
    parsed_onnxlight = _parse_onnxlight()
    assert parsed_onnxlight.ir_version == onxl.ir_version
    assert len(parsed_onnxlight.graph.node) == len(onxl.graph.node)
    parsed_onnxlight_x4 = _parse_onnxlight_x4()
    assert parsed_onnxlight_x4.ir_version == onxl.ir_version
    assert len(parsed_onnxlight_x4.graph.node) == len(onxl.graph.node)
    parsed_onnxlight_nc = _parse_onnxlight_nc()
    assert parsed_onnxlight_nc.ir_version == onxl.ir_version
    assert len(parsed_onnxlight_nc.graph.node) == len(onxl.graph.node)
    parsed_onnxlight_nc_x4 = _parse_onnxlight_nc_x4()
    assert parsed_onnxlight_nc_x4.ir_version == onxl.ir_version
    assert len(parsed_onnxlight_nc_x4.graph.node) == len(onxl.graph.node)

    data.append(measure("parse/x1/onnx", _parse_onnx))
    print_stats("parse/x1/onnx", data[-1])
    data.append(measure("parse/x1/onnxlight", _parse_onnxlight))
    print_stats("parse/x1/onnxlight", data[-1])
    data.append(measure("parse/x4/onnxlight", _parse_onnxlight_x4))
    print_stats("parse/x4/onnxlight", data[-1])

    # %%
    # Parse with zero-copy (``no_copy=True``): raw tensor data is not copied.
    # The pointer inside each TensorProto points directly into ``serialized_onnxlight``.
    # The bytes object **must** remain alive for as long as the parsed model is used.

    data.append(measure("parse/nc/onnxlight", _parse_onnxlight_nc))
    print_stats("parse/nc/onnxlight", data[-1])

    # %%
    # Parse with zero-copy **and** parallel tensor reads (``no_copy=True, parallel=True``).
    # Combines the allocation savings of zero-copy with multi-threaded I/O for large models.

    data.append(measure("parse/ncx4/onnxlight", _parse_onnxlight_nc_x4))
    print_stats("parse/ncx4/onnxlight", data[-1])


# %%
# Save benchmarks
# ---------------
#
# Save once with external data (not benchmarked) using ``onnx_light.onnx`` so
# that the in-memory model is not modified (``ClearExternalData`` restores it
# after the C++ write).
# Absolute paths ensure onnxlight stores only the basename in the ``.onnx``
# metadata, letting both ``onnx.load`` and ``onnxl.load`` resolve the data
# file automatically.

if _run_scenario("save"):
    # %%
    # Save with ``onnx``.

    out_onnx = os.path.join(tmp_dir, "out_onnx.onnx")
    data.append(measure("save/1filex1/onnx", lambda: onnx.save(onx, out_onnx)))
    print_stats("save/1filex1/onnx", data[-1])

    # %%
    # Save with ``onnx`` using external data.
    # This is the slow path: Python iterates every tensor, creates a numpy
    # intermediate, and calls Python I/O for each weight blob.

    out_onnx_ext = os.path.join(tmp_dir, "out_onnx_ext.onnx")
    out_onnx_ext_location = "out_onnx_ext.data"
    out_onnx_ext_data = os.path.join(tmp_dir, out_onnx_ext_location)

    def _save_onnx_external_with_flush() -> None:
        onnx.save_model(
            onx,
            out_onnx_ext,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=out_onnx_ext_location,
        )
        _flush_file(out_onnx_ext_data)
        _flush_file(out_onnx_ext)

    data.append(measure("save/2filex1/onnx", _save_onnx_external_with_flush, n=1, warmup=0))
    print_stats("save/2filex1/onnx", data[-1])

    # %%
    # The onnx file is modified to store the external data.
    # Let's make sure it is not used again.
    onx = None

    # %%
    # Save with ``onnx_light.onnx``.

    out_onnxl = os.path.join(tmp_dir, "out_onnxlight.onnx")
    data.append(measure("save/1filex1/onnxlight", lambda: onnxl.save(onxl, out_onnxl)))
    print_stats("save/1filex1/onnxlight", data[-1])

    # %%
    # Save with ``onnx_light.onnx`` parallelized.

    out_onnxl_x4 = os.path.join(tmp_dir, "out_onnxlight_x4.onnx")
    data.append(
        measure(
            "save/1filex4/onnxlight",
            lambda: onnxl.save(onxl_x4, out_onnxl_x4, parallel=True, num_threads=4),
        )
    )
    print_stats("save/1filex4/onnxlight", data[-1])

    # %%
    # Save with ``onnx_light.onnx`` using external data.
    # All work is done in C++: ``PopulateExternalData`` attaches metadata once,
    # ``SerializeToStream`` routes large ``raw_data`` blobs directly to the
    # weights file via ``TwoFilesWriteStream``, and ``ClearExternalData``
    # restores the in-memory model.  No numpy arrays are created.
    # As for the ``onnx`` row, the two output files are explicitly ``fsync``-ed
    # so both benchmarks include descriptor flush/write-back costs.
    # The main ``.onnx`` structure is accumulated in a ``StringWriteStream``
    # (memory buffer) and flushed to disk in a single write after all tensor
    # data has been written, mirroring the sequential I/O pattern used by
    # ``onnx.save_model`` and allowing OS-level write coalescing.

    out_ext = os.path.join(tmp_dir, "out_ext.onnx")
    out_ext_data = out_ext + ".data"

    def _save_onnxlight_external_with_flush() -> None:
        onnxl.save(onxl, out_ext, location=out_ext_data)
        _flush_file(out_ext_data)
        _flush_file(out_ext)

    data.append(measure("save/2filex1/onnxlight", _save_onnxlight_external_with_flush))
    print_stats("save/2filex1/onnxlight", data[-1])

    # %%
    # Save with ``onnx_light.onnx`` using external data parallelized.

    out_ext_x4 = os.path.join(tmp_dir, "out_ext_x4.onnx")
    out_ext_x4_data = out_ext_x4 + ".data"
    data.append(
        measure(
            "save/2filex4/onnxlight",
            lambda: onnxl.save(
                onxl, out_ext_x4, location=out_ext_x4_data, parallel=True, num_threads=4
            ),
        )
    )
    print_stats("save/2filex4/onnxlight", data[-1])

# %%
# C++ benchmarks
# --------------
#
# Run the standalone C++ benchmark executables when available.
# These scenarios measure the same operations as ``load`` and ``save``
# but use the compiled C++ timing executables directly, bypassing the
# Python interpreter overhead entirely.

if _run_scenario("cpp"):
    # %%
    # Load with standalone C++ ``load_onnx_light_time`` example when available.
    # The executable uses ``FileStream`` as well, so this row measures the same
    # file-backed parsing path as ``onnxl.load(onnx_path)``.

    cpp_load_x1 = _measure_cpp_load_with_example(onnx_path, num_threads=1)
    if cpp_load_x1 is not None:
        data.append(cpp_load_x1)
        print_stats(cpp_load_x1["name"], cpp_load_x1)
    else:
        print(
            "load_onnx_light_time executable not found (or failed), skipping C++ load benchmark."
        )

    cpp_load_x4 = _measure_cpp_load_with_example(onnx_path, num_threads=4)
    if cpp_load_x4 is not None:
        data.append(cpp_load_x4)
        print_stats(cpp_load_x4["name"], cpp_load_x4)

    # %%
    # Load with standalone C++ ``load_onnx_time`` example when available.
    # The executable uses the standard onnx protobuf library for loading.

    cpp_load_onnx_x1 = _measure_cpp_load_with_example(
        onnx_path, num_threads=1, executable_name="load_onnx_time"
    )
    if cpp_load_onnx_x1 is not None:
        data.append(cpp_load_onnx_x1)
        print_stats(cpp_load_onnx_x1["name"], cpp_load_onnx_x1)
    else:
        print("load_onnx_time executable not found (or failed), skipping C++ load benchmark.")

    # %%
    # Save with standalone C++ ``save_onnx_light_time`` example when available.

    cpp_save_x1 = _measure_cpp_save_with_example(onnx_path, num_threads=1)
    if cpp_save_x1 is not None:
        data.append(cpp_save_x1)
        print_stats(cpp_save_x1["name"], cpp_save_x1)
    else:
        print(
            "save_onnx_light_time executable not found (or failed), skipping C++ save benchmark."
        )

    cpp_save_x4 = _measure_cpp_save_with_example(onnx_path, num_threads=4)
    if cpp_save_x4 is not None:
        data.append(cpp_save_x4)
        print_stats(cpp_save_x4["name"], cpp_save_x4)

# %%
# Load with ``onnx`` using external data
# --------------------------------------
#
# Reload the model previously saved with external data using ``onnx.load``.

if _run_scenario("load"):
    data.append(
        measure("load/2filex1/onnx", lambda: onnx.load(ext_load_onnx, load_external_data=True))
    )
    print_stats("load/2filex1/onnx", data[-1])

    # %%
    # Load with ``onnx_light.onnx`` using external data.
    # Reload the same external-data model using ``onnxl.load``.

    data.append(
        measure(
            "load/2filex1/onnxlight", lambda: onnxl.load(ext_load_onnx, location=ext_load_data)
        )
    )
    print_stats("load/2filex1/onnxlight", data[-1])

    # %%
    # Load with ``onnx_light.onnx`` using external data and parallel tensor loading.
    # Combine external-data loading with ``parallel=True`` for maximum throughput.

    data.append(
        measure(
            "load/2filex4/onnxlight",
            lambda: onnxl.load(
                ext_load_onnx, location=ext_load_data, parallel=True, num_threads=4
            ),
        )
    )
    print_stats("load/2filex4/onnxlight", data[-1])

    # %%
    # Load with ``onnxruntime`` using external data (all optimizations disabled).
    # Reload the external-data model with ``onnxruntime``, keeping
    # ``ORT_DISABLE_ALL`` so only loading overhead is measured.

    data.append(
        measure(
            "load/2filex1/ort",
            lambda: ort.InferenceSession(ext_load_onnx, sess_options=_ort_sess_opts),
        )
    )
    print_stats("load/2filex1/ort", data[-1])

# %%
# Results
# --------

df = pandas.DataFrame(data).set_index("name").sort_index()
print(df)
df = df.sort_index(ascending=False)

# %%
# Plot the results.
# The average and median are shown for each operation, with a 95% confidence
# interval annotation derived from the measured standard deviation.
# Bars are colored by library: blue family for ``onnx``, orange family for
# ``onnx_light``, green family for ``onnxruntime``.  Solid shades represent
# the average; lighter shades the median.
import matplotlib.patches as mpatches

_onnx_avg = "steelblue"
_onnx_med = "lightsteelblue"
_onnx_light_avg = "darkorange"
_onnx_light_med = "moccasin"
_ort_avg = "seagreen"
_ort_med = "lightgreen"

ax = df[["avg", "median"]].plot.barh(
    title=(
        f"onnx vs onnx_light vs ort load/save (s), size={file_size / 2 ** 20:.2f} MB "
        f"(lower is better)\n"
        f"benchmark key: <op>/<files>x<threads>/<lib>\n"
        f"op=load|save|parse|serialize, files=1|2, threads=1|4, lib=onnx|onnxlight|ort"
    ),
    xlabel="seconds",
    legend=False,
    figsize=(12, 8),
)

# Row names use "onnxlight" / "ort" as recorded during benchmarking.
row_names = df.index.tolist()
for container, col in zip(ax.containers, ["avg", "median"]):
    for bar, name in zip(container, row_names):
        if "onnxlight" in name:
            if col == "avg":
                bar.set_facecolor(_onnx_light_avg)
            elif col == "median":
                bar.set_facecolor(_onnx_light_med)
        elif "/ort" in name:
            if col == "avg":
                bar.set_facecolor(_ort_avg)
            elif col == "median":
                bar.set_facecolor(_ort_med)
        else:
            if col == "avg":
                bar.set_facecolor(_onnx_avg)
            elif col == "median":
                bar.set_facecolor(_onnx_med)

first_container = ax.containers[0]
for bar, name in zip(first_container, row_names):
    std = df.loc[name, "std"]
    if not np.isfinite(std):
        continue
    ci = 1.96 * std
    ax.text(
        bar.get_width(),
        bar.get_y() + bar.get_height() / 2.0,
        f" ±{ci * 1e3:.1f} ms",
        va="center",
        ha="left",
    )

legend_handles = [
    mpatches.Patch(color=_onnx_avg, label="onnx avg"),
    mpatches.Patch(color=_onnx_med, label="onnx median"),
    mpatches.Patch(color=_onnx_light_avg, label="onnx_light avg"),
    mpatches.Patch(color=_onnx_light_med, label="onnx_light median"),
    mpatches.Patch(color=_ort_avg, label="ort avg"),
    mpatches.Patch(color=_ort_med, label="ort median"),
]
ax.legend(handles=legend_handles)
ax.grid(axis="x")
for label in ax.get_yticklabels():
    label.set_horizontalalignment("left")
ax.tick_params(axis="y", pad=120)
ax.figure.tight_layout()
ax.figure.savefig("plot_onnx_time.png")

# %%
# Cleanup
# --------
# Remove all temporary files created during the benchmark.

shutil.rmtree(tmp_dir, ignore_errors=True)
