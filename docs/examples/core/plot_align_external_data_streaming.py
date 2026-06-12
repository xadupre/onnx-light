"""
.. _l-example-plot-align-external-data-streaming:

Benchmark streaming vs in-memory alignment of external data
===========================================================

This example benchmarks two ways to re-align tensor offsets in an
ONNX model that stores its weights as external data:

1. **Streaming approach** —
   :func:`onnx_light.onnx.align_external_data_streaming` rewrites a
   ``(src.onnx, src.data)`` pair into a new ``(dst.onnx, dst.data)`` pair
   with every tensor offset aligned to a power-of-two boundary, **without
   ever loading the weights in RAM**.  Only the proto metadata is parsed;
   tensor bytes are streamed file-to-file (via ``splice(2)`` on Linux for
   zero-copy in kernel space).

2. **Usual in-memory approach** —
   :func:`onnx_light.onnx.load` followed by ``SerializeToFile`` with
   :attr:`onnx_light.onnx.SerializeOptions.alignment` set to the same
   boundary.  This loads the full set of weights into memory before
   re-writing them.

The example reports wall-clock time and peak process-resident memory
usage for each approach.  At equal alignment the two outputs are
byte-equivalent for the tensor payloads — only the peak memory
footprint differs.

.. note::
    Peak memory is sampled from the process's resident set size (RSS),
    not from :mod:`tracemalloc`.  ``tracemalloc`` only tracks the
    Python allocator (``pymalloc``), so it misses the C++-owned buffers
    where onnx-light stores tensor ``raw_data`` — making both approaches
    appear similarly small and hiding the very overhead this example
    is meant to illustrate.  RSS captures both Python and C++
    allocations.
"""

import os
import shutil
import threading
import time

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas

import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh

# %%
# Build a synthetic model
# -----------------------
#
# A handful of large initializers is enough to make the in-memory peak
# noticeably larger than the streaming peak.  The example runs with a
# smaller model under ``UNITTEST_GOING=1`` to keep CI fast.

N_INIT = 8 if os.environ.get("UNITTEST_GOING") == "1" else 24
DIM = 256 if os.environ.get("UNITTEST_GOING") == "1" else 1024
ALIGNMENT = 4096


def make_model(n_init: int = N_INIT, dim: int = DIM) -> onnxl.ModelProto:
    """Creates a synthetic ONNX model with *n_init* float32 Gemm weights."""
    initializers = []
    nodes = []
    inputs = [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [None, dim])]

    prev = "X"
    for i in range(n_init):
        weight_name = f"W{i}"
        out_name = f"Y{i}"
        w = np.random.randn(dim, dim).astype(np.float32)
        initializers.append(onh.from_array(w, name=weight_name))
        nodes.append(oh.make_node("Gemm", [prev, weight_name], [out_name], transB=1))
        prev = out_name

    outputs = [oh.make_tensor_value_info(prev, onnxl.TensorProto.FLOAT, [None, dim])]
    graph = oh.make_graph(nodes, "bench_graph", inputs, outputs, initializer=initializers)
    return oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)], ir_version=9)


out_dir = "temp_plot_align_external_data_streaming"
os.makedirs(out_dir, exist_ok=True)

model = make_model()
total_weight_bytes = sum(int(np.prod(init.dims)) * 4 for init in model.graph.initializer)
print(f"Number of initializers : {len(model.graph.initializer)}")
print(f"Total weight bytes     : {total_weight_bytes / 2 ** 20:.2f} MB")
print(f"Alignment              : {ALIGNMENT} bytes")

# %%
# Save the source two-file model
# ------------------------------
#
# Every benchmark below starts from the same on-disk source pair
# ``(src.onnx, src.data)``.

src_onnx = os.path.join(out_dir, "src.onnx")
src_data = src_onnx + ".data"


def _save_source() -> None:
    """Writes the synthetic model to a two-file ``(.onnx, .data)`` pair."""
    onnxl.save(model, src_onnx, location=src_data)


_save_source()
print(f"Source two-file model : {src_onnx} + {src_data}")
print(f"Source weights size   : {os.path.getsize(src_data) / 2 ** 20:.2f} MB")


# %%
# Helpers
# -------
#
# Each benchmark is wrapped in a function that returns the elapsed wall
# time and the peak process-resident memory (RSS) observed while the
# function runs.  RSS is sampled from ``/proc/self/statm`` (Linux) on a
# background thread; on platforms where ``/proc`` is unavailable the
# sampler falls back to :func:`resource.getrusage` (Unix) or reports
# ``0``.  Unlike :mod:`tracemalloc`, RSS includes the C++-owned buffers
# where onnx-light stores tensor ``raw_data`` during an in-memory load
# — which is exactly the overhead the streaming variant avoids.


_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def _read_rss_bytes() -> int:
    """Returns the current process resident set size in bytes (0 if unknown)."""
    try:
        with open("/proc/self/statm", encoding="ascii") as stream:
            return int(stream.read().split()[1]) * _PAGE_SIZE
    except OSError:
        pass
    try:
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB, macOS reports bytes; both branches are >= our usage scale.
        return ru * 1024 if ru < 2**40 else ru
    except (ImportError, OSError):
        return 0


class _PeakRssSampler:
    """Samples process RSS on a background thread and records the peak observed."""

    def __init__(self, interval: float = 0.001):
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.baseline = 0
        self.peak = 0

    def __enter__(self):
        self.baseline = _read_rss_bytes()
        self.peak = self.baseline
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        cur = _read_rss_bytes()
        if cur > self.peak:
            self.peak = cur

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            cur = _read_rss_bytes()
            if cur > self.peak:
                self.peak = cur


def measure(name: str, fn) -> dict:
    """Runs *fn* and returns elapsed time (s) and peak RSS delta (MB)."""
    with _PeakRssSampler() as sampler:
        t0 = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - t0
    peak_mb = max(sampler.peak - sampler.baseline, 0) / 2**20
    print(f"{name:<40} time={elapsed * 1e3:8.1f} ms   peak={peak_mb:7.2f} MB")
    return {"name": name, "time_s": elapsed, "peak_mb": peak_mb}


def _flush(path: str) -> None:
    """fsync a file so timings include write-back to disk."""
    with open(path, "r+b") as stream:
        stream.flush()
        os.fsync(stream.fileno())


# %%
# Approach 1 — streaming alignment
# --------------------------------
#
# The source ``.onnx`` is parsed with ``skip_raw_data=True``, so only the
# metadata (a few KB) reaches RAM; tensor bytes are streamed
# file-to-file in chunks of ``chunk_size`` bytes, zero-padding to the
# requested alignment between tensors.

streaming_onnx = os.path.join(out_dir, "streaming.onnx")
streaming_data = os.path.join(out_dir, "streaming.data")


def run_streaming() -> None:
    onnxl.align_external_data_streaming(
        src_onnx_path=src_onnx,
        dst_onnx_path=streaming_onnx,
        dst_weights_path=streaming_data,
        alignment=ALIGNMENT,
        chunk_size=4 * 1024 * 1024,
    )
    _flush(streaming_data)
    _flush(streaming_onnx)


# %%
# Approach 2 — in-memory load + re-save with alignment
# ----------------------------------------------------
#
# The "usual" path: load the full model (weights and all) into RAM, then
# re-serialize it with :attr:`SerializeOptions.alignment` set to the
# same boundary.

inmem_onnx = os.path.join(out_dir, "inmem.onnx")
inmem_data = os.path.join(out_dir, "inmem.data")


def run_in_memory() -> None:
    loaded = onnxl.load(src_onnx, load_external_data=True, num_threads=1)
    opts = onnxl.SerializeOptions()
    opts.alignment = ALIGNMENT
    # Disable honouring existing external_data.location entries so every
    # tensor is written into the destination weights file passed below.
    opts.use_external_data_location = False
    loaded.SerializeToFile(inmem_onnx, opts, inmem_data)
    _flush(inmem_data)
    _flush(inmem_onnx)


# %%
# Run the benchmarks
# ------------------

results = []
results.append(measure("align/streaming", run_streaming))
results.append(measure("align/in-memory", run_in_memory))

# %%
# Verify byte-equivalence of the aligned outputs
# ----------------------------------------------
#
# Both approaches must produce the same tensor payloads at the same
# aligned offsets.  We compare each tensor's bytes read from both
# destination weights files.

streaming_loaded = onnxl.load(streaming_onnx, load_external_data=False)
inmem_loaded = onnxl.load(inmem_onnx, load_external_data=False)

with open(streaming_data, "rb") as f_stream, open(inmem_data, "rb") as f_inmem:
    for s_init, m_init in zip(streaming_loaded.graph.initializer, inmem_loaded.graph.initializer):
        s_meta = {e.key: e.value for e in s_init.external_data}
        m_meta = {e.key: e.value for e in m_init.external_data}
        s_off, s_len = int(s_meta["offset"]), int(s_meta["length"])
        m_off, m_len = int(m_meta["offset"]), int(m_meta["length"])
        assert s_off % ALIGNMENT == 0, f"streaming offset {s_off} not aligned"
        assert m_off % ALIGNMENT == 0, f"in-memory offset {m_off} not aligned"
        assert s_len == m_len, f"length mismatch: {s_len} vs {m_len}"
        f_stream.seek(s_off)
        f_inmem.seek(m_off)
        assert f_stream.read(s_len) == f_inmem.read(m_len), f"payload mismatch on {s_init.name}"
print("Both approaches produced byte-equivalent aligned tensor payloads.")

# %%
# Results
# -------

df = pandas.DataFrame(results).set_index("name").sort_index()
print(df)

# %%
# Plot
# ----
#
# The streaming variant trades a slightly different I/O pattern for an
# **O(metadata) + chunk_size** memory footprint, independent of the
# total weights size.  The in-memory variant peaks proportionally to
# the model's weight payload.

fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(12, 5))

row_names = df.index.tolist()
colors = ["darkorange" if "streaming" in n else "steelblue" for n in row_names]

ax_time.barh(row_names, df["time_s"], color=colors)
ax_time.set_title("Wall time (s, lower is better)")
ax_time.set_xlabel("seconds")
ax_time.grid(axis="x")

ax_mem.barh(row_names, df["peak_mb"], color=colors)
ax_mem.set_title("Peak RSS delta (MB, lower is better)")
ax_mem.set_xlabel("MB (resident set size)")
ax_mem.grid(axis="x")

fig.suptitle(
    f"Alignment={ALIGNMENT}B   weights={total_weight_bytes / 2 ** 20:.1f} MB"
    f"   #initializers={len(model.graph.initializer)}"
)
fig.legend(
    handles=[
        mpatches.Patch(color="darkorange", label="streaming"),
        mpatches.Patch(color="steelblue", label="in-memory"),
    ],
    loc="lower center",
    ncol=2,
)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig("plot_align_external_data_streaming.png")

# %%
# Cleanup
# -------

shutil.rmtree(out_dir, ignore_errors=True)
