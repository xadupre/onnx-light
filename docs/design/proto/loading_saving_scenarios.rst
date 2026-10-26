.. _l-design-loading-saving-scenarios:

Complex loading and saving scenarios
====================================

The basic load/save patterns of *onnx-light* (one file, two files,
parallel I/O, alignment) are covered side by side in
:ref:`l-howto-load-save-onnx-files`.  This page focuses on two more
advanced scenarios that go beyond the basic patterns and motivated the
addition of two dedicated helpers:

* :func:`onnx_light.onnx.align_external_data_streaming`
* :func:`onnx_light.onnx.save_model_with_shared_external_data`

Both helpers share a common goal: **manipulate large two-file (or
multi-file) ONNX models without paying the cost of materialising the full
set of weights in RAM** — a constraint that the standard :func:`onnx_light.onnx.load` /
:func:`onnx_light.onnx.save` pair cannot satisfy for very large models, and that the
protobuf-free design of *onnx-light* makes natural to implement.

Scenario 1 — Re-align external weights without loading them
-----------------------------------------------------------

**Problem.** A model has already been saved as a
``(.onnx, .data[, .data.1, ...])`` pair, possibly several gigabytes of
weights spread over multiple files.  A downstream consumer (mmap-based
loader, accelerator runtime) now requires each tensor offset in the
weights file to be aligned to a power-of-two boundary, typically
``4096`` bytes.  The host running the conversion does not have enough
memory to load the full set of weights just to re-serialize them.

**Naive approach.**  ``load(load_external_data=True)`` followed by
:meth:`SerializeToFile` with :attr:`SerializeOptions.alignment` set to the
desired boundary.  This works but requires the full weights payload to
fit in RAM during the round-trip.

**Streaming approach.**
:func:`onnx_light.onnx.align_external_data_streaming` parses **only the
proto metadata** of the source ``.onnx`` (using ``skip_raw_data=True``)
and then streams each tensor's bytes from the source weights file(s) to a
single consolidated destination weights file in chunks of ``chunk_size``
bytes, zero-padding to the requested alignment between tensors.  Peak
heap usage is bounded by *(metadata size + chunk_size)* — independent of
the total weights size.

Key properties:

* Source files are read-only and unmodified.
* The destination always uses a **single** consolidated weights file,
  even when the source spread tensors across multiple
  ``external_data.location`` files.
* On Linux the in-kernel ``splice(2)`` syscall is used for zero-copy
  file-to-file transfer when the kernel supports it for the underlying
  file types; otherwise a regular ``read`` + ``write`` loop bounded by
  ``chunk_size`` is used.
* The output is byte-equivalent to the in-memory variant for the tensor
  payloads at the same alignment — only the peak memory footprint
  differs.

Recipe and step-by-step explanation:
:ref:`l-howto-align-external-data-streaming`.
Runnable benchmark vs the in-memory variant:
:ref:`l-example-plot-align-external-data-streaming`.

Scenario 2 — Save a variant model sharing weights with another on-disk model
----------------------------------------------------------------------------

**Problem.** A first model has been saved to disk with external data,
producing ``first.onnx`` + ``first.onnx.data``.  A variant of that model
is then assembled in memory by mixing some of the first model's
initializers (loaded **without** external data, so they still carry their
original ``external_data`` metadata) with brand-new initializers that
carry inline ``raw_data`` (for example a fresh head, a quantized branch,
or extra adapters).  The variant must be saved as ``second.onnx`` next to
``first.onnx`` **without duplicating the bytes of the reused
initializers**.

**Naive approach.**  Calling ``onnxl.save(second, "second.onnx",
location="second.onnx.data")`` materialises every initializer — including
the reused ones — into ``second.onnx.data``.  Storage cost is doubled
and write time is multiplied by the total weight size, even though most
of those bytes already exist on disk.

**Shared-data approach.**
:func:`onnx_light.onnx.save_model_with_shared_external_data` walks the
model's initializers and:

* For each initializer already marked ``EXTERNAL``, copies its
  ``external_data`` entry verbatim into ``second.onnx``.  No byte is
  read from the source weights file; the ``location``, ``offset``, and
  ``length`` are written out unchanged.  The caller is responsible for
  the recorded ``location`` remaining resolvable relative to
  ``second.onnx``'s parent directory.
* For each initializer carrying inline ``raw_data``, writes the bytes
  into a single secondary file at ``second.onnx.data`` at aligned
  offsets, clears the inline bytes from the in-memory proto, and updates
  the proto's ``external_data`` entry to point at that secondary file.
  The secondary file is not created at all when every initializer is
  reused.

Key properties:

* The first model's files are read-only and unmodified.
* ``second.onnx.data`` contains *only* the new weights, never the reused
  ones — minimising both disk usage and write time.
* The function honours :attr:`SerializeOptions.alignment` for the new
  initializers, so the secondary file is mmap-friendly when needed.
* The input model is mutated in place: new initializers no longer carry
  inline ``raw_data`` after the call; they reference the secondary file
  instead.

Recipe and step-by-step explanation:
:ref:`l-howto-save-model-with-shared-external-data`.

Automatic external-write scheduling
-----------------------------------

External-data output uses a buffered-write policy, not the cold-read policy.
With parallel serialization enabled, ``min_block_size=0`` selects a **1 MiB**
per-block cutoff. Blocks at or above the cutoff are submitted to workers;
smaller blocks stay on the calling thread. Positive cutoffs are preserved,
including ``min_block_size=1`` to submit every nonempty block. ``num_threads=1``
(or ``0``) disables parallel serialization. Explicit worker counts greater
than one are preserved; automatic workers retain the half-logical-core ceiling,
bounded by the amount of work available. Cold reads retain their 16 MiB default.

The distinction matters for LLM initializers: a 16 MiB cutoff excludes common
1--12 MiB attention and feed-forward matrices. In the Qwen3-0.6B workload
reported in issue :issue:`5633`, approximately 55% of weight bytes were below
16 MiB. The reported isolated save with an explicit 1 MiB cutoff took 5.83 s
versus 8.98 s with the old automatic cutoff, but the latter repeated at 6.32 s.
These variable timings motivate a lower write cutoff, not a universal speedup
claim. A 1 MiB cutoff still avoids scheduling individual tiny normalization
weights; batching them is not required to move substantial matrix bytes off the
calling thread.

The existing ``bench_save_file`` benchmark supports ``-b`` (parallel cutoff in
bytes; zero means automatic) and ``-p llm``. In that profile, ``-i`` specifies
layer groups, each containing two 4 KiB blocks and matrices of 1, 2, 4, 8, 12,
12, and 16 MiB, plus one shared 64 MiB embedding-like block. This is a synthetic
size distribution, not pretrained Qwen weights. ``-p uniform`` retains the
original square-matrix workload.

For example, after building the ``bench_save_file`` CMake target:

.. code-block:: bash

    ./build/bench_save_file -p llm -i 8 -n 5 -t 1
    ./build/bench_save_file -p llm -i 8 -n 5 -t -1 -b 16777216
    ./build/bench_save_file -p llm -i 8 -n 5 -t -1 -b 4194304
    ./build/bench_save_file -p llm -i 8 -n 5 -t -1 -b 1048576
    ./build/bench_save_file -p llm -i 8 -n 5 -t -1 -b 0
    ./build/bench_save_file -i 2048 -d 16 -n 5 -t -1 -b 1
    ./build/bench_save_file -i 2048 -d 16 -n 5 -t -1 -b 0

Every iteration closes and removes its outputs before the next save, and reports
the resolved workers, threshold, and actually submitted bytes. Timings cover
native serialization and buffered file writes, **not** durable ``fsync`` or the
independent full-model copy in the Python save path. Compare repeated runs on the
same filesystem; do not interpret this native timing as end-to-end Python export.

A local Release build (GCC 13.3, Linux/ext4, four logical processors, five
iterations per configuration) produced the following results for ``-p llm -i 8``.
The payload is 504.0625 MiB across 73 initializers; automatic scheduling resolves
to two workers.

.. list-table::
   :header-rows: 1

   * - Workers / cutoff
     - Submitted MiB
     - Median ms
     - Min--max ms
   * - Sequential
     - 0
     - 759.1
     - 538.7--836.0
   * - Automatic / 16 MiB
     - 192
     - 805.6
     - 682.6--808.2
   * - Automatic / 4 MiB
     - 480
     - 804.7
     - 583.7--853.9
   * - Automatic / 1 MiB
     - 504
     - 806.9
     - 640.9--852.6
   * - Automatic / default (1 MiB)
     - 504
     - 787.5
     - 673.4--792.8
   * - Four / 1 MiB
     - 504
     - 692.7
     - 549.8--804.0

The explicit and automatic 1 MiB policies are identical; their different timings
illustrate run variability. These measurements do not establish a stable speedup
or an optimal universal cutoff. They support 1 MiB as a coverage/overhead compromise:
it submits all substantial matrices here instead of leaving 312 MiB on the caller,
while excluding the 64 KiB of tiny weights. In the separate 2048-by-1-KiB workload,
submitting every block (``-b 1``) took a median 20.4 ms (13.7--36.3 ms), versus
12.9 ms (12.7--14.3 ms) with the default and no submitted blocks. Batching or a
different explicit cutoff can still be useful for other storage and workloads.
