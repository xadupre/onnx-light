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
set of weights in RAM** — a constraint that the standard ``load`` /
``save`` pair cannot satisfy for very large models, and that the
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
``SerializeToFile`` with :attr:`SerializeOptions.alignment` set to the
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

Why these scenarios are first-class in *onnx-light*
---------------------------------------------------

Both scenarios rely on two capabilities that are awkward to express with
the standard protobuf-based ``onnx`` package and that *onnx-light*
exposes by design:

* **Skip raw data on parse.** ``ParseOptions::skip_raw_data`` (and the
  Python equivalent) lets a caller read only the proto metadata — even
  for multi-GB models — so the metadata-driven re-alignment of scenario 1
  is possible without ever touching the weights bytes.
* **Honour pre-existing ``external_data`` entries on serialize.**
  ``SerializeOptions::use_external_data_location`` controls whether the
  serializer respects per-tensor ``external_data.location`` entries that
  are already set on initializers.  Scenario 2 builds on that to keep
  reused initializers pointing at the first model's weights file while
  the freshly added initializers are routed to a new file.

The two helpers documented here package those primitives into ready-made
recipes for the two most common "complex" scenarios encountered when
shipping large ONNX models.
