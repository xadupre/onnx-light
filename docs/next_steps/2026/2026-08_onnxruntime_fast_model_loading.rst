.. _l-next-steps-model-loading:

Using ``onnx-light`` fast loading in ``onnxruntime``
====================================================

:Date: 2026-08

**ready; native completion is implemented and issue #4612 remains open**

Objective
+++++++++

This is step 4 and the final part of the fast-loading sequence. It starts only
after :ref:`l-next-steps-native-fast-loading-completion` has stabilized the
native ownership, payload-resolution, prepared-object, overlap, residency, and
device-variant contracts.

The completed :ref:`l-next-steps-ort-onnx-light` roadmap made
``onnxruntime_USE_ONNX_LIGHT`` a build-time replacement for protobuf and the
standard ``onnx`` library. The bug-fix roadmap accelerates that compatibility
path without changing ORT. This document contains only the additional
ownership-aware work that requires an explicit contract between repositories.
Native graph resolution, prepared caches, offloading, and first-token overlap
are completed first in ``xadupre/onnx-light``; they are not ORT PRs.

Benchmark contract
++++++++++++++++++

Use the same model identity, compiler, build type, graph optimization level,
thread policy, affinity, and cache state for:

.. list-table::
    :header-rows: 1
    :widths: 30 20 25 25

    * - Consumer
      - Format
      - Payload mode
      - Purpose
    * - ORT + protobuf
      - ``.onnx``
      - ordinary ORT loading
      - compatibility baseline
    * - ORT + ``onnx-light``
      - ``.onnx``
      - ordinary ORT loading
      - parser/adapter effect
    * - ORT + ``onnx-light``
      - ``.onnx``
      - owned mapped payloads
      - explicit integration effect
    * - ORT
      - ``.ort``
      - ORT-optimized
      - strongest startup baseline

Report at least main-protobuf completion, external payload availability,
session ready, first inference, first token, peak private memory, physical and
logical bytes read, bytes copied/borrowed/mapped, and major/minor page faults.
An untouched mmap result is address-space construction, not weight readiness.

Ownership contract
++++++++++++++++++

Every payload handed across the repository boundary has one explicit state:

.. code-block:: text

    owned bytes
    borrowed mapped bytes + shared mapping owner
    borrowed caller bytes + caller lifetime token
    lazy descriptor + source owner
    final-destination read descriptor

A raw pointer without an owner or final-destination contract is invalid. ORT
may borrow an immutable mapped range only when graph optimization and the
selected execution provider do not require writable memory, different
alignment, relocation, or another physical layout. Otherwise ``onnx-light``
must describe the source range so ORT reads directly into its final allocation;
it must not introduce an intermediate whole-tensor buffer.

Completed prerequisite -- expose the payload contract
++++++++++++++++++++++++++++++++++++++++++++++++++++++

**Repository:** ``xadupre/onnx-light``

Expose a narrow C++ contract independent of ORT types:

.. code-block:: cpp

    struct MappedPayload {
      const void* data;
      size_t size;
      size_t alignment;
      std::shared_ptr<void> owner;
      PayloadIdentity identity;
    };

The same API exposes a validated final-destination read descriptor for
ineligible ranges.

Acceptance:

* mapped payloads retain a shared owner and stable identity;
* tests cover owner release, alignment, file replacement, truncated ranges,
  concurrent views, and source-path confinement;
* ineligible ranges require no intermediate full-tensor allocation;
* the public contract contains no ORT-specific type.

**Status: implemented in** ``onnx-light`` (``onnx_light/onnx_proto/onnx_mapped_payload.h``).
``PayloadIdentity``, ``MappedPayload``, ``FinalDestinationReadDescriptor``, and
``MappedPayloadSource`` are exposed independently of ORT, built on the existing
``mmap_file_as_shared_ptr`` / ``validate_external_weights_read_path``
confinement helpers in ``stream.h``. This early onnx-light prerequisite did not start the onnxruntime
implementation. The native completion roadmap is now finished, so issue #4612
can consume the stable contract.

Final ORT PR -- consume owned payloads in session state
+++++++++++++++++++++++++++++++++++++++++++++++++++++++

**Repository:** ``microsoft/onnxruntime``

Tracked by `issue #4612
<https://github.com/xadupre/onnx-light/issues/4612>`_. This PR follows
microsoft/onnxruntime#29723 and depends on the completed
onnx-light payload contract plus every native completion issue through #4623.
Attach ``MappedPayload::owner`` to ``SessionState`` and create an immutable CPU
initializer view only when the selected consumers permit borrowing. Otherwise
read through the final-destination descriptor into the final ORT allocation.

Acceptance:

* every borrowed initializer has a session-owned lifetime token;
* optimization and execution-provider prepacking cannot retain dangling
  pointers;
* ineligible tensors take a documented direct-read path;
* the ``onnx-light`` build regresses by no more than 3% at
  ``T_first_token`` against the same ORT protobuf configuration;
* every performance claim also reports the equivalent ORT ``.ort`` result.

Security and failure semantics
++++++++++++++++++++++++++++++

External locations remain normalized and confined to the model directory
unless a caller supplies a trusted resolver. Symlinks, offsets, lengths,
integer overflow, file replacement, and truncated reads are validated before a
pointer is exposed. Checksums and source identity are verified before a payload
becomes visible. Failures preserve phase, tensor, source, range, and underlying
reason and are never converted into empty tensors or successful sessions.

Completion
++++++++++

After the final onnxruntime PR merges, run the complete four-configuration
benchmark contract. This closes the fast-loading sequence; no unfinished
onnx-light loading issue may be deferred behind the cross-repository work.
