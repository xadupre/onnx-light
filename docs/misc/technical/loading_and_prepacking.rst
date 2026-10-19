.. _l-design-runtime-loading-prepacking:
.. _l-technical-details-loading-prepacking:

Model loading and kernel preparation
====================================

Model loading and kernel preparation are separate operations. The current
``RuntimeSession`` path prepares immutable kernel inputs synchronously during
its first run; it does not overlap model execution with a dynamic graph of
weight-loading and preparation tasks.

Session preparation
-------------------

The root session owns a non-null ``ValueStore`` from construction, creating
one unless ``RuntimeSessionOptions::value_store`` explicitly supplies an owner.
Nested graphs and model-local function sessions receive the same owner.

During the first ``RuntimeSession::Run``:

1. The session installs its store into the runtime context.
2. It resolves kernels and their tuning, and identifies immutable inputs.
3. It invokes kernel preparation hooks before executing the session's nodes.

``Gemm`` prepares a non-overridable constant ``B`` initializer through this
path. Its key includes the source contents, type, shape, transpose mode, and
prepared layout. Preparation allocates and publishes the packed tensor once.
An initializer that is also a graph input remains overridable and is not
treated as an immutable weight.

The kernel retains a validated ``ValueStore::BoundSlot`` for replay. Subsequent
runs read the prepared value by direct indexing, without a string-keyed lookup,
readiness wait, preparation mutex, or scheduler dispatch.

Ownership and memory
--------------------

The store owns the prepared values and their allocation arenas. The session
and any contexts sharing the store keep that owner alive, so prepared storage
is not released while its session is still alive. A bound slot itself is
borrowed and must not be used after the store is destroyed.

Independent sessions have independent stores by default. Sharing a store must
be explicit, and all sharing sessions must serialize their runs.

``RuntimeSession::prepared_bytes()`` sums live buffer sizes in the store,
including allocated but unpublished values. It excludes free retained
capacity and the context's execution and I/O arenas. There is no preparation
residency budget or eviction policy on this path.

Loading and execution resources
-------------------------------

The native loader's external-data I/O workers are separate from the session's
CPU executor. They do not constitute a runtime preparation scheduler.

Explicit preparation before inference and overlap with model loading are
tracked by `issue #5382 <https://github.com/xadupre/onnx-light/issues/5382>`_.
See :ref:`l-design-runtime` and the C++ API pages for
:doc:`../../api/cpp/onnx_core/runtime/runtime_session` and
:doc:`../../api/cpp/onnx_core/runtime/value_store`.
