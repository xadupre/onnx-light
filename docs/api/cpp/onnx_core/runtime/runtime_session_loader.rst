runtime_session_loader.h
========================

``LoadedRuntimeSession(path, context, session_options, load_options)`` synchronously
loads an owning, reusable session. The wrapper retains the parsed ``ModelProto``
until after its ``RuntimeSession`` and kernels are destroyed. Use ``session()``
to run inference and ``model()`` for read-only inspection. Register operator
kernels as for ordinary ``RuntimeSession`` construction.

Loading first parses the model without external payload reads and reserves all
source bytes. External attribute/function and nested graph payloads load before
kernel resolution, because function binding and custom factories can copy them.
Root initializers referenced by quantization annotations also load before session
construction, which builds the shared parameter catalogue. Their external-data
metadata remains intact until catalogue validation has completed.
The finalized graph then declares preparations with the session's selected tuning.
A single dedicated I/O worker reads one external root initializer ahead while the calling thread
seeds the previous initializer and executes ready declarations using the shared
CPU executor. I/O never occupies CPU-executor workers, and only the caller mutates
the runtime context and preparation store. Set ``overlap_preparation=false`` for
the synchronous serial baseline, which reads all external payloads before any
preparation. In overlap mode, ready inline initializers can prepare while the
worker reads the first external payload.
Each loaded source wakes only its indexed preparation consumers; completed
dependencies wake their dependents. Unavailable declarations are not rescanned
for every unrelated weight.

Initializer names installed temporarily during loading are removed on return,
including on failure; pre-existing context values are preserved. Subsequent
``Run`` calls seed ordinary initializer views from the owned model. Such views,
including exported initializer-backed outputs, must not outlive the wrapper.
Serialize access to the session/context and any explicitly shared preparation
store as for ordinary sessions.
Caller overrides must be installed before construction; initializer names
also declared as graph inputs remain overridable after loading.

Owning pipelines
----------------

``LoadedRuntimePipeline(path, context, session_options, load_options)`` loads a
``PipelineProto`` with a required prefill model and an optional embedding model.
A separate decode model is rejected, matching native ``Generate`` support.
The owner retains both models and their prepared stage sessions. It exposes
``pipeline()`` for read-only inspection, ``prefill_session()`` and the nullable
``embedding_session()`` for direct stage access.

All stages are inspected and their combined payload reservation is admitted
before any external read. Stages then load sequentially, embedding first, using
the same bounded reader/preparation machinery as single models. At most one
I/O worker and one queued root tensor exist across the pipeline. Serial mode
reads a stage's payloads before preparing that stage. The shared memory limit
includes all stages' retained sources, conversion reserves, function copies,
catalogues and workspaces, plus all already-prepared values in the shared store.
``statistics()`` aggregates these counters across stages; peak counters are
pipeline-wide maxima rather than sums of independent stage peaks.
``kComplete`` is emitted once, after all stages have loaded.

Conversion reserves and catalogue construction workspace allowances are
intentionally summed across stages, even though stages load sequentially.
Admission computes one fixed pipeline-wide reservation before any external
read and keeps it throughout loading. These reservations are not released
between stages to increase the shared preparation budget. This conservative
policy can reject a pipeline that would fit with peak-based workspace reuse;
the temporary counters do not measure simultaneously live scratch memory.

Use the owning overload to preserve loading-time session settings and the shared
static preparation store automatically:

.. code-block:: cpp

    #include "onnx_core/runtime/generation.h"
    #include "onnx_core/runtime/runtime_session_loader.h"

    RuntimeLoadOptions loading;
    loading.overlap_preparation = true;
    loading.memory_limit = 8ULL * 1024 * 1024 * 1024;
    LoadedRuntimePipeline loaded("pipeline.onnx", context, session_options, loading);
    Tensor tokens = Generate(loaded, context, feeds);
    // Explicit generation options replace the pipeline's stored sampling settings.
    Tensor other = Generate(loaded, context, feeds, generation_options);

Each generation call creates fresh invocation sessions and persistent state;
requests do not inherit another request's cache. Static representations published
in the shared store, such as packed ``Gemm`` weights, are reused without repacking.
The owning overload temporarily installs a shared CPU executor for all stages,
even when the supplied context has none. It inherits an existing context executor
unless the loading-time options explicitly select another CPU policy, and
restores the caller's executor on return or failure.
Kernel-local initialization and unresolved runtime preparations still occur in
the invocation sessions, outside the loading budget. For low-level callers,
``session_options()`` exposes the retained options including the shared store;
calling ``Generate(loaded.pipeline(), ...)`` without them does not reuse that store.
Calls sharing the loader's context or static preparation store must not run
concurrently. Different contexts do not make a shared store safe for concurrent
access. Source views and initializer-backed outputs must not outlive the owner.
The same payload restrictions, callback contracts and path protections described
below apply to every stage.

Admission and failures
----------------------

``memory_limit`` bounds conservative admitted payload bytes: all retained source
tensor payloads, a conversion reserve at least as large as the source payloads,
retained root function-binding payload copies, shared quantization catalogue
buffers and their construction workspaces, and declared preparation output and
temporary work. It excludes proto metadata/parser allocations, CPU pools, allocator
free capacity and inference allocations; it is not a process-RSS limit.
Models that cannot be admitted fail rather than falling back to unbounded loading.
Ready declarations, including source-free work in a model with no tensors,
finish before construction returns. Declarations requiring a runtime feed or
an intermediate produced by inference remain explicitly pending. The session
waits until those producers have executed and establishes readiness before
each affected consumer; stale context intermediates do not satisfy that barrier.
External tensors may omit ``length`` (or its alias ``size``). The loader derives
the payload byte count from the fixed-width element type and shape, including
packed sub-byte types, and supplies that length to readers before any payload
allocation. Reads are bounded to the admitted tensor range, starting at ``offset``
(zero by default), rather than an arbitrary remainder of the external file.
Explicit lengths must be non-negative and match the shape. Unknown/string
external payload types, invalid dimensions and overflowing ranges are rejected.
``statistics()`` reports source bytes, conversion
reserve, function-binding bytes, quantization catalogue and temporary bytes,
peak admitted bytes, external tensor count and
the one-item queue bound. Function-binding admission includes each root call's
copied body/default payloads and each bound attribute reference, including repeated
references in nested graphs. Sessions created lazily during inference remain
outside the loading budget.
Shared quantization admission inspects compact inline type descriptors before
external reads or session construction. It reserves expanded block parameters,
codebooks, common payloads and local ranges, as well as numerical conversion and
validation workspaces. Scalar numerical parameters still reserve the full
broadcast expansion. The reservation also reduces the budget available to
kernel preparations; ``prepared_bytes()`` alone does not include catalogue storage.
The conversion reserve also covers the largest serialized binding copy, including
functions registered externally in the runtime context and packed integer fields
whose wire representation can exceed their in-memory payload size.
Encoded and paged-cache initializers are currently unsupported, including in
nested graphs: the loader rejects them before external reads or declaration
rather than omitting their payloads from source admission.

Default reads use ``TensorProto::LoadExternalData`` and retain its directory
containment, symlink and hard-link protections. An optional ``external_reader``
is a trusted replacement for testing or application-controlled I/O; it must
honor the declared length, implement equivalent security for untrusted files,
and change only the supplied tensor's payload. Callbacks must not mutate the
source topology, runtime context or preparation store. Trace callbacks receive
steady-clock nanosecond timestamps and phase events on the caller or reader
thread, so shared callback state must be synchronized. They must not wait for
future loader work.
Preparation phase spans can include a no-op readiness check.
``kPreparationFinished`` additionally reports ``prepared_bytes_added``, the positive
growth in live static allocations during that phase. For ``Gemm`` packing,
nonzero growth provides evidence of actual packed-output creation; benchmarks
must exclude zero-growth phases when reporting demonstrated packing overlap.
Zero growth alone does not distinguish no-op checks from zero-sized/custom work.

The API exposes no asynchronous handle. Every return and exception path joins
the reader before releasing callbacks, tensors or model storage. Worker exceptions
propagate to the caller. In-progress external I/O is not forcibly interrupted:
custom readers must eventually return or throw.

Nested graphs and functions
--------------------------

Payload traversal includes nested graph attributes, repeated graph/tensor
attributes, sparse tensor components and model-local functions (including typed
default attributes). All payloads finish loading before return. Branches,
function bodies and other nested sessions whose bindings are not known during
top-level declaration retain lazy preparation when execution resolves them;
loading their payloads does not eagerly execute or prepare unresolved branches.

.. doxygenfile:: onnx_core/runtime/runtime_session_loader.h
   :project: onnx-light
