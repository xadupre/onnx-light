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
The finalized graph then declares preparations with the session's selected tuning.
A single dedicated I/O worker reads one external root initializer ahead while the calling thread
seeds the previous initializer and executes ready declarations using the shared
CPU executor. I/O never occupies CPU-executor workers, and only the caller mutates
the runtime context and preparation store. Set ``overlap_preparation=false`` for
the synchronous serial baseline, which reads all external payloads before any
preparation. In overlap mode, ready inline initializers can prepare while the
worker reads the first external payload.

Initializer names installed temporarily during loading are removed on return,
including on failure; pre-existing context values are preserved. Subsequent
``Run`` calls seed ordinary initializer views from the owned model. Such views,
including exported initializer-backed outputs, must not outlive the wrapper.
Serialize access to the session/context and any explicitly shared preparation
store as for ordinary sessions.

Admission and failures
----------------------

``memory_limit`` bounds conservative admitted payload bytes: all retained source
tensor payloads, an equal conversion reserve, and declared preparation output and
temporary work. It excludes proto metadata/parser allocations, CPU pools, allocator
free capacity and inference allocations; it is not a process-RSS limit.
Models that cannot be admitted fail rather than falling back to unbounded loading.
External tensors must declare an explicit non-negative ``length`` (or ``size``)
matching their fixed-width shape. Unknown/string external payload types and
missing lengths are rejected. ``statistics()`` reports source bytes, conversion
reserve, peak admitted bytes, external tensor count and the one-item queue bound.
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
