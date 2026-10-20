runtime_session.h
=================

Sessions, including subgraphs, read directly usable initializer storage from
the immutable source graph. No ownership mode is required. The graph and its
backing buffers must outlive the session and its initializer views, including
when supplied through ``SetInitializers``.

Host initializers are not copied into the execution allocator for CPU execution
(``kCPU`` or the default ``kUndefined`` device). Raw storage and native float,
double, int32, int64 and uint64 fields can be borrowed; representations requiring
decoding, including strings, still use normal tensor conversion. This host
borrowing rule does not implement transfers to another device.

Initializer views do not own the source model. C++ callers keep it alive through
all such views, including returned or retained results. Ownerless externally
borrowed buffers cannot be retained. Python bindings retain the source model
when exporting initializer views.

Kernel preparation
------------------

Construction fixes a non-null ``ValueStore`` owner, creating a private store
unless ``RuntimeSessionOptions::value_store`` explicitly supplies one.
Preparation separates declaration from execution:

* ``DeclarePreparations(context)`` resolves and tunes the planned kernels and
  records their immutable work without seeding or reading initializer payloads.
  A ``KernelPreparation`` contains its metadata in ``info``, a
  ``KernelPreparationInfo`` describing source inputs, a representation, output
  slots and dependency indices, alongside a memory estimator and an execution
  callable.
* ``ExecutePreparations(context, memory_limit)`` executes ready declarations
  whose dependencies and source tensors are available and whose estimated peak
  fits the budget. It does not load or seed missing initializers.
  The overload taking ``changed_sources`` schedules only their indexed
  consumers, newly satisfied dependencies and work unblocked by a larger budget.
  Source hints never bypass availability, intermediate-provenance or tuning
  validation. ``preparation_readiness_checks()`` reports candidate inspections.
* ``Prepare(context)`` seeds available initializers and executes ready work
  without running inference or requiring example feeds.
* ``pending_preparations()`` exposes metadata-only diagnostics for deferred
  work, including unavailable sources and budget-limited work. Diagnostics
  copy the declaration's ``info`` and add the session index and current tuning
  status. They contain no execution callables, remain readable after session
  destruction, and do not force deferred branches to run.
  ``EstimatePreparationMemory(context, index)`` estimates a declaration by its
  diagnostic index without exporting its session-borrowing callbacks.
  ``awaits_runtime_tuning`` distinguishes work whose input type is not yet
  known well enough to resolve its tuning profile.

Kernel declaration callables borrow their kernel/model state and remain
session-owned. Session entry points enforce dependency, readiness, and
admission checks before execution.
Unknown input types defer tuning rather than selecting an undefined-type
profile. Resolution uses the declaration-time registry snapshot once the
actual input becomes available, before packing or the affected inference
consumer executes. A later registry update does not change that snapshot.

An explicit preparation entry point permits loading and packing before the
first real input arrives:

.. code-block:: cpp

   RuntimeSession session(model); // model and its buffers remain alive
   RuntimeContext context;
   session.Prepare(context);      // no guessed input tensor or inference
   context.Set("x", real_input);
   session.Run(context);

Existing callers can retain lazy initialization:

.. code-block:: cpp

   RuntimeSession session(model);
   RuntimeContext context;
   context.Set("x", real_input);
   session.Run(context);          // initializes and prepares before replay
   context.Put("x", next_input, RuntimeEventKind::kInput);
   session.Run(context);          // reuses resolved kernels and packed values

Immutability and availability are separate facts. An external initializer can
be known immutable before its bytes are loaded; this permits declaration, not
early payload access. ``Gemm`` declares packing for a constant ``B`` initializer.
An initializer that is also a graph input, or a value supplied by the caller
before initialization, remains overridable and is not statically prepared.
Missing dynamic inputs are checked by inference, not invented by preparation.
Initializer overrides must be supplied before kernel initialization, whether
initialization starts through declaration, explicit preparation, or the first
inference. Initializers exposed as graph inputs remain mutable throughout.

Each entry point installs the session's store into the ``RuntimeContext``; it
never adopts a context's previous store. ``If``, ``Loop``, ``Scan``,
``SequenceMap``, and model-local function sessions inherit the parent's owner.
Their sessions remain deferred until their branch, iteration, or function call
is entered: preparing the root does not execute control flow, bind fictitious
function arguments, or eagerly prepare every possible branch. Immutable
captures and immutable function arguments propagate when those child sessions
are created.

Ready publication occurs only after a preparation succeeds. A preparation
failure is sticky: subsequent preparation or execution rethrows it rather than
retrying partially completed work. Calls on one session/context, and calls on
sessions sharing a store, must be serialized.

Declarations that depend on a node-produced value remain deferred during
explicit preparation. Inference executes their producers first, then drains
ready work and enforces readiness immediately before each affected consumer.
A same-named intermediate already present in the context is not evidence that
its producer has executed. Missing sources or incomplete dependencies prevent
the consumer from running. Once all declarations are complete, warm replay
does not redispatch preparation work.

A kernel binds a store-specific ``ValueStore::Slot`` with a stable integer
index during initialization. After publication, ``BindForRead`` validates it
once and returns a store-pointer/index binding. ``Gemm`` retains this binding
for direct warm reads: ordinary ``ExecutionPlan`` replay does not wait on
readiness, acquire a preparation mutex, or repeat a string-keyed lookup.
Repeated nodes using the same immutable matrix and packing representation
share the published preparation rather than repacking it.
``RuntimeSession::prepared_bytes`` reports live buffer sizes in the static
store's allocators, including allocated but unpublished values and preparations
created by nested sessions. It excludes retained free capacity, execution/I/O
arenas. Explicitly sharing a store
between otherwise independent sessions also shares this accounting: it is
not a per-session partition. All sessions sharing one store must serialize
their runs. Contexts may keep the owner alive after a session is destroyed.

``value_store()`` returns a reference, available even before the first run.
Use ``value_store_owner()`` when passing shared ownership to options or a
nested session. Const session/context access returns ``const ValueStore&``
or ``std::shared_ptr<const ValueStore>``; initialization-time mutation and
sharing a mutable owner require a non-const session/context.

The store does not itself schedule asynchronous loading, track generations,
or evict prepared values. Ordinary preparation calls are synchronous.

Owning external-weight loading
-----------------------------

``runtime_session_loader.h`` provides ``LoadedRuntimeSession`` for orchestrated
loading. Unlike a borrowing ``RuntimeSession``, this wrapper owns the parsed
``ModelProto`` and destroys its session before releasing the model. Keep the
wrapper alive while using its session or any initializer-backed result views.
Its ``model()`` accessor is const; loaded source payloads remain immutable.

.. code-block:: cpp

   #include "onnx_core/runtime/runtime_session_loader.h"

   RuntimeContext context;
   RuntimeLoadOptions loading;
   loading.overlap_preparation = true; // false selects serial read/prepare
   loading.memory_limit = 256 * 1024 * 1024;
   LoadedRuntimeSession loaded("model.onnx", context, {}, loading);
   context.Set("x", real_input); // only after loading/preparation
   loaded.session().Run(context);

The loader parses metadata and finalizes external attribute/function and
nested-graph payloads before kernel resolution. Root-session preparation is
then declared without touching unavailable external initializer payloads.
Overlap mode uses one I/O worker and a one-tensor look-ahead: reading the next
weight can overlap seeding and packing the previous weight on the caller's CPU
executor. Serial mode reads all external payloads before preparation on the
caller, with no I/O worker.
Only the caller mutates ``RuntimeContext`` and ``ValueStore``. Nested
external payloads are loaded, but this does not eagerly execute or prepare
their deferred branch/function sessions.
Successful construction completes all ready root preparations; an
insufficient admission budget raises an exception instead of silently
postponing that work until inference. This root readiness guarantee does not
claim that deferred child sessions have already been initialized. Root
declarations requiring runtime feeds or node-produced values remain visible
through ``pending_preparations()`` and complete before their inference consumers.

``RuntimeLoadStatistics`` reports source bytes, a conservative conversion
reserve, logical peak admitted bytes, external tensor count, and bounded queue
occupancy. ``memory_limit`` is an admission budget, not an operating-system RSS
limit: source payloads remain owned by the model, and persistent packed buffers
remain in the store. Proto metadata, parser allocations, executor pools, and
inference allocations are outside this budget. The loader requires explicit
external tensor lengths and validates them against fixed-width types and
dimensions before admitting reads; it does not accept unbounded file-remainder
reads.
Peak admission includes the source and temporary work of each executed
declaration, including dependent preparations executed in the same drain.
``RuntimeSession::preparation_peak_bytes()`` exposes the executor's peak
preparation reservation, excluding the loader's separate source/conversion
reservation.

The optional ``trace`` callback receives ``RuntimeLoadEvent`` phase,
tensor name, and ``steady_clock`` nanosecond timestamp. Events distinguish
declaration, read start/finish, preparation start/finish, and completion.
Callbacks can run on the caller or I/O thread and must be thread-safe; they must
not mutate the context/model or block waiting for later loader work.
The finished preparation event reports ``prepared_bytes_added``; a positive
value witnesses actual static-buffer preparation rather than a no-ready
drain. Intersecting read intervals with those positive-growth preparation
intervals demonstrates genuine packing/I/O overlap, not merely the presence
of an overlap option.

Loading is synchronous, not a detached task or future, and exposes no
asynchronous cancellation handle. Exceptions stop further scheduling, join the
I/O worker, and propagate to the caller. Cancellation during cleanup does not
interrupt a read already in progress: cleanup waits for it to return.
A trusted ``external_reader`` override must
therefore finish or throw, honor the admitted length, and modify only the
requested tensor's raw payload.

Preparation benchmark
---------------------

The self-contained ``bench_session_preparation`` target generates a real
external-data FLOAT Gemm chain using standard C++ APIs. By default it creates
eight distinct 1024-by-1024 matrices and sixteen Gemm nodes, reusing each
matrix twice. No helper script, downloaded model, or example inference feed is
needed for loading/preparation.

.. code-block:: sh

   cmake -S . -B build -DONNX_LIGHT_BUILD_BENCHMARKS=ON
   cmake --build build --target bench_session_preparation --parallel 6
   ./build/bench_session_preparation --directory bench-preparation-data --iterations 3

``--directory`` must name a new directory; generated files are removed on
exit. Without this flag, the benchmark creates its named directory beneath
``std::filesystem::temp_directory_path()``. ``--weights`` and ``--dimension``
configure the workload (at most 512 MiB of source weights). Mode order
alternates across iterations.
The benchmark reports last-external-read latency (``model_load_ms``, including
intervening packing), total load-to-ready latency (``load_ready_ms``), summed
preparation time
(``packing_ms``, including dispatch/publication), and first and warm inference
latencies. The first inference feed is supplied only after the loader returns.
It checks finite, numerically equivalent serial/overlap outputs, one shared
packing per distinct matrix, and unchanged store size/output on warm replay.

Packing time and overlap include only intervals whose finished event reports
positive ``prepared_bytes_added``, excluding no-ready calls; the summed delta
must match the final packed allocation size.

Logical admission peak and queue occupancy are reported separately from
process-lifetime peak RSS, available on Linux/macOS (zero otherwise). RSS
includes model generation and earlier measurements, so it is not an isolated
per-mode peak. Read/preparation interval intersections and an example timestamp
pair document observed overlap. Generated files are likely already in the OS
page cache, tracing adds overhead, and these measurements do not represent
cold-disk loading or imply a guaranteed speedup.

.. doxygenfile:: onnx_core/runtime/runtime_session.h
   :project: onnx-light

See :doc:`runtime_session_loader` for the owning loader API reference.
