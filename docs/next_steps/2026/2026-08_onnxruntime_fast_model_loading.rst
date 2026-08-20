.. _l-next-steps-model-loading:

Exploiting ``onnx-light`` fast loading in ``onnxruntime``
=========================================================

:Date: 2026-08

**ready to implement**

Objective
+++++++++

`microsoft/onnxruntime#29723
<https://github.com/microsoft/onnxruntime/pull/29723>`_ proves that
:epkg:`onnxruntime` (ORT) can replace protobuf and the standard :epkg:`onnx`
library with ``onnx-light``. It is an integration milestone, not yet a loading
optimization: ORT calls the protobuf-compatible ``onnx-light`` methods, then
continues graph resolution, external-weight materialization, kernel selection,
and prepacking through its existing runtime.

The objective of this plan is to minimize time and memory from a serialized
model to the first useful inference, with Qwen3 INT4 CPU inference as the first
large-model target. It covers two consumers:

* the native ``onnx-light`` runtime, where the loader, graph resolver, prepared
  execution plan, and kernels can share one ownership model;
* ORT built with ``onnxruntime_USE_ONNX_LIGHT=ON``, where immediate parser
  improvements can be drop-in but weight and prepared-session improvements
  require an explicit ORT integration.

The target is not the smallest reported ``load()`` number. A memory map can
make parsing appear almost free while moving page faults into the first
inference. The target is a truthful reduction in time to first token, bytes
read and copied, peak private memory, and repeated preparation work.

Relationship to the completed ORT integration
++++++++++++++++++++++++++++++++++++++++++++++

This is the second roadmap after the completed
:ref:`l-next-steps-ort-onnx-light` compatibility plan. The first roadmap made
``onnx-light`` a build-time replacement for protobuf and the standard
``onnx`` library inside ORT. It deliberately preserved ORT's existing loading
pipeline.

This roadmap starts where that compatibility work stops. It uses the native
loader's source ownership, mapped ranges, selective payload materialization,
prepared tensors, and asynchronous preparation instead of limiting
``onnx-light`` to protobuf-compatible method calls. The implementation has two
coordinated tracks:

* **drop-in ORT improvements**, where direct parsing can accelerate the
  existing ``onnxruntime_USE_ONNX_LIGHT`` build without an ORT source change;
* **ownership-aware integration**, where ORT or the native runtime explicitly
  consumes mapped, lazy, or kernel-ready payloads and preserves their owners.

The tracks share formats, identities, benchmarks, and failure semantics, but
they do not force ORT and the native runtime to use the same graph resolver or
execution planner.

Opportunity map
+++++++++++++++

The plan treats fast parsing as the entry point, not the final optimization.
Every opportunity exposed by the native loader belongs to one measurable
layer:

.. list-table::
    :header-rows: 1
    :widths: 20 34 28 18

    * - Layer
      - ``onnx-light`` capability
      - Intended result
      - Main PR
    * - Main protobuf
      - Direct bounded array/file-descriptor streams, adaptive blocks, optional
        owned mappings
      - No whole-file staging copy and lower parse latency
      - PR02
    * - External data
      - Validated source descriptors, shared mappings, lazy reads, explicit
        ownership
      - One parse, one owner per source, no redundant payload copy
      - PR03
    * - Runtime state
      - Existing owned ``ByteSpan`` payloads plus persistent runtime
        initializer tensors
      - Names, shapes, typed payloads, and borrowed views are materialized once
        per session
      - PR04
    * - Graph resolution
      - Metadata-first parsing and recoverable exact source ranges
      - Dead or superseded weights are never read
      - PR05
    * - Kernel preparation
      - Stable payload identities and kernel-specific compiled tensors
      - Compatible packed weights bypass portable reads and prepacking
      - PR06
    * - Prepared execution
      - :ref:`l-next-steps-prepared-execution`, lazy payloads, and
        session-scoped I/O/preparation dependencies
      - Early blocks run while later blocks are still prepared
      - PR07
    * - ORT ownership
      - Mapping lifetime tokens and final-destination read descriptors
      - Eligible ORT initializers borrow safely; others avoid intermediate
        buffers
      - PR08a + PR08b
    * - Measurement
      - Phase events, byte/copy counters, page-fault attribution, and cache-state
        labels
      - Parser wins cannot hide deferred I/O or first-token regressions
      - PR01

These layers are intentionally end to end. A change is not considered a
loading improvement when it only moves copies, page faults, graph work, or
prepacking past the measured boundary.

Scope and non-goals
+++++++++++++++++++

The first implementation targets:

* Linux and Windows local files;
* ONNX protobuf models with inline or external tensor data;
* Qwen3-like decoder models, including INT4 weights;
* CPU execution;
* both cold and warm page-cache conditions;
* one model session reused for many invocations.

Remote object stores, encrypted models, accelerator placement, weight
offloading between tokens, and distributed loading are later extensions. ORT
``.ort`` flatbuffers remain an essential comparison baseline, but implementing
the complete ORT flatbuffer format is not a prerequisite for the first
``.onnx`` improvements.

What PR #29723 changes -- and what it does not
++++++++++++++++++++++++++++++++++++++++++++++

With ``onnxruntime_USE_ONNX_LIGHT=ON``, ORT's file-path load opens a file
descriptor and calls ``ModelProto::ParseFromFileDescriptor``. In the current
``onnx-light`` compatibility implementation that method:

1. reads 4096-byte chunks into a growing ``std::string``;
2. constructs a ``StringStream`` over the complete string;
3. parses the model from that second pass.

This path does not use ``ParseOptions``, adaptive block sizes, parallel raw-data
reads, or mmap ownership. ``ParseFromArray`` similarly constructs a temporary
``std::string`` before parsing. These are compatibility-adapter bugs, not
parser requirements: the native parser already accepts bounded
``BinaryStream`` implementations. PR #29723 makes them urgent because ORT now
routes model loading through these adapters.

External weights are a different path. The ONNX protobuf contains only
``location``, ``offset``, and ``length`` metadata; ORT resolves and
materializes those bytes later while constructing its session. Making
``ParseFromFileDescriptor`` faster therefore cannot, by itself, make a
multi-gigabyte Qwen session ready. The benchmark must show how much time belongs
to the main protobuf, external data, graph resolution, kernel initialization,
and prepacking before deciding where to optimize next.

Current ``onnx-light`` baseline
+++++++++++++++++++++++++++++++

The native loader already provides useful building blocks:

* ``FileLoadMode.AUTO``, ``MMAP``, and ``IFSTREAM``;
* shared ownership for mapped main files and external-weight files;
* borrowed ``ByteSpan`` payloads and ``TensorFromProto`` zero-copy views over
  ``raw_data``;
* parallel reads of large raw-data blocks;
* a raw-data ownership callback;
* metadata-only parsing with ``skip_raw_data``;
* optional page touching for honest mmap measurements;
* tensor-size and recursion limits.

The missing pieces are mostly orchestration and consumer ownership:

* Python external-data auto-discovery calls ``_find_external_location()``,
  which parses a metadata-only model before the real load;
* ``onnxl.load()`` defaults to ``num_threads=-1`` and
  ``min_block_size=0``; the parser can create a thread pool even when the file
  has too little eligible work to repay that overhead;
* ``FileLoadMode.AUTO`` currently selects a buffered stream, and true
  single-file no-copy loading requires an explicit ``MMAP`` choice;
* the regular runtime recreates initializer ``Tensor`` wrappers when a
  ``ReferenceEvaluatorRunner`` invocation seeds a cleared
  ``RuntimeContext``; payload bytes are borrowed, but names and shapes are
  rebuilt;
* the evaluator constructor does not build its ``ExecutionPlan`` or
  ``RuntimeSession``. ``EnsureSession()`` is called on the first run, so a
  benchmark that times only construction does not measure a prepared session;
* mutable ``GraphBuilder`` imports may copy initializer proto objects even
  when the source payload has safe shared ownership;
* all payloads are normally loaded before graph transformations and kernel
  requirements determine which initializers and physical layouts are needed;
* portable weights may be prepacked again for every new process.

This plan connects, rather than duplicates, three existing designs:

* :ref:`l-next-steps-model-resolution` decides graph and payload liveness
  before large reads;
* :ref:`l-next-steps-compiled-tensor` persists compatible kernel-ready weight
  representations;
* :ref:`l-next-steps-prepared-execution` schedules session-scoped reads and
  preparation and can overlap them with the first invocation.

Benchmark contract
++++++++++++++++++

Fixtures
^^^^^^^^

The benchmark suite must contain real serialized files:

* a small deterministic model for CI correctness and phase instrumentation;
* a Qwen3-shaped reduced fixture whose tensor count, external-data layout,
  alignment, and INT4 packing are representative;
* at least one complete supported Qwen3 export used in scheduled performance
  runs;
* one large inline-data model to expose the file-descriptor and array parsing
  paths;
* one model whose external tensors span multiple files and nested subgraphs.

Randomly materializing metadata-only initializers before the measured section
does not reproduce file I/O, mmap faults, external-data discovery, or packed
cache loading. Such a fixture remains useful for isolated graph-construction
microbenchmarks but is not a loading benchmark.

Comparisons
^^^^^^^^^^^

Every Qwen run records the same model identity, machine, storage device,
compiler, build type, thread policy, affinity, and runtime commit. The minimum
comparison matrix is:

.. list-table::
    :header-rows: 1
    :widths: 25 18 22 35

    * - Consumer
      - Format
      - Weight state
      - Purpose
    * - ORT + protobuf
      - ``.onnx``
      - portable
      - upstream compatibility baseline
    * - ORT + ``onnx-light``
      - ``.onnx``
      - portable
      - effect of PR #29723 and parser changes
    * - ORT
      - ``.ort``
      - ORT-optimized
      - strongest ORT loading baseline
    * - native ``onnx-light``
      - ``.onnx``
      - portable
      - native ownership and resolution path
    * - native ``onnx-light``
      - ``.onnx`` plus prepared store
      - compatible packed cache
      - final repeat-start target

ORT graph optimization levels and offline-optimized artifacts are reported
explicitly. Comparing an unoptimized portable ORT session with a prepared
``onnx-light`` session would not establish parity.

Measured phases
^^^^^^^^^^^^^^^

One monotonic trace identifier follows a load through all layers. At minimum,
the trace reports:

.. code-block:: text

    open sources
    parse model metadata
    discover and validate external locations
    resolve graph and payload liveness
    map/read portable payloads
    resolve compatible compiled payloads
    build execution plan
    select kernels
    read compiled payloads or prepack portable payloads
    session ready
    first prefill
    first decode token
    fully prepared

The public summary exposes unambiguous milestones:

``T_metadata``
    Model structure and payload descriptors are valid and available.

``T_resolved``
    Transformations, kernel requirements, compiled-cache selection, and the
    required payload manifest are frozen.

``T_session_ready``
    The documented minimum input can run without additional graph planning or
    kernel selection. The state may still contain declared asynchronous
    preparation tasks.

``T_first_token``
    The first decode token has completed. All page faults, just-in-time
    preparation, and waits paid by that token are included.

``T_fully_prepared``
    Every mandatory session-scoped read and prepack task has completed.

Resource counters
^^^^^^^^^^^^^^^^^

For each milestone, report:

* wall and CPU time;
* process RSS, PSS when available, and peak private bytes;
* file-backed versus anonymous resident bytes;
* major and minor page faults;
* physical I/O bytes and logical bytes requested;
* bytes copied by the loader and bytes borrowed or mapped;
* temporary and persistent allocations;
* worker count, tasks submitted, and time waiting for I/O or preparation.

Mmap runs are measured twice: untouched and with
``touch_raw_data_pages=True``. The untouched number describes address-space
construction, not weight readiness.

Cache states
^^^^^^^^^^^^

Each performance result is labelled as one of:

* **cold storage**: a separate process, with best-effort cache eviction
  recorded and verified through physical-I/O counters;
* **warm page cache**: a separate process after the files have been read, but
  with no runtime objects reused;
* **warm prepared cache**: page cache warm and compatible compiled tensors
  already persisted;
* **same session**: repeated inference, excluded from model-load statistics.

Global operating-system cache drops are not part of normal CI. A benchmark
must not claim a cold run unless counters confirm that the required bytes came
from storage.

Ownership contract
++++++++++++++++++

Every payload handed to a runtime tensor has one explicit state:

.. code-block:: text

    owned bytes
    borrowed mapped bytes + shared mapping owner
    borrowed caller bytes + caller lifetime token
    lazy descriptor + source owner
    compiled payload + prepared-store owner

A raw pointer without one of these ownership paths is invalid. The owner must
outlive the model, every initializer view, all kernel prepack tasks, and any
kernel that retains the bytes.

Model metadata and payload storage have separate lifetimes. A consumer may
copy or transform graph metadata without copying immutable tensor bytes.
Conversely, releasing a transformed ``ModelProto`` must not invalidate payloads
retained by a prepared session.

For ORT, mmap-backed initializers are opt-in. ORT may borrow an immutable mapped
range only when its tensor and kernel contracts do not require writable memory,
different alignment, or another physical layout. Otherwise ``onnx-light``
reads directly into the final ORT allocation or the final prepacked
destination; it must not introduce an intermediate whole-tensor buffer.

Implementation sequence
+++++++++++++++++++++++

PR01 -- truthful loading benchmark and trace
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Repository:** ``xadupre/onnx-light``

Add one benchmark driver shared by the native and ORT comparison scripts. It
creates the phase trace and resource report defined above, runs each sample in
a separate process, and stores machine-readable results.

Correct the Qwen initialization example so its labels distinguish:

* evaluator construction;
* actual execution-plan/session construction;
* first inference;
* complete file load.

Add a native ``prepare()`` entry point if needed to measure plan and kernel
construction without executing a token. ``prepare()`` must perform the work
its name promises; it cannot merely allocate an evaluator wrapper.

Acceptance:

* CI validates phase ordering and counter consistency on the small fixture;
* scheduled runs cover the reduced and full Qwen fixtures;
* the report distinguishes main protobuf bytes, external bytes, copied bytes,
  mapped bytes, and deferred page faults;
* ORT + protobuf and ORT + ``onnx-light`` use otherwise identical builds.

PR02 -- fix compatibility-parser whole-file staging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Repository:** ``xadupre/onnx-light``

This PR fixes a performance and peak-memory bug. The compatibility adapter was
implemented by accumulating the descriptor into a convenient ``std::string``
and then reusing ``ParseFromString``; that shortcut is functionally correct but
defeats the streaming parser and creates an allocation proportional to the
complete model.

Change ``ParseFromFileDescriptor`` to parse directly from ``FdReadStream``
instead of first accumulating the complete file in a ``std::string``.

For a seekable regular file:

* use ``fstat``/the platform equivalent to determine remaining bytes;
* choose an adaptive read block rather than one syscall per 4096 bytes;
* preserve the descriptor's current offset and protobuf-compatible EOF/error
  behavior;
* add an overload accepting ``ParseOptions``;
* optionally use an owned file mapping when no-copy was explicitly requested.

For pipes and other non-seekable descriptors, retain bounded streaming with no
seek or mmap assumption. ``FdReadStream`` buffers are unstable across
``Next()`` calls, so raw payloads from that path are copied unless another
owner is explicitly installed.

Change ``ParseFromArray`` to construct a bounded ``StringStream`` directly over
the supplied range instead of first copying it into ``std::string``. The
protobuf-compatible overload still cannot retain a borrowed payload past the
call because its caller supplies no lifetime token; a separate ownership-aware
API is required for no-copy.

Acceptance:

* no allocation proportional to the whole main file exists before parsing;
* truncated files, malformed varints, non-zero descriptor offsets, pipes, and
  read errors have regression tests;
* the direct path preserves parser size and recursion limits;
* parsing a large inline fixture is at least 20% faster or the PR documents,
  from counters, why parsing is already I/O-bound;
* representative small models regress by no more than 3%.

This PR benefits the existing #29723 ORT integration without requiring an ORT
source change.

PR03 -- fix double parsing and make external I/O adaptive
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Repository:** ``xadupre/onnx-light``

This PR fixes a loader-orchestration bug. When ``load_external_data=True`` and
the caller does not provide ``location``, ``load()`` currently calls
``_find_external_location()``. That helper parses a temporary metadata-only
``ModelProto``, returns the first external location, and discards the model.
``load()`` then constructs another ``ModelProto`` and parses the main protobuf
again for the real load. Raw tensor bytes are skipped during discovery, but
the main file and protobuf structure are still traversed twice.

Replace Python's ``_find_external_location()`` preliminary parse with one of
these equivalent one-pass contracts:

* parse the main model once, retain external descriptors, then hydrate selected
  descriptors on the same model; or
* let a path-aware stream open validated ``external_data.location`` files when
  each ``TensorProto`` is complete.

The implementation must support multiple locations and nested graphs. A
single guessed "primary" weights file is not sufficient. Every location is
resolved relative to the model directory and passes the existing traversal,
symlink, offset, length, and file-size checks before any read or map.

Make parallelism lazy and evidence-based:

* do not construct a pool until an eligible block is submitted;
* keep small metadata and tiny tensors on the caller;
* calibrate the minimum block size and useful worker count by storage class
  and cache state;
* cap outstanding bytes as well as task count;
* never use the runtime execution pool for blocking file I/O.

``num_threads=-1`` remains an automatic policy, not "use every logical CPU".
The resolved worker count and threshold appear in the trace.

Acceptance:

* auto-discovered external data causes exactly one protobuf parse;
* one mapping/read owner is shared by all views into the same file;
* no pool is created for a metadata-only or wholly mmap-borrowed load;
* automatic parallel loading beats or matches single-thread loading on the
  full fixture and does not regress the reduced fixture by more than 3%;
* failures identify the tensor, resolved source, range, and reason.

PR04 -- cache runtime initializer tensors in the prepared session
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Repository:** ``xadupre/onnx-light``

Payload ownership already exists: inline bytes belong to the ``ModelProto`` and
mapped bytes retain their mapping owner. The bug is repeated runtime
materialization. ``ReferenceEvaluatorRunner::Run`` clears its
``RuntimeContext``, iterates over every graph initializer, and calls
``TensorFromProto`` again. Raw payloads are borrowed again, names and shapes
are rebuilt, and typed repeated fields may be copied again.

Introduce a persistent initializer tensor store. It converts each live
``TensorProto`` or lazy payload descriptor into a runtime ``Tensor`` once,
retains or references the existing ownership token, and lends immutable tensor
views to invocation contexts. Clearing a ``RuntimeContext`` must not rebuild
initializer names, shapes, payload wrappers, or typed payloads on every token.

The prepared session owns:

.. code-block:: text

    resolved model metadata
    source and mapping owners
    initializer tensor descriptors
    execution plan
    selected kernel descriptors
    completed prepared objects

Graph import gains a move/borrow path for immutable initializers. Transforming
nodes and value metadata must not imply deep-copying large payloads. A mutable
transformation that really changes a tensor creates a new owner and invalidates
the old compiled-cache key.

Acceptance:

* payload addresses and owners remain stable across repeated invocations;
* no initializer wrapper or payload allocation is proportional to token count;
* ``T_session_ready`` includes plan construction and kernel selection;
* releasing the source Python model after session creation is safe when the
  session owns the source, and fails explicitly when a caller-lifetime token
  is required but absent.

PR05 -- resolve before reading large payloads
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Repository:** ``xadupre/onnx-light``

Implement the first executable slice of
:ref:`l-next-steps-model-resolution`:

1. parse graph metadata and exact payload descriptors;
2. apply the bounded transformation pipeline;
3. remove dead initializers recursively;
4. select kernels and their physical weight requirements;
5. freeze ``RequiredPayloadManifest``;
6. only then submit large reads.

External data already supplies source ranges. Inline ``raw_data`` requires the
parser to retain its exact source offset and length in a seekable model source;
``skip_raw_data`` alone is not enough if the bytes cannot later be recovered.
Small constants needed for shape inference may be loaded under an explicit
threshold.

Acceptance:

* a fixture with dead and fused weights reads neither dead portable payloads
  nor superseded payloads;
* selected byte ranges exactly match the frozen manifest;
* model transformations cannot mutate payload requirements after reads begin;
* fallback to eager loading remains available until all supported
  transformations can declare their payload effects.

PR06 -- load compiled tensors before portable weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Repository:** ``xadupre/onnx-light``

Implement the CPU subset of :ref:`l-next-steps-compiled-tensor`. Resolution
checks source digest, CPU/ISA, runtime, kernel layout, and format compatibility
using metadata before selecting a prepared payload.

On a hit:

.. code-block:: text

    read/map compatible packed payload -> publish PreparedKey

On a miss:

.. code-block:: text

    read portable source -> prepack -> publish PreparedKey
                                      -> persist atomically in background

The portable source remains the correctness fallback but is not read on a
valid hit unless digest verification requires content that is not already
represented by trusted model identity metadata. Cache writes use a temporary
file, checksum, ``fsync`` where required by policy, and atomic rename. A
crashed or incompatible cache is a diagnosed miss, never a partially valid
success.

Start with the Qwen kernels whose prepacking and weight volume dominate
session creation. Do not add a generic cache layer before those kernels expose
stable prepared-layout identifiers.

Acceptance:

* a compatible hit performs no portable-weight read or prepack for that entry;
* a miss produces the same numerical result and a reusable atomic cache entry;
* stale, corrupt, wrong-ISA, and wrong-kernel-layout entries are rejected;
* warm prepared-cache ``T_first_token`` improves by at least 20% on the reduced
  fixture, or profiling proves that preparation is not on its critical path.

PR07 -- integrate ``prepared_execution`` with first inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Repository:** ``xadupre/onnx-light``

Connect the resolved payload manifest to
:ref:`l-next-steps-prepared-execution`. Prioritize the embedding and earliest
decoder blocks, then read and prepare later blocks under bounded memory and I/O
admission. Execution may start when its declared ``PreparedKey`` dependencies
are ready; it never observes a partially prepared weight.

The scheduler uses distinct resource classes for file I/O and CPU prepack.
Runtime kernel workers do not block on file reads while runnable preparation or
inference work exists. Cancellation, timeout, and preparation failures reach
all dependents with the original diagnostic.

Acceptance:

* a trace proves overlap rather than merely concurrent task submission;
* first-block priority reduces ``T_first_token`` without increasing peak
  private memory beyond the configured budget;
* ``T_fully_prepared`` is deterministic and observable;
* synchronous mode remains available as the correctness reference.

PR08a -- expose the ORT payload-ownership contract
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Repository:** ``xadupre/onnx-light``

First measure ORT's existing external-data reader. If it already reads directly
into final tensor allocations, retain it for portable tensors. Do not replace a
direct path with an ``onnx-light`` intermediate buffer merely to share code.

For ranges that ORT can safely borrow, add a narrow adapter:

.. code-block:: cpp

    struct MappedPayload {
      const void* data;
      size_t size;
      size_t alignment;
      std::shared_ptr<void> owner;
      PayloadIdentity identity;
    };

The ``onnx-light`` PR defines the C++ contract, source identity, alignment
guarantees, ownership tests, and the descriptor used when a consumer must read
into its own final allocation. It does not modify ORT session state.

Acceptance:

* mapped payloads retain a shared owner and stable identity;
* ineligible ranges expose a final-destination read descriptor rather than an
  intermediate whole-tensor buffer;
* tests cover owner release, alignment, file replacement, truncated ranges,
  and concurrent views;
* the public contract does not expose ORT-specific types.

PR08b -- consume owned payloads in ORT session state
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Repository:** ``microsoft/onnxruntime``

This PR follows microsoft/onnxruntime#29723 and depends on PR08a. ORT attaches
``MappedPayload::owner`` to session state and creates an immutable CPU
initializer view only when graph optimization and the selected execution
provider permit borrowing. If ORT or an execution provider needs writable,
relocated, or packed memory, it asks the PR08a descriptor to read directly into
the final ORT allocation.

The ORT integration must compare:

* default ORT protobuf ``.onnx``;
* ORT + ``onnx-light`` portable ``.onnx``;
* ORT + ``onnx-light`` mapped initializers where eligible;
* ORT ``.ort`` format.

Acceptance:

* every borrowed ORT initializer has a session-owned lifetime token;
* ineligible tensors take a documented direct-copy path;
* ORT graph optimization and provider prepacking cannot retain dangling
  pointers;
* the ``onnx-light`` build is no slower than the protobuf build by more than
  3% at ``T_first_token`` on the Qwen fixture;
* any claimed win remains present against the ``.ort`` baseline appropriate
  to the same graph and kernels.

Tuning method
+++++++++++++

Only four loader controls are tuned initially:

* descriptor read block size;
* minimum parallel payload size;
* I/O worker count;
* maximum outstanding read bytes.

Tune them separately for cold storage, warm page cache, buffered reads, and
mmap. A result from one state is not reused silently for another. Candidate
values are evaluated on the reduced fixture, then confirmed on the full model:

.. code-block:: text

    read block:              64 KiB, 256 KiB, 1 MiB, 4 MiB
    parallel threshold:      256 KiB, 1 MiB, 4 MiB, 16 MiB
    I/O workers:             1, 2, 4, 8
    outstanding-byte cap:    16 MiB, 64 MiB, 256 MiB, 1 GiB

This grid is an experiment definition, not a permanent ABI or a promise that
all combinations remain configurable. Select the smallest policy that is
within 2% of the best median ``T_first_token`` and does not worsen p95, peak
private bytes, or physical bytes read. Persist the chosen policy with machine
and storage descriptors in benchmark output; do not create a versioned
``tuning_abi``.

Prepack scheduling is tuned only after loader I/O is stable. Vary CPU prepack
concurrency under the session execution policy and reject a setting that
improves preparation time by stealing enough CPU to worsen first-token
latency.

Security and failure semantics
++++++++++++++++++++++++++++++

All optimized paths preserve or strengthen current limits:

* external locations remain relative, normalized, and confined to the model
  directory unless the caller explicitly supplies a trusted resolver;
* symlinks, offsets, lengths, integer overflow, file replacement, and truncated
  reads are validated before a pointer is exposed;
* mmap does not bypass maximum tensor or total mapped-byte budgets;
* a source identity includes enough file metadata or content digest to detect
  replacement between resolution and materialization;
* checksums are verified before a prepared object becomes visible;
* task failures are never converted into an empty tensor, cache hit, or
  successful session;
* diagnostic messages identify phase, tensor, source, range, and underlying
  error without exposing unrelated paths.

The one-pass external loader must not reintroduce a time-of-check/time-of-use
gap. Prefer opening a validated directory handle and resolving files relative
to it on platforms that support that contract.

Success criteria
++++++++++++++++

The roadmap is complete when all of the following are true:

* the #29723 file-descriptor path has no whole-file staging allocation;
* Python external auto-discovery parses the model only once;
* large payload I/O begins only after a frozen live-payload manifest exists;
* native sessions retain initializer owners and do not rebuild initializer
  wrappers per token;
* compatible Qwen prepared weights load without reading or prepacking their
  portable counterparts;
* cold, warm-page-cache, first-token, and fully-prepared numbers are reported
  separately;
* ORT + ``onnx-light`` has no material first-token regression against the same
  ORT protobuf configuration;
* any performance claim against ORT also reports the equivalent ``.ort``
  result.

Can this beat ORT loading?
++++++++++++++++++++++++++

``onnx-light`` can plausibly beat protobuf parsing by removing staging,
retaining compact message storage, and borrowing owned ranges. That matters
only when parsing is a measurable part of session creation.

The larger opportunity is the native end-to-end path: resolve the graph before
reading dead weights, map immutable live payloads, consume exact
kernel-compatible packed tensors, and overlap later-block preparation with
early execution. This can beat a generic portable ``.onnx`` load. Beating ORT
with an optimized ``.ort`` artifact is harder and must be established by the
benchmark, not inferred from parser microbenchmarks.

For ORT itself, #29723 lets ``onnx-light`` improve the main protobuf path
immediately. It cannot independently change ORT's graph resolver, external
weight allocator, or execution-provider prepacking. Those gains require the
explicit ownership adapter in PR08a and ORT changes in PR08b.

Dependencies
++++++++++++

The strict implementation order is:

.. code-block:: text

    xadupre/onnx-light:
      PR01 benchmark and phases
      -> PR02 direct compatibility parser
      -> PR03 one-pass external load and adaptive I/O
      -> PR04 persistent runtime initializer tensors
      -> PR05 model resolution and payload liveness
      -> PR06 compiled tensor cache
      -> PR07 prepared_execution integration

      PR02 -> PR08a ownership contract
      PR03 + PR04 -> PR08a mapped ownership

    microsoft/onnxruntime:
      onnx-light PR08a -> PR08b payload consumption

PR02 and the ORT parser comparison may proceed before native prepared
execution. PR06 depends on kernel-specific stable packed layouts. PR07 is the
loading roadmap's implementation bridge into
:ref:`l-next-steps-prepared-execution`; it depends on the real session executor
from
:ref:`l-next-steps-session-execution-pools`; it must not create another hidden
CPU pool.

See also
++++++++

* :ref:`l-next-steps-ort-onnx-light` -- the compatibility milestone delivered
  by microsoft/onnxruntime#29723.
* :ref:`l-next-steps-model-resolution` -- graph and payload resolution before
  large reads.
* :ref:`l-next-steps-compiled-tensor` -- persistent compatible prepared
  weights.
* :ref:`l-next-steps-prepared-execution` -- session-scoped asynchronous reads
  and preparation.
* :ref:`l-next-steps-session-execution-pools` -- truthful session CPU
  execution policy.
