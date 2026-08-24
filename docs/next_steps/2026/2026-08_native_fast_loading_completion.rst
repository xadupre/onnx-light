.. _l-next-steps-native-fast-loading-completion:

Completing native fast model loading
====================================

:Date: 2026-08

**implemented; native PR01--PR06 are implemented**

Objective
+++++++++

This is step 3 of the fast-loading sequence. It consumes the session-scoped
preparation tasks, dependency events, bounded resources, and prepared-object
residency from :ref:`l-next-steps-prepared-execution`. The onnx-light payload
ownership contract from issue #4611 is already available. This work does not
depend on onnxruntime consuming that contract.

The objective is to connect the native loader, graph resolver, prepared tensor
cache, and prepared executor end to end. The target is time to first token and
fully prepared state, not a parser-only ``load()`` number.

Native PR01 -- adaptive external-data I/O
+++++++++++++++++++++++++++++++++++++++++

**Repository:** ``xadupre/onnx-light``

Make parallelism lazy and evidence-based. Do not create I/O workers until an
eligible read is submitted. Calibrate block size, worker count, and outstanding
bytes separately for cold storage, warm page cache, buffered reads, and mmap.
Blocking file I/O must not occupy runtime CPU executor participants.

Acceptance:

* metadata-only and wholly mapped loads create no I/O worker pool;
* automatic policy matches or beats one worker on the full fixture and
  regresses the reduced fixture by no more than 3%;
* traces report resolved workers, thresholds, physical bytes, logical bytes,
  page faults, and bytes in flight.

Native PR02 -- resolve before materializing payloads
++++++++++++++++++++++++++++++++++++++++++++++++++++

**Repository:** ``xadupre/onnx-light``

**Implemented by issue #4619.**

Implement the executable slice of :ref:`l-next-steps-model-resolution` that
freezes ``RequiredPayloadManifest`` before large reads. Concrete useful cases
include a constant ``If`` branch, unused training/debug heads, initializers
removed by graph rewrites, and portable weights replaced by a compatible
prepared object.

This PR does not claim a gain when every serialized initializer remains live.
If representative fixtures avoid no material payload bytes, fold the manifest
plumbing into Native PR03 instead of shipping a standalone performance PR.

Acceptance:

* no task reads a range absent from the frozen manifest;
* dead, superseded, and prepared-cache-replaced payload bytes are counted;
* graph transformations cannot mutate requirements after reads begin;
* eager loading remains a diagnosed fallback for unsupported transformations.

The native prepared-task boundary now carries payload identities and validates
every read against an immutable manifest. The manifest reports bytes omitted
as dead, superseded, or replaced by a prepared-cache payload; unsupported
transformations explicitly select and diagnose eager loading.

Native PR03 -- consume prepared tensors before portable weights
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

**Repository:** ``xadupre/onnx-light``

**Implemented by issue #4620.**

Implement the CPU subset of :ref:`l-next-steps-compiled-tensor`. Resolve source
digest, CPU/ISA, runtime, kernel layout, and format compatibility from metadata.
On a hit, read or map the packed payload and skip the portable source and
prepack. On a miss, read the source, prepack, publish immediately, and persist
an atomic cache entry in the background.

Acceptance:

* compatible hits perform no portable-weight read or prepack;
* misses produce identical numerical results and reusable atomic entries;
* stale, corrupt, wrong-ISA, and wrong-layout entries are diagnosed misses;
* warm prepared-cache ``T_first_token`` improves by at least 20% on the reduced
  fixture, or profiling proves preparation is not on its critical path.

The native prepared-tensor cache validates the source digest, CPU architecture
and ISA, runtime version, kernel layout, format version, payload length, and
payload digest before publication. Compatible entries publish directly without
calling the portable loader or prepacker. Diagnosed misses publish the freshly
prepared value first and atomically replace the cache entry on a background
writer. ``bench_prepared_cache_startup`` compares these two first-token paths on
the reduced fixture and enforces the 20% warm-cache target.

Native PR04 -- connect and tune first-token overlap
+++++++++++++++++++++++++++++++++++++++++++++++++++

**Repository:** ``xadupre/onnx-light``

**Implemented by issue #4621.**

Connect the frozen payload manifest and prepared-cache choices to the
session-scoped tasks in :ref:`l-next-steps-prepared-execution`. Prioritize
embeddings and early decoder blocks; prepare later blocks under bounded memory
and I/O admission. Execution waits only for the exact ``PreparedKey`` objects
it consumes and never observes a partially published object.

Acceptance:

* the trace proves real I/O/prepack/inference overlap;
* first-block priority improves ``T_first_token`` without exceeding the memory
  budget;
* hot fully prepared execution retains the direct ``ExecutionPlan`` cost
  envelope;
* synchronous mode remains the correctness reference;
* dedicated x86 and ARM results separate cold storage, warm page cache, warm
  prepared cache, first token, and fully prepared milestones.

``PreparedMaterialization`` now joins each resolver-selected packed or portable
recipe to its manifest-backed session task chain. Invocation descriptors name
only the ``PreparedKey`` objects they consume; plan construction adds those exact
publisher dependencies and rejects missing or ambiguous publishers. Critical
embedding and early-block chains outrank explicitly prefetched later blocks while
the existing per-resource and global admission budgets remain authoritative.

Every executed task also emits a timestamped ``PreparedTaskTrace`` interval with
its effective priority and resource class. This makes I/O, prepack, and inference
overlap directly testable instead of inferring it from aggregate load time.
``bench_prepared_cache_startup`` reports portable and prepared-cache first-token
times, while ``bench_prepared_hot_path`` reports the fully prepared cost against
the direct ``ExecutionPlan`` envelope. Run both binaries on each x86 and ARM
target, once after dropping the storage cache and once with a warm page cache;
retain their machine-readable output as the dedicated platform result.

Native PR05 -- add prepared-object eviction and reload
++++++++++++++++++++++++++++++++++++++++++++++++++++++

**Repository:** ``xadupre/onnx-light``

**Implemented by issue #4622.**

Add pinning, bounded prepared residency, eviction, and reload from the
companion prepared model. An active inference pins every object it consumes;
eviction may remove residency but never identity or an in-use allocation.

Acceptance:

* the configured residency budget is respected;
* no active consumer observes an evicted object;
* a later token reloads a compatible packed payload without reading or
  prepacking its portable source;
* critical reloads outrank speculative preparation and background cache writes.

The prepared scheduler's memory budget now also bounds completed object
residency. Least-recently-used inactive generations are evicted first, while
consumer pins delay eviction until the active task releases them. Eviction keeps
the object key and materialization recipe, so later invocations replay only the
selected companion packed-payload read and publication tasks. Existing
dependency priority propagation promotes demanded reloads above prefetch and
background persistence work.

Native PR06 -- add device preparation variants
+++++++++++++++++++++++++++++++++++++++++++++++

**Repository:** ``xadupre/onnx-light``

**Implemented by issue #4623.**

After fixed CPU placement is stable, add one CUDA ``Gemm`` whose initialization
chooses CPU-pack-plus-copy or device-side packing. Add alternative execution
device variants and residency policy only after that path is measurable.

Acceptance:

* device copies and preparation use explicit task resources and events;
* host and device ownership survive asynchronous submission;
* the selected variant records its device, layout, ABI, and source lineage;
* unsupported or failed variants fall back explicitly without publishing a
  partial prepared object.

The fixed-placement CUDA ``Gemm`` preparation contract now expands either
``I/O -> CPU pack -> device copy -> publish`` or
``I/O -> device copy -> device pack -> publish`` with explicit resource classes.
Backend submissions return an explicit completion event and retain their source
owners until that event completes. The published allocation retains a separate
device owner through residency, active-consumer pins, and eviction.

The prepared key length-prefixes the CUDA ordinal and architecture, packed
layout, kernel ABI, and ordered source lineage. Unsupported preferred paths
select the alternative directly. Failed submitted paths mark their generation
failed before the alternative produces a new generation, so only a complete
event-confirmed allocation can become resident. This PR deliberately adds no
dynamic placement or device-specific residency policy.

Validation on the PR build host used a kernel-enabled release build. The five
focused CUDA preparation tests and the fifteen existing prepared scheduler and
residency tests passed in approximately 52--54 ms. The 4 MiB prepared-cache benchmark measured
35.744 ms for portable preparation and 6.319 ms for the warm prepared entry
(82.3% improvement). The unchanged hot-path benchmark measured 39.407
microseconds of prepared dispatch overhead and did not meet its existing
1-microsecond envelope; the same benchmark and asynchronous dispatch path are
unchanged from the branch baseline, so that existing miss is not attributed to
device preparation. This host exposes no ``nvidia-smi``, so these results validate
the backend contract and fallback/ownership behavior but are not concrete CUDA
kernel timings; a CUDA backend must supply and measure the two submission
callbacks before adding placement or residency variants.

Completion criteria
+++++++++++++++++++

The native-completion step is complete when prepared execution is stable and
the native path:

* reads only the frozen live payload set;
* loads compatible prepared weights without their portable counterparts;
* overlaps later preparation with useful inference;
* reports truthful time, memory, I/O, page-fault, and copy counters;
* compares native ``.onnx`` plus prepared store against ORT protobuf ``.onnx``
  and ORT ``.ort``.

Only after all six native issues are closed does
:ref:`l-next-steps-model-loading` begin the final onnxruntime implementation
and add the ORT-plus-onnx-light integrated comparison.
