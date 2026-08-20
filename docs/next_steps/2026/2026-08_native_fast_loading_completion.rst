.. _l-next-steps-native-fast-loading-completion:

Completing native fast model loading
====================================

:Date: 2026-08

**blocked by prepared execution**

Objective
+++++++++

This is step 4 of the fast-loading sequence. It starts only after
:ref:`l-next-steps-prepared-execution` provides session-scoped preparation
tasks, dependency events, bounded I/O and CPU resources, prepared-object
residency, and synchronous/asynchronous execution through one plan.

The objective is to connect the native loader, graph resolver, prepared tensor
cache, and prepared executor end to end. The target is time to first token and
fully prepared state, not a parser-only ``load()`` number.

Final PR01 -- adaptive external-data I/O
++++++++++++++++++++++++++++++++++++++++

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

Final PR02 -- resolve before materializing payloads
+++++++++++++++++++++++++++++++++++++++++++++++++++

**Repository:** ``xadupre/onnx-light``

Implement the executable slice of :ref:`l-next-steps-model-resolution` that
freezes ``RequiredPayloadManifest`` before large reads. Concrete useful cases
include a constant ``If`` branch, unused training/debug heads, initializers
removed by graph rewrites, and portable weights replaced by a compatible
prepared object.

This PR does not claim a gain when every serialized initializer remains live.
If representative fixtures avoid no material payload bytes, fold the manifest
plumbing into Final PR03 instead of shipping a standalone performance PR.

Acceptance:

* no task reads a range absent from the frozen manifest;
* dead, superseded, and prepared-cache-replaced payload bytes are counted;
* graph transformations cannot mutate requirements after reads begin;
* eager loading remains a diagnosed fallback for unsupported transformations.

Final PR03 -- consume prepared tensors before portable weights
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

**Repository:** ``xadupre/onnx-light``

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

Final PR04 -- connect and tune first-token overlap
++++++++++++++++++++++++++++++++++++++++++++++++++

**Repository:** ``xadupre/onnx-light``

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

Final PR05 -- add prepared-object eviction and reload
+++++++++++++++++++++++++++++++++++++++++++++++++++++

**Repository:** ``xadupre/onnx-light``

Add pinning, bounded prepared residency, eviction, and reload from the
companion prepared model. An active inference pins every object it consumes;
eviction may remove residency but never identity or an in-use allocation.

Acceptance:

* the configured residency budget is respected;
* no active consumer observes an evicted object;
* a later token reloads a compatible packed payload without reading or
  prepacking its portable source;
* critical reloads outrank speculative preparation and background cache writes.

Final PR06 -- add device preparation variants
++++++++++++++++++++++++++++++++++++++++++++++

**Repository:** ``xadupre/onnx-light``

After fixed CPU placement is stable, add one CUDA ``Gemm`` whose initialization
chooses CPU-pack-plus-copy or device-side packing. Add alternative execution
device variants and residency policy only after that path is measurable.

Acceptance:

* device copies and preparation use explicit task resources and events;
* host and device ownership survive asynchronous submission;
* the selected variant records its device, layout, ABI, and source lineage;
* unsupported or failed variants fall back explicitly without publishing a
  partial prepared object.

Completion criteria
+++++++++++++++++++

The four-document roadmap is complete when bug fixes are merged, ORT consumes
the explicit payload contract safely, prepared execution is stable, and the
native path:

* reads only the frozen live payload set;
* loads compatible prepared weights without their portable counterparts;
* overlaps later preparation with useful inference;
* reports truthful time, memory, I/O, page-fault, and copy counters;
* compares native ``.onnx`` plus prepared store against ORT protobuf ``.onnx``,
  ORT plus ``onnx-light``, and ORT ``.ort``.
