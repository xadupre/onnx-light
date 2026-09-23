runtime_context.h
==================

Selected result ownership
-------------------------

``set_retained_outputs`` lists exact whole result names whose storage will be
transferred to an external owner. ``PersistentValueState`` supplies this list from
the graph declarations. Only those names bypass allocator migration and final
output materialization; other values use the ordinary runtime policies.
New function/subgraph contexts do not inherit the list. ``If`` and function
transport explicitly translate selected caller output names into child names.
Kernel resolution is unchanged.

Ordinary tensors remain in ``tensors()``. ``values()`` carries the structured
and encoded representations documented in :doc:`runtime_value`.

Persistent-storage events
--------------------------

``RuntimeContextOptions.events_enabled`` controls all runtime event recording,
including persistent-storage auditing, and defaults to ``false``. In Python,
pass ``events_enabled=True`` when constructing ``RuntimeContext``. There is no
separate persistent-storage statistics getter or always-on counter collection.

All producers use ``RecordEvent(RuntimeEvent)``. The caller supplies the action
and payload; the context supplies node/subgraph metadata and allocator memory.
A nonzero timestamp is preserved (for example, the start of a kernel dispatch);
otherwise the recording time is used. Disabled recording leaves the log unchanged.

.. code-block:: cpp

   if (rt.events_enabled())
     rt.RecordEvent({.action = RuntimeEventAction::kPersistentStorage,
                     .storage_append_copied_bytes = copied_bytes});

``RuntimeEventAction::kPersistentStorage`` (integer value ``4``) is rendered as
``"persistent_storage"`` by ``RuntimeEventActionName`` and Python ``as_dict()``.
The work is stored directly in five unsigned integer fields of ``RuntimeEvent``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Meaning
   * - ``storage_allocations``
     - Number of persistent-storage allocations.
   * - ``storage_allocated_bytes``
     - Bytes allocated for persistent storage.
   * - ``storage_prefix_copied_bytes``
     - Bytes copied from an existing persistent prefix.
   * - ``storage_append_copied_bytes``
     - Bytes copied from newly appended values.
   * - ``storage_reuse_count``
     - Number of persistent-storage reservations reused without allocation.

These fields describe work for that event, not cumulative totals. Sum the
event fields over the desired interval to obtain totals. All storage fields
default to zero, and are read-only in Python. ``RuntimeEvent.allocated_bytes``
and ``RuntimeEvent.peak_bytes`` continue to describe allocator live/peak memory;
they are not persistent-storage allocation traffic.

The existing ``events()`` API returns these records alongside tensor mutations
and node dispatches. Python ``RuntimeEvent.as_dict()`` includes the five
``storage_*`` fields as top-level integers only for ``kPersistentStorage`` events,
preserving the dictionary schema of other event actions:

.. code-block:: python

   from onnx_light.onnx_py._onnxpykernels import runtime

   context = runtime.RuntimeContext(
       runtime.KernelContext(runtime.default_opset(23)), events_enabled=True
   )
   # Runs an existing PersistentValueState with its ordinary non-retained feeds.
   outputs = state.run(context, feeds)
   copied = sum(
       event.storage_prefix_copied_bytes
       for event in context.events()
       if event.action == runtime.RuntimeEventAction.kPersistentStorage
   )
   context.clear_events()

Context copies, subgraphs, functions, feedback invocations and half-precision
scratch contexts share the same event log, even when recording is disabled.
The ``events_enabled`` flag controls recording, not the existence of the log.
Events are recorded directly in this log, preserving existing entries even
when execution fails, without any forwarding or scope-exit merge. Independent
root contexts keep independent logs; a child keeps its log alive even after
its parent is destroyed.

Runtime recording serializes appends from concurrent children. Direct access
through ``events()`` requires no concurrent recording, clearing or modification.
The existing ``clear_events()`` / ``ClearEvents()`` clears the shared log for all
related contexts without changing feedback values or disabling recording.
``Clear()`` also clears the shared log but resets only the receiving context's
value maps. Allocation failures while recording propagate normally.

Kernel usage recording
----------------------

Backends can record their selected implementation names from
``KernelBase::Run(RuntimeContext &rt)`` with ``rt.RecordKernelUsage(name)``.
Recording is disabled by default and independent of tensor event logging.

.. code-block:: cpp

   rt.set_kernel_usage_enabled(true);
   session.Run(rt);
   const auto names = rt.GetKernelUsage();
   rt.set_kernel_usage_enabled(false);  // Keeps the recorded names.
   rt.ClearKernelUsage();

Each independently constructed context owns its recorder. Subgraph and
model-local function contexts, as well as context copies, share the owning
context's recorder, including changes made after children are created.
The recorder does not use the custom-kernel registry.

Recording, enable/disable, clear, and snapshot operations are thread-safe.
Other context operations are not made thread-safe by enabling recording.
The log retains the first ``RuntimeContext::kKernelUsageLimit`` names
(including duplicates) and drops further entries until explicitly cleared.
Snapshots own their strings. ``RuntimeContext::Clear()`` preserves recording
state and names across runs; ``ClearKernelUsage()`` clears only the shared log.

.. doxygenfile:: onnx_core/runtime/runtime_context.h
   :project: onnx-light
