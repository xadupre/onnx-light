runtime_context.h
==================

Selected result ownership
-------------------------

``set_retained_outputs`` lists exact whole result names whose storage will be
transferred to an external owner. ``FeedbackState`` supplies this list from
the graph declarations. Only those names bypass allocator migration and final
output materialization; other values use the ordinary runtime policies.
New function/subgraph contexts do not inherit the list. ``If`` and function
transport explicitly translate selected caller output names into child names.
Kernel resolution is unchanged.

Ordinary tensors remain in ``tensors()``. ``values()`` carries the structured
and encoded representations documented in :doc:`runtime_value`.

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
