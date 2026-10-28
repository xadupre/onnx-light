compute_context.h
=================

Serialized rewrites preserve existing tags for retained values throughout tag
seeding and inference, including results replaced by initializers. New value
names still participate in inference; removed names lose their tags. This
protection is scoped to rewriting and does not change ordinary tag propagation.

Memory timeline boundaries
--------------------------

``GraphBuilder`` maintains shapes, tags, lifetimes and reuse decisions during
construction, but does not compute memory or scratch profiles. Its
``Compute().Memory()`` and ``Compute().PeakMemory()`` vectors remain empty.
``ToGraph``, ``ToModel`` and ``ToFunction`` compute profiles at export, including
nested graphs and local functions. A subsequent mutation discards those optional
profiles immediately, so callers never observe an outdated profile.

Standalone ``ComputeContext`` keeps its incremental behavior by default.
The explicit C++ ``MemoryProfiling::kExplicit`` constructor policy disables
automatic profiling during mutations without changing eager annotations;
``RefreshIncrementalMemory`` and ``ComputePeakMemory`` explicitly compute the
requested profiles. No query triggers analysis.

``ComputeContext::NodeMemory(i)`` (also ``Memory()[i]``) exposes symbolic or
concrete ``before_bytes``, ``during_bytes``, ``after_bytes`` and
``released_bytes`` scalars for every node, including the final node.
``after_bytes`` is computed from the profiler's retained allocations after
release decisions: transferring an in-place buffer to an output does not free
it, and a smaller output still owns the full reused allocation.

Call ``ComputeInPlaceReuseGraph`` after shape inference, then
``ComputePeakMemory`` with the desired device (or use ``Compute``).
``during_bytes`` includes the peak scratch estimate; ``released_bytes`` is
``during_bytes - after_bytes`` and includes scratch deallocation, temporary
function-body allocations and unused outputs. Before scratch estimation the
scratch contribution is zero. Graph inputs protected from overwriting remain
in the retained count, as do retained initializers and graph outputs.

The existing ``already_allocated_bytes``, ``output_allocation_bytes``,
``total_bytes`` and tagged buckets keep their original meaning.
Python exposes the same entries through ``node_memory`` / ``memory`` and the
``NODE_MEMORY_BEFORE_BYTES_KEY``, ``NODE_MEMORY_DURING_BYTES_KEY``,
``NODE_MEMORY_AFTER_BYTES_KEY`` and ``NODE_MEMORY_RELEASED_BYTES_KEY`` constants
in ``onnx_light.onnx_core.shape_inference``.

.. doxygenfile:: onnx_core/compute/compute_context.h
   :project: onnx-light
