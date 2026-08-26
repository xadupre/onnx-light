cpu_executor.h
==============

.. doxygenfile:: onnx_core/runtime/tuning/cpu_executor.h
   :project: onnx-light

Cost-aware loops
----------------

Kernels may describe per-iteration reads, writes, and relative compute cycles
with ``CpuLoopCost`` instead of selecting a machine-specific byte threshold.
``CpuExecutor::PlanParallelFor`` combines that descriptor with the resolved
session participant limit and the optional kernel participant limit to choose a
task grain and participant count. The executor does not impose a
kernel-independent ceiling: kernels whose throughput saturates before the
session limit should pass their measured ceiling explicitly.
``CpuParallelConstraints`` also lets a calibrated kernel request an exact
participant target once parallel execution is worthwhile. The executor always
clamps that preference to the kernel ceiling and resolved session limit.
Explicit kernel tuning can continue to use the fixed-grain ``ParallelFor``
overload when a calibrated profile requires it.
