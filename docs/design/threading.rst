.. _l-design-threading:

Threading implementation and ownership
=====================================

The root-level ``onnx_threads/`` project is an independent C++20 library.
The dependency direction is:

.. code-block:: text

    platform threading facilities
                 ^
            onnx_threads
                 ^
             onnx_proto
                 ^
             onnx_light

It contains queued I/O workers and their lazy batch adapter, persistent
computation workers, CPU topology and affinity policies, compatible-executor
sharing, nested dispatch, thread-local executor/profiling scopes, hardware
counter collection, completion state, and the bounded fork/join implementation
used by the BLAKE3 adapter.

Existing proto and runtime headers forward to the same definitions. There is
no dependency from the threading library to serialization, ONNX model types,
Python, or the model runtime. The generic CPU descriptor lives with the
topology-dependent execution policies; runtime tuning still consumes that
descriptor through its existing include path.

Sharing code is not sharing workers
-----------------------------------

I/O and computation have different completion and scheduling contracts.
``utils::ThreadPool`` lazily creates workers and joins them at the end of a
batch, whereas ``utils::WorkerPool`` supports persistent queued work.
``core::runtime::ThreadPool`` executes synchronous bounded parallel regions
using persistent workers and caller participation. These policies remain
distinct inside one library rather than being forced onto one queue.

The session holds its ``CpuExecutor`` lease. Compatible policies may share
workers through the existing bounded registry; an independent registry or
explicit pool still creates independent instances. No new mandatory process-wide
pool or shared I/O/computation budget is introduced. In particular, blocking
I/O does not consume a computation executor's participants.

Thread-local executor, external-dispatch and profiling scopes retain their
existing restoration and borrowing rules. Explicit serial executors override
parallel dispatch within their scope. Nested work stays inline by default or
uses only idle participants when bounded nesting is enabled.

ONNX-specific integration
-------------------------

Streams retain ownership of read/write lifetimes, adaptive I/O policy and
tensor materialization. Sessions retain graph execution and kernel integration.
Persistent-value transactions retain their publication logic while using the
generic completion primitive. BLAKE3 retains its hash-specific callback and
parallelization threshold, delegating worker admission and joining.

See :doc:`../api/cpp/onnx_threads/index` for the implementation API and
``onnx_threads/README.rst`` for standalone build, install, consumption, ownership,
and binary deployment requirements. Existing CMake package targets propagate
the dependency; shared deployments must also ship the new threading library.
