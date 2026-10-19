Standalone threading library
============================

``onnx_threads`` owns the repository's threading implementation. It depends
only on C++20 and platform threading facilities, not on Python, ONNX messages,
``onnx_proto``, or ``onnx_light``.

Build and consume
-----------------

.. code-block:: bash

    cmake -S onnx_threads -B build/threads \
        -DONNX_THREADS_BUILD_TESTS=ON -DCMAKE_INSTALL_PREFIX=/your/prefix
    cmake --build build/threads --parallel 6
    ctest --test-dir build/threads --output-on-failure
    cmake --install build/threads

Use ``ONNX_THREADS_BUILD_SHARED=ON`` for a shared library. Tests include both
an in-tree consumer and, for standalone installations, an installed-package
consumer. Only tests require GoogleTest.

.. code-block:: cmake

    find_package(onnx_threads CONFIG REQUIRED)
    target_link_libraries(my_program PRIVATE onnx_threads::lib_onnx_threads)

Include ``onnx_threads/thread_pool.h`` for the batch I/O adapter,
``onnx_threads/worker_pool.h`` for queued persistent jobs, and
``onnx_threads/cpu_executor.h`` / ``onnx_threads/parallel_for.h`` for
synchronous computation. Names remain in ``onnx_light::utils`` and
``onnx_light::core::runtime`` to preserve the existing types and signatures.
CPU topology and feature detection retain ``onnx_light::core::platform``.

Ownership and policies
----------------------

One implementation library does not imply one worker pool instance.

* The I/O ``ThreadPool`` creates workers lazily and drains and joins them on
  ``Wait()``. ``WorkerPool`` keeps its workers until shutdown. Queued callables
  are owned; their referenced data must remain alive until completion. Waiting
  drains work and propagates task failures; destruction drains without throwing.
* Computation uses persistent workers, spin/park and affinity policies, a bounded
  compatible-executor registry, and bounded or inline nesting. Sessions retain
  executor leases. Thread-local scopes borrow their executor and restore the
  previous binding, including explicit serial overrides and external dispatch.
* The bounded synchronous ``ForkJoin`` helper owns worker-launch and join logic
  used by BLAKE3. Its callbacks do not throw and their contexts remain borrowed
  until return. The hash adapter retains its existing size threshold.
* ``TaskCompletion`` owns shared completion, failure, and cancellation state.
  ONNX-specific persistent-value publication remains in the model runtime.

I/O pools and computation pools stay independent. Blocking an I/O worker does
not occupy a computation participant. There is no new global I/O/CPU admission
budget, mandatory pool sharing, or changed concurrency default.

Compatibility and deployment
-----------------------------

Existing proto/runtime headers are forwarding facades; their names, namespaces,
types, member layouts, signatures, and defaults remain unchanged.
``onnx_proto::lib_onnx_proto`` and the existing ``onnx_light`` CMake targets
transitively link the threading library. No source migration is required for
target-based consumers. The proto project can build its sibling dependency or
find an independently installed ``onnx_threads`` package.

Definitions previously exported by the proto/core binaries now reside in the
threading binary. Rebuild/relink consumers with the updated CMake packages and
deploy the new shared library alongside the other runtime libraries. Existing
Windows import libraries alone do not provide the relocated definitions.
Manually specified include and link lists must include the new dependency;
the old libraries are not self-contained binary replacements. C++ consumers
must continue to use a compatible compiler and standard-library ABI.

CPU topology, affinity, profiling, executor sharing, thread-local bindings,
and completion state are implemented only here. Model parsing, adaptive I/O
decisions, kernel dispatch, inference, and persistent-value transactions stay
outside the library.
