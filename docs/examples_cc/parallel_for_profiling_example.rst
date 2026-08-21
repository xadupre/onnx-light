.. _l-cpp-parallel-for-profiling-example:

Inspect ParallelFor profiling from C++
========================================

This runnable example creates a fixed-capacity collector, passes it through
:cpp:class:`onnx_light::core::runtime::RuntimeSessionOptions`, runs inference,
and requests an owning :cpp:class:`onnx_light::core::runtime::ParallelRegionReport`.
The report copies its events and dropped count, so it can be retained without
exposing or depending on the collector's live storage.

Build an installed onnx-light tree and run the example with:

.. code-block:: bash

    cmake -S examples/parallel_for_profiling -B build-parallel-for-profiling \
          -DCMAKE_PREFIX_PATH=/usr/local
    cmake --build build-parallel-for-profiling
    ./build-parallel-for-profiling/parallel_for_profiling

The collector capacity is one and the session runs twice, so the report contains
one event and reports one dropped event.

.. literalinclude:: ../../examples/parallel_for_profiling/main.cc
    :language: cpp
    :linenos:
