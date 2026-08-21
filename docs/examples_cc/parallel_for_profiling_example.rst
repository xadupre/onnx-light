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

Portable timing is the default and performs no hardware-counter syscall. On
Linux, pass ``--hardware-counters`` to request one ``perf_event_open`` group
containing cycles, retired instructions, LLC references, and LLC misses:

.. code-block:: bash

    ./build-parallel-for-profiling/parallel_for_profiling --hardware-counters

Access may require ``CAP_PERFMON`` or a sufficiently permissive
``kernel.perf_event_paranoid`` value (commonly ``1`` or lower). The report
distinguishes ``unsupported``, ``permission_denied``, ``multiplexed``,
``overflowed``, and ``valid`` samples. This backend deliberately rejects
multiplexed samples instead of scaling them: IPC and LLC miss rate are present
only for isolated single-thread regions when ``time_running == time_enabled``.
Multi-thread regions are also marked ``multiplexed`` because one perf group
cannot represent their aggregate work. When counters are unavailable, the
example still reports portable wall and process CPU timing. Non-Linux systems
report an opted-in request as ``unsupported``.

For an isolated, single-threaded workload, compare a valid sample with
``perf stat -e cycles,instructions,cache-references,cache-misses``. Counts from
separate runs are expected to agree within 10%; pin the workload to one CPU and
repeat it if frequency scaling or scheduler activity causes more variance.

The collector capacity is one and the session runs twice, so the report contains
one event and reports one dropped event.

.. literalinclude:: ../../examples/parallel_for_profiling/main.cc
    :language: cpp
    :linenos:
