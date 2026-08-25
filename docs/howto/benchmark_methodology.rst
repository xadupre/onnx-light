.. _l-benchmark-methodology:

Benchmark methodology
=====================

CPU benchmarks are sensitive to persistent thread pools, lazy preparation,
allocator reuse, and runtime-specific caches. Comparative gallery examples
follow the rules below so a backend does not perturb the backend measured
after it.

Separate runtime phases
-----------------------

Prepare identical models and inputs first. Then measure every case for
``onnx-light`` before constructing any ONNX Runtime session. The ONNX Runtime
phase starts only after all ``onnx-light`` sessions from the measured phase
have been released. Other multithreaded libraries, such as NumPy linked to a
threaded BLAS, run in their own phase as well.

Do not alternate runtimes in a timing loop. Both runtimes keep their documented
default spin policy: changing spin behavior changes the runtime being measured.
Phase separation, rather than disabling spin, prevents one runtime's active
workers from consuming CPU while another runtime is timed.

Bound and report measurements
-----------------------------

Warm-up calls are untimed. A timed loop stops when its requested repetition
count is reached or when measured execution reaches two seconds cumulatively,
whichever comes first. A call already in progress is allowed to finish. Report
the median of the collected samples; retain raw samples when the benchmark
produces an artifact.

Keep model parsing, session construction, first-run preparation, and steady
state separate unless startup is the explicit subject of the benchmark.

Validate execution
------------------

Use the same logical inputs for every backend and validate complete outputs
with an explicit numerical tolerance. Kernel libraries must also verify the
library-qualified kernel name outside the timed region. Correct output alone
does not prove that the intended accelerated kernel ran.

Shared CI machines provide diagnostic measurements, not stable performance
gates. Publish the runtime versions, thread counts, CPU topology, and build
configuration when results are intended for comparison across machines.
