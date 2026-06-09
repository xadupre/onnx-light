.. _l-design-fuzz:

Fuzzing
=======

``onnx-light`` ships a set of :epkg:`libFuzzer`
instrumented C++ harnesses
that exercise the public API surface from random / malformed inputs.
They live in the top-level ``fuzz/`` directory and are intended to be
driven by :epkg:`OSS-Fuzz` for
long-running coverage-guided campaigns, while a short smoke campaign
is also run in CI to catch regressions in the harnesses themselves.

The harnesses are written directly against the onnx-light C++ API,
mirroring the upstream ONNX harnesses introduced in
`onnx/onnx#8052 <https://github.com/onnx/onnx/pull/8052>`_ but using
``LLVMFuzzerTestOneInput`` instead of a Python (atheris) wrapper, so
they can be built with the standard clang + ``-fsanitize=fuzzer,...``
toolchain that OSS-Fuzz expects for C++ projects.

Harnesses
+++++++++

.. list-table::
    :header-rows: 1
    :widths: 35 35 30

    * - File
      - Entry point fuzzed
      - Input path
    * - ``fuzz/fuzz_checker.cc``
      - ``ModelProto::ParseFromString`` + :cpp:func:`onnx_light::checker::check_model`
      - Raw bytes → protobuf parser
    * - ``fuzz/fuzz_model_loader.cc``
      - ``ModelProto::ParseFromString`` + top-level graph walk + :cpp:func:`onnx_light::checker::check_model`
      - Raw bytes → protobuf parser
    * - ``fuzz/fuzz_parser.cc``
      - :cpp:func:`onnx_light::OnnxParser::Parse` (``ModelProto`` overload)
      - UTF-8 text (ONNX text format)
    * - ``fuzz/fuzz_shape_inference.cc``
      - :cpp:func:`onnx_light::shape_inference::InferShapes`
      - Raw bytes → protobuf parser
    * - ``fuzz/fuzz_optim_shape_inference.cc``
      - :cpp:func:`onnx_light::onnx_optim::shapes::InferShapesModel`
      - Raw bytes → protobuf parser
    * - ``fuzz/fuzz_version_converter.cc``
      - :cpp:func:`onnx_light::version_conversion::ConvertVersion`
      - Raw bytes → protobuf parser
    * - ``fuzz/make_seed_corpus.cc``
      - *(seed generator, not a fuzzer)*
      - Writes seed files for OSS-Fuzz / libFuzzer

Building the harnesses
++++++++++++++++++++++

The harnesses are compiled as standalone libFuzzer executables when
``ONNX_LIGHT_BUILD_FUZZERS=ON`` is passed to CMake. Clang is required
because libFuzzer (``-fsanitize=fuzzer,...``) ships with Clang.

.. code-block:: bash

    CC=clang CXX=clang++ cmake -S . -B build-fuzz \
        -DONNX_LIGHT_BUILD_FUZZERS=ON \
        -DONNX_LIGHT_BUILD_PYTHON=OFF
    cmake --build build-fuzz -j

The build produces one executable per ``fuzz/fuzz_*.cc`` source file
(``fuzz_checker``, ``fuzz_model_loader``, ``fuzz_parser``,
``fuzz_shape_inference``, ``fuzz_optim_shape_inference``,
``fuzz_version_converter``) plus the ``make_seed_corpus`` helper.

The default sanitizer set is ``address`` (so the actual link line is
``-fsanitize=fuzzer,address``). Pass ``-DONNX_LIGHT_FUZZER_SANITIZERS=...``
to override it (for example ``undefined`` or ``memory``); the
``fuzzer`` sanitizer is always added automatically.

Running a harness locally
+++++++++++++++++++++++++

Each harness takes the standard :epkg:`libFuzzer command line`.
To run a short smoke campaign:

.. code-block:: bash

    ./build-fuzz/fuzz_checker -runs=1000
    ./build-fuzz/fuzz_model_loader -runs=1000
    ./build-fuzz/fuzz_parser -runs=1000
    ./build-fuzz/fuzz_shape_inference -runs=1000
    ./build-fuzz/fuzz_optim_shape_inference -runs=1000
    ./build-fuzz/fuzz_version_converter -runs=1000

To generate the seed corpora that OSS-Fuzz uses as starting inputs:

.. code-block:: bash

    ./build-fuzz/make_seed_corpus \
        /tmp/fuzz_seeds/version_converter \
        /tmp/fuzz_seeds/parser \
        /tmp/fuzz_seeds/shape_inference

Each directory is created if needed and filled with one seed file per
model. The harnesses accept a seed-corpus directory as a positional
argument:

.. code-block:: bash

    ./build-fuzz/fuzz_shape_inference /tmp/fuzz_seeds/shape_inference -runs=1000

How OSS-Fuzz uses these files
+++++++++++++++++++++++++++++

The companion OSS-Fuzz infrastructure clones this repository and
builds each ``fuzz/fuzz_*.cc`` target with its standard C++ pipeline.
The OSS-Fuzz ``build.sh`` should:

1. Configure the project with
   ``-DONNX_LIGHT_BUILD_FUZZERS=ON -DONNX_LIGHT_BUILD_PYTHON=OFF``.
2. Build each ``fuzz_*`` target.
3. Run ``make_seed_corpus`` and zip each directory into the matching
   ``$OUT/fuzz_<target>_seed_corpus.zip``.

Continuous fuzzing in CI
++++++++++++++++++++++++

The ``.github/workflows/fuzz.yml`` workflow builds the harnesses with
Clang + libFuzzer and runs a short smoke campaign (``-runs=2000``
per harness) on a weekly schedule (Mondays at 06:00 UTC), on manual
``workflow_dispatch``, and on pull requests that touch ``fuzz/**``,
``.github/workflows/fuzz.yml`` or ``CMakeLists.txt``. It is meant to
catch regressions in the harnesses themselves and obvious shallow
bugs; long-running coverage-guided campaigns are still expected to be
driven by OSS-Fuzz.

Design notes
++++++++++++

Why ``catch (...) { return 0; }``?
----------------------------------

Fuzz targets must never crash on expected errors — only on *unexpected*
ones (memory corruption, hangs, sanitizer reports). All protobuf
parse failures, ``ValidationError``, ``InferenceError``,
``ConvertError``, etc. are expected when the fuzzer feeds random
bytes. Catching them lets libFuzzer keep searching for inputs that
cause real bugs.

Why ``LLVMFuzzerTestOneInput``?
-------------------------------

``LLVMFuzzerTestOneInput`` is the standard :epkg:`libFuzzer entry point.
Each harness defines it with ``extern "C"`` so libFuzzer's runtime can call it
directly without name mangling.

Adding a new harness
++++++++++++++++++++

1. Create ``fuzz/fuzz_<name>.cc`` following the pattern of an existing
   harness (single ``extern "C" int LLVMFuzzerTestOneInput(const
   uint8_t *data, size_t size)`` entry point that catches every
   exception).
2. CMake picks the new file up automatically because the fuzzer target
   list is globbed from ``fuzz/fuzz_*.cc``.
3. If the fuzzer benefits from seed inputs, add them to
   ``fuzz/make_seed_corpus.cc`` and wire up the matching output
   directory in the OSS-Fuzz ``build.sh``.
4. Open a PR here; once merged, update the OSS-Fuzz ``build.sh`` if a
   new seed directory was added.
