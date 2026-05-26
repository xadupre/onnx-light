.. _l-cpp-run-backend-test-ort-example:

Standalone C++ example: run every backend test through onnxruntime
==================================================================

This page documents ``examples/run_backend_test_ort``, a self-contained CMake
project that demonstrates how to run **every** C++-generated backend test
case (the same registry exposed by ``lib_onnx_backend_test`` and used by the
Python reference backend tests) through a downloaded `onnxruntime
<https://onnxruntime.ai/>`_ release, and verify that the runtime outputs
match the expected outputs declared by each test case.

It is the C++ counterpart of
``unittests/backend/test_backend_with_onnxruntime.py`` and reuses a similar
exclusion list for cases that ORT cannot run as-is
(``test_cc_roialign_max``, ``test_cc_flex_attention_*``, ``test_cc_adam_*``),
plus a few extras specific to this simplified harness:
``test_cc_acos`` / ``test_cc_acosh`` (stamped with opset 22, beyond ORT
1.19.x's supported max of opset 21) and ``test_cc_sequence_construct*``
(model output is an ``Ort::Value`` sequence rather than a tensor, which the
small comparator in ``main.cc`` does not handle).

Models stamped by ``lib_onnx_backend_test`` use the latest ONNX IR version
(13). The downloaded onnxruntime release pins a lower maximum IR version
(10 in 1.19.x), so the example downgrades ``model.ir_version`` to that
maximum before serializing -- the IR version is the serialization-format
identifier and does not change the semantics of the embedded opsets. Pass
``-DONNX_LIGHT_RUN_BACKEND_TEST_ORT_IR_VERSION=<n>`` at CMake configure time
to override the cap if you point the example at a newer onnxruntime release.

Why a pre-built onnxruntime release?
------------------------------------

Building onnxruntime from source is heavy (gigabytes of dependencies, long
compile times, Python tooling) and not the point of this example.  The
official pre-built CPU release archives published at
https://github.com/microsoft/onnxruntime/releases already ship the C++
headers and the dynamic library, so this example just consumes one of those
archives directly.

Step 1 -- Install the onnx_light C++ library
---------------------------------------------

From the *onnx-light* repository root, build and install the static library
and its public headers (the Python extension is not needed):

.. code-block:: bash

    cmake -S . -B build-install \
          -DCMAKE_BUILD_TYPE=Release \
          -DONNX_LIGHT_BUILD_PYTHON=OFF \
          -DCMAKE_INSTALL_PREFIX=/usr/local
    cmake --build  build-install
    cmake --install build-install

Step 2 -- Obtain a pre-built onnxruntime CPU release
----------------------------------------------------

You have two options:

* **Option A -- let CMake download it for you.** The example ships with
  ``cmake/FindOrt.cmake`` (a trimmed copy of
  `onnx-extended's FindOrt.cmake <https://github.com/sdpython/onnx-extended/blob/main/_cmake/externals/FindOrt.cmake>`_)
  that downloads the prebuilt release via ``FetchContent``. Pick the version
  via ``-DORT_VERSION=<x.y.z>`` at configure time (default: ``1.19.2``).
  Skip to Step 3 -- Option B.

* **Option B -- point at an existing extracted release.** Pick the archive
  that matches your platform on
  https://github.com/microsoft/onnxruntime/releases, extract it, and
  remember the path -- for example ``/opt/onnxruntime-linux-x64-1.19.2``.
  The directory must contain ``include/onnxruntime_cxx_api.h`` and
  ``lib/libonnxruntime.so.*`` (``onnxruntime.dll`` / ``onnxruntime.lib`` on
  Windows, ``libonnxruntime.dylib`` on macOS).

Step 3 -- Build the example
---------------------------

Option A -- point at an existing extracted release:

.. code-block:: bash

    cmake -S examples/run_backend_test_ort -B build-run-backend-test-ort \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_PREFIX_PATH=/usr/local \
          -DONNXRUNTIME_ROOT_DIR=/opt/onnxruntime-linux-x64-1.19.2
    cmake --build build-run-backend-test-ort

Option B -- let CMake download the release:

.. code-block:: bash

    cmake -S examples/run_backend_test_ort -B build-run-backend-test-ort \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_PREFIX_PATH=/usr/local \
          -DORT_VERSION=1.19.2
    cmake --build build-run-backend-test-ort

Step 4 -- Run the example
-------------------------

.. code-block:: bash

    LD_LIBRARY_PATH=/opt/onnxruntime-linux-x64-1.19.2/lib \
        ./build-run-backend-test-ort/run_backend_test_ort

The binary accepts an optional positional argument that is treated as a
regular expression filter on the test case name -- handy for narrowing down
a single case:

.. code-block:: bash

    LD_LIBRARY_PATH=/opt/onnxruntime-linux-x64-1.19.2/lib \
        ./build-run-backend-test-ort/run_backend_test_ort '^test_cc_add'

One-shot script
---------------

To download onnxruntime, install onnx_light, and build the example in one
go:

.. code-block:: bash

    bash examples/run_backend_test_ort/build.sh

On Windows:

.. code-block:: bat

    examples\run_backend_test_ort\build.bat

Set ``ONNXRUNTIME_VERSION=<x.y.z>`` to pick a release tag, or
``ONNXRUNTIME_ROOT_DIR=/path/to/extracted/release`` to reuse an already
downloaded archive.
