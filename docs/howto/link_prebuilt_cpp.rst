.. _l-howto-link-prebuilt-cpp:

:html_theme.sidebar_secondary.remove:

How to link a C++ project against the pre-built onnx-light artifacts
====================================================================

This page explains how to download the pre-built *onnx-light* C++ static
libraries and headers published for each release and how to consume them
from a CMake project without building *onnx-light* from source.

.. note::

   If you prefer to build *onnx-light* from source and install it locally,
   see :ref:`l-howto-install-onnx-light` and :ref:`l-design-cpp-linking` instead.

Pre-built artifacts
-------------------

Every *onnx-light* release publishes a set of platform-specific archives to
the `GitHub Releases page
<https://github.com/xadupre/onnx-light/releases>`_.  The archives contain:

* all public C++ headers under ``include/onnx_light/``,
* static libraries under ``lib/`` (``lib_onnx_proto``, ``lib_onnx_core``,
  ``lib_onnx_lib``, ``lib_onnx_op``, ``lib_onnx_shape``, ``lib_onnx_kernels``
  and ``lib_onnx_backend_test``), and
* CMake package files under ``lib/cmake/onnx_light/`` so that
  ``find_package(onnx_light)`` works out of the box.

The naming convention for the archive files is::

    onnx-light-cpp-<platform>-<arch>.<ext>

Where ``<platform>-<arch>`` is one of:

.. list-table::
    :header-rows: 1
    :widths: 35 20 45

    * - File
      - OS
      - Notes
    * - ``linux-x86_64.tar.gz``
      - Linux x86-64
      - Built on Ubuntu 22.04 (GCC, static libs)
    * - ``linux-aarch64.tar.gz``
      - Linux AArch64
      - Built on Ubuntu 24.04 ARM (GCC, static libs)
    * - ``windows-AMD64.zip``
      - Windows x64
      - Built with MSVC, ``.lib`` static archives
    * - ``windows-ARM64.zip``
      - Windows ARM64
      - Built with MSVC, ``.lib`` static archives
    * - ``macos-universal2.tar.gz``
      - macOS Apple Silicon & Intel
      - Universal binary (arm64 + x86_64), built on macOS (Clang, static libs)

Step 1 – Download and extract
------------------------------

Linux / macOS
^^^^^^^^^^^^^

Replace ``<version>`` with the release tag (e.g. ``v0.1.0``) and
``<platform>`` with the appropriate suffix from the table above.

.. code-block:: bash

    VERSION=v0.1.0
    PLATFORM=linux-x86_64    # adjust to your platform
    ARCHIVE="onnx-light-cpp-${PLATFORM}.tar.gz"
    PREFIX="${HOME}/.local/onnx-light"   # installation directory

    curl -fL "https://github.com/xadupre/onnx-light/releases/download/${VERSION}/${ARCHIVE}" \
         -o "${ARCHIVE}"
    mkdir -p "${PREFIX}"
    tar -xzf "${ARCHIVE}" -C "${PREFIX}" --strip-components=1

After extraction the directory looks like this::

    ~/.local/onnx-light/
    ├── include/
    │   └── onnx_light/      ← CMake target include dir; #include "onnx.h" not "onnx_light/onnx.h"
    │       ├── onnx.h
    │       ├── stream.h
    │       └── ...
    └── lib/
        ├── liblib_onnx_proto.a
        ├── liblib_onnx_core.a
        ├── liblib_onnx_lib.a
        ├── liblib_onnx_op.a
        ├── liblib_onnx_shape.a
        ├── liblib_onnx_kernels.a
        ├── liblib_onnx_backend_test.a
        └── cmake/
            └── onnx_light/
                ├── onnx_lightConfig.cmake
                ├── onnx_lightConfigVersion.cmake
                └── onnx_lightTargets.cmake

Windows
^^^^^^^

.. code-block:: powershell

    $Version  = "v0.1.0"
    $Platform = "windows-AMD64"    # or windows-ARM64
    $Archive  = "onnx-light-cpp-${Platform}.zip"
    $Prefix   = "C:\opt\onnx-light"

    Invoke-WebRequest `
        -Uri "https://github.com/xadupre/onnx-light/releases/download/${Version}/${Archive}" `
        -OutFile $Archive
    Expand-Archive -Path $Archive -DestinationPath $Prefix -Force
    # Flatten the top-level "onnx-light-cpp" subdirectory if present:
    # Move-Item -Path "$Prefix\onnx-light-cpp\*" -Destination $Prefix

Step 2 – Use in a CMake project
---------------------------------

Once extracted, point ``CMAKE_PREFIX_PATH`` at the installation directory and
call ``find_package(onnx_light REQUIRED)`` in your project.

Minimal ``CMakeLists.txt``
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.15)
    project(my_project LANGUAGES CXX)

    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    find_package(onnx_light REQUIRED)

    add_executable(my_binary main.cc)

    # Choose the target that matches what you need (see "Available targets" below).
    target_link_libraries(my_binary PRIVATE onnx_light::lib_onnx_proto)

Configure with ``-DCMAKE_PREFIX_PATH`` pointing at the extracted prefix:

.. tab-set::

   .. tab-item:: Linux / macOS
      :sync: unix

      .. code-block:: bash

          cmake -S . -B build \
                -DCMAKE_BUILD_TYPE=Release \
                -DCMAKE_PREFIX_PATH="${HOME}/.local/onnx-light"
          cmake --build build

   .. tab-item:: Windows
      :sync: windows

      .. code-block:: bat

          cmake -S . -B build ^
                -DCMAKE_BUILD_TYPE=Release ^
                -DCMAKE_PREFIX_PATH="C:\opt\onnx-light"
          cmake --build build --config Release

Available targets
-----------------

After ``find_package(onnx_light REQUIRED)`` succeeds the following CMake
imported targets are available.  Link only what you need to keep build times
short.

.. list-table::
    :header-rows: 1
    :widths: 40 60

    * - CMake target
      - What it provides
    * - ``onnx_light::lib_onnx_proto``
      - Proto parsing / serialization only (``ModelProto``, ``GraphProto``,
        ``TensorProto``, …).  Lightest option.
    * - ``onnx_light::lib_onnx_core``
      - Intermediate layer between ``lib_onnx_proto`` and the higher-level
        libraries: owns the ``TensorType`` enumeration and ``ToTypeString``
        converter and provides the graph-node helpers (``CollectExternalInputs``,
        ``CollectNodeInputs``, ``CollectRemainingInputs``).  Transitively links
        ``lib_onnx_proto``.
    * - ``onnx_light::lib_onnx_manipulations``
      - ModelProto/GraphProto manipulation helpers independent from operator
        schemas: textual proto parser/printer, attribute and tensor-proto
        utilities, data-type name helpers, and graph-analysis utilities
        (``CollectExternalInputs``, ``CollectRemainingInputs``,
        ``CollectNodeInputs``).  Transitively links ``lib_onnx_proto``.
    * - ``onnx_light::lib_onnx_op``
      - Lightweight ``LightOpSchema`` registrations, no shape inference.
        Transitively links ``lib_onnx_proto``.
    * - ``onnx_light::lib_onnx_lib``
      - Full ONNX-compatible schemas with history, checker, inliner, shape
        inference and version converter.  Transitively links ``lib_onnx_proto``.
    * - ``onnx_light::lib_onnx_shape``
      - Shape-inference dispatch table, expression engine and graph
        optimization helpers.  Transitively links ``lib_onnx_op``.
    * - ``onnx_light::lib_onnx_kernels``
      - C++ reference implementation of ONNX operators (``RunGraph``,
        ``RunModel``, ``struct Tensor``, …).  Depends on ``lib_onnx_proto``.
    * - ``onnx_light::lib_onnx_backend_test``
      - Backend-test infrastructure and per-operator case registries.
        Transitively links ``onnx_kernels``.

Minimal example (proto parsing only)
-------------------------------------

The snippet below reads an ``.onnx`` file and prints the number of nodes
using only ``onnx_light::lib_onnx_proto``.

.. code-block:: cpp

    #include "onnx.h"
    #include "stream.h"

    #include <iostream>

    int main(int argc, char **argv) {
      if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " model.onnx\n";
        return 1;
      }
      onnx::ModelProto model;
      onnx::utils::FileStream stream(argv[1]);
      onnx::ParseModelProtoFromStream(model, stream, {});
      if (model.has_graph()) {
        std::cout << "nodes: " << model.ref_graph().node_size() << "\n";
      }
      return 0;
    }

The matching ``CMakeLists.txt``:

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.15)
    project(load_model LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    find_package(onnx_light REQUIRED)
    add_executable(load_model main.cc)
    target_link_libraries(load_model PRIVATE onnx_light::lib_onnx_proto)

Notes
-----

* All targets require C++20 or later (``set(CMAKE_CXX_STANDARD 20)``).
* The pre-built archives are compiled *without* the Python extension modules
  (``-DONNX_LIGHT_BUILD_PYTHON=OFF``) and *with* OpenSSF compiler hardening
  flags (``-DONNX_HARDENING=ON``).
* OpenSSL is an **optional** dependency of ``onnx_light::lib_onnx_proto`` (it
  enables encrypted ``.onnxc`` model save/load).  If your project does not
  need encryption you can ignore it.  If you do, install OpenSSL on the
  target machine and the exported CMake targets will pick it up automatically
  through the bundled ``find_dependency(OpenSSL QUIET)`` call.
* On Windows the static libraries use the ``/MT`` (MSVC static runtime) or
  ``/MD`` flag matching the runner's default toolchain.  Check the GitHub
  Actions build log for the exact flag if you need to match the runtime
  library in your own project.

See also
--------

* :ref:`l-howto-install-onnx-light` – build and install *onnx-light* from
  source.
* :ref:`l-design-cpp-linking` – full description of every exported CMake
  target and the ``add_subdirectory`` alternative.
* :ref:`l-cpp-load-onnx-light-time-example` – a complete runnable example.
* :ref:`l-howto-onnx-graph-manipulations` – how to use the graph-analysis
  helpers from ``onnx_light::lib_onnx_manipulations``.
