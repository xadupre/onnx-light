onnx_proto
==========

The C++ proto library lives in the top-level ``onnx_proto/`` directory and can
be built and installed without Python, operator schemas, or the runtime:

.. code-block:: bash

    cmake -S onnx_proto -B build-proto -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/path/to/proto-install
    cmake --build build-proto --parallel 6
    cmake --install build-proto

Consumers use ``find_package(onnx_proto CONFIG REQUIRED)`` and link
``onnx_proto::lib_onnx_proto``. The same target is available through the full
``onnx_light`` package. Both ``#include "onnx.h"`` and
``#include "onnx_proto/onnx.h"`` continue to work.
Installed flat headers forward to the canonical ``onnx_proto/`` tree, so both
spellings can be mixed safely in the same translation unit.

``ONNX_PROTO_BUILD_SHARED`` selects a shared library instead of the default
static library; ``ONNX_PROTO_INSTALL`` controls installation. OpenSSL is optional
and detected automatically; OpenMP is not required. The project includes serialization,
structural validation, DLPack support, BLAKE3, and the common C++ helpers.
It does not provide the ONNX operator schemas or schema-based checker.

``ONNX_PROTO_WERROR`` treats compiler warnings as errors when enabled (default:
``OFF``). The full repository build follows ``ONNX_LIGHT_WERROR`` instead.
Vendored BLAKE3 sources remain exempt from warnings-as-errors.

Configure with ``-DONNX_PROTO_BUILD_TESTS=ON`` and run
``ctest --test-dir build-proto --output-on-failure`` to run the native proto
suites, including serialization, structural validation, external data,
threading, cryptography, DLPack, and common helpers. Test sources, fixtures,
and test utilities live in ``onnx_proto/tests/``. Tests require GoogleTest,
which is found locally or fetched at the pinned version.

The same suites are registered by the full repository build when
``ONNX_LIGHT_BUILD_TESTS=ON``; they are not duplicated in its main test executable.
Schema, parser, and core graph-helper integration tests stay under ``unittests/cc``.
The small project under ``onnx_proto/tests/consumer/`` separately checks both
exported target names against the installed package without requiring GoogleTest.
On Windows, build-tree DLLs and test executables share the runtime output
directory; installed-package consumer checks copy the proto DLL beside each
executable before execution.

The repository's main build includes this project directly and uses a shared
proto library for its Python extensions. Python modules, imports, and bindings
remain in ``onnx_light`` and are unchanged.

.. toctree::
    :maxdepth: 1

    onnx
    onnx_helper
    onnx_alias
    onnx_crypt
    onnx_dlpack
    onnx_ort_flatbuffers
    onnx_mapped_payload
    onnx_io_policy
    stream_class
    stream
    simple_string
    simple_span
    thread_pool
    worker_pool
    fields
    onnx_verify
    onnx_tree_ensemble
    serialize_options
    tml
    type_helper
    visibility
    google_protobuf_compat
