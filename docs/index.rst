onnx-light
==========

.. image:: https://github.com/xadupre/onnx-light/actions/workflows/ci_core.yml/badge.svg
    :target: https://github.com/xadupre/onnx-light/actions/workflows/ci_core.yml
    :alt: core

.. image:: https://github.com/xadupre/onnx-light/actions/workflows/build.yml/badge.svg
    :target: https://github.com/xadupre/onnx-light/actions/workflows/build.yml
    :alt: build

.. image:: https://github.com/xadupre/onnx-light/actions/workflows/mypy.yml/badge.svg
    :target: https://github.com/xadupre/onnx-light/actions/workflows/mypy.yml
    :alt: mypy

.. image:: https://github.com/xadupre/onnx-light/actions/workflows/docs.yml/badge.svg
    :target: https://github.com/xadupre/onnx-light/actions/workflows/docs.yml
    :alt: Documentation

.. image:: https://github.com/xadupre/onnx-light/actions/workflows/style.yml/badge.svg
    :target: https://github.com/xadupre/onnx-light/actions/workflows/style.yml
    :alt: Style

.. image:: https://github.com/xadupre/onnx-light/actions/workflows/pyrefly.yml/badge.svg
    :target: https://github.com/xadupre/onnx-light/actions/workflows/pyrefly.yml
    :alt: pyrefly

.. image:: https://github.com/xadupre/onnx-light/actions/workflows/spelling.yml/badge.svg
    :target: https://github.com/xadupre/onnx-light/actions/workflows/spelling.yml
    :alt: Spelling

.. image:: https://codecov.io/gh/xadupre/onnx-light/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/xadupre/onnx-light

.. image:: https://img.shields.io/github/repo-size/xadupre/onnx-light
    :target: https://github.com/xadupre/onnx-light

``onnx-light`` started from the upstream ONNX pull request
`onnx/onnx#7208 <https://github.com/onnx/onnx/pull/7208>`_, which is the
initial code base from which this project diverged.

onnx without protobuf and more freedom
++++++++++++++++++++++++++++++++++++++

- **Files larger than 2 GB** – :epkg:`protobuf` enforces a 2 GB message-size
  limit. ``onnx-light`` does not have this constraint.
- **External-data / multi-file models** – external files are supported
  natively in C++.
- **Parallel loading and saving** – 
  :func:`onnx_light.onnx.load` and :func:`onnx_light.onnx.save` are parallelized.
  In practice loading or saving large models is roughly **3 times faster with 4 threads**
  (see the :ref:`threads benchmark example <l-example-plot-threads-load-save>`).
- **Zero-copy parsing** – When parsing from an in-memory bytes buffer, the
  ``no_copy=True`` option makes each tensor's ``raw_data`` point directly into
  the source bytes without allocating an extra copy.  This eliminates one
  ``malloc + memcpy`` per tensor initializer.
- **Encrypted save / load** – Models can be encrypted with AES-256-CBC
  (PBKDF2-HMAC-SHA256 key derivation) and saved to a single self-contained
  ``.onnxc`` file, or serialized to an in-memory ``bytes`` object.
- **No serialize/parse round-trip for C++ tools** – the Python ``ModelProto``
  exposed by ``onnx_light.onnx`` *is* the C++ ``ModelProto`` (bound through
  nanobind). No serialization is need from Python to C++.

See :ref:`l-design-differences` for more details.

Modular C++ libraries
+++++++++++++++++++++

The C++ code is shipped as several small libraries so that downstream
projects can link only what they need:

- ``onnx_light::lib_onnx_proto`` – protobuf-compatible message types,
  parser / serializer, external data, optional AES-256 encrypted save / load.
- ``onnx_light::lib_onnx_op`` – lightweight ``LightOpSchema``
  registrations for ONNX operator domains, with no shape inference.
- ``onnx_light::onnx_light`` – full ONNX-compatible schemas (with
  history), checker, inliner, shape inference and version converter.
- ``onnx_light::lib_onnx_optim`` – shape-inference dispatch table,
  expression engine and graph optimization helpers.
- ``onnx_light::onnx_kernels`` – C++ kernels, a C++ reference implementation,
  it is used to generate the expected outputs for the backend test.
- ``onnx_light::onnx_backend_test`` – C++ backend test
  infrastructure and reference operator kernels.

In addition, ``onnx_light::onnx_lib`` replicates the current C++ API
from :epkg:`onnx` package.
See :ref:`l-design-library-split` for the detailed breakdown of each
assembly and :ref:`l-design-cpp-linking` for the matching CMake usage.

Kernels
+++++++

It is a C++ reference implementation and used to generate the expected
outputs for the backend tests. It is not parallelized on purpose to enforce
reproducibility.

Backend tests
+++++++++++++

They are fully written in C++. They can be called from any language.
Every output is generated with a C++ implementation of the operator.

.. toctree::
    :maxdepth: 1
    :caption: Contents

    getting_started
    intro/index
    design/index
    howto/index
    api/index
    operators/index
    examples
    misc/index
