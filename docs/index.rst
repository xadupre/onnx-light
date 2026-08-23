onnx-light
==========

.. image:: https://github.com/xadupre/onnx-light/actions/workflows/ci_core.yml/badge.svg
    :target: https://github.com/xadupre/onnx-light/actions/workflows/ci_core.yml
    :alt: core

.. image:: https://github.com/xadupre/onnx-light/actions/workflows/build_reduced_wheel.yml/badge.svg
    :target: https://github.com/xadupre/onnx-light/actions/workflows/build_reduced_wheel.yml
    :alt: build-reduced

.. image:: https://github.com/xadupre/onnx-light/actions/workflows/build_release_cpp.yml/badge.svg
    :target: https://github.com/xadupre/onnx-light/actions/workflows/build_release_cpp.yml
    :alt: Build C++ Release Artifacts

.. image:: https://github.com/xadupre/onnx-light/actions/workflows/cq_asan_ubsan.yml/badge.svg
    :target: https://github.com/xadupre/onnx-light/actions/workflows/cq_asan_ubsan.yml
    :alt: asan-ubsan

.. image:: https://github.com/xadupre/onnx-light/actions/workflows/cq_fuzz.yml/badge.svg
    :target: https://github.com/xadupre/onnx-light/actions/workflows/cq_fuzz.yml
    :alt: fuzz

.. image:: https://github.com/xadupre/onnx-light/actions/workflows/docs.yml/badge.svg
    :target: https://github.com/xadupre/onnx-light/actions/workflows/docs.yml
    :alt: Documentation

.. image:: https://github.com/xadupre/onnx-light/actions/workflows/doc_cpp.yml/badge.svg
    :target: https://github.com/xadupre/onnx-light/actions/workflows/doc_cpp.yml
    :alt: Doxygen

.. image:: https://github.com/xadupre/onnx-light/actions/workflows/style.yml/badge.svg
    :target: https://github.com/xadupre/onnx-light/actions/workflows/style.yml
    :alt: Style

.. image:: https://github.com/xadupre/onnx-light/actions/workflows/typing.yml/badge.svg
    :target: https://github.com/xadupre/onnx-light/actions/workflows/typing.yml
    :alt: Typing

.. image:: https://github.com/xadupre/onnx-light/actions/workflows/cq_sbom.yml/badge.svg
    :target: https://github.com/xadupre/onnx-light/actions/workflows/cq_sbom.yml
    :alt: SBOM

.. image:: https://github.com/xadupre/onnx-light/actions/workflows/spelling.yml/badge.svg
    :target: https://github.com/xadupre/onnx-light/actions/workflows/spelling.yml
    :alt: Spelling

.. image:: https://github.com/xadupre/onnx-light/actions/workflows/int_ir_py.yml/badge.svg
    :target: https://github.com/xadupre/onnx-light/actions/workflows/int_ir_py.yml
    :alt: INT ir-py

.. image:: https://codecov.io/gh/xadupre/onnx-light/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/xadupre/onnx-light

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
  In practice loading or saving large models is significantly faster
  (see the :ref:`threads benchmark example <l-example-plot-threads-load-save>`).
- **Zero-copy parsing** – When parsing from an in-memory bytes buffer, the
  ``no_copy=True`` option makes each tensor's ``raw_data`` point directly into
  the source bytes without allocating an extra copy.  This eliminates one
  ``malloc + memcpy`` per tensor initializer.
- **Encrypted save / load** – Models can be encrypted with AES-256-CBC
  (ONNXCRY1) or ChaCha20-Poly1305 (ONNXCRY2), both using
  PBKDF2-HMAC-SHA256 key derivation, and saved to a single self-contained
  ``.onnxc`` file, or serialized to an in-memory ``bytes`` object.
- **No serialize/parse round-trip for C++ tools** – the Python :class:`~onnx_light.onnx_lib.ModelProto`
  exposed by ``onnx_light.onnx`` *is* the C++ :class:`~onnx_light.onnx_lib.ModelProto` (bound through
  nanobind). No serialization is need from Python to C++.
- Supports protobuf (onnx) and flatbuffers (onnxruntime) format.

Modular C++ libraries
+++++++++++++++++++++

The C++ code is shipped as several small libraries so that downstream
projects can link only what they need:

- ``onnx_light::lib_onnx_proto`` – protobuf-compatible message types,
  parser / serializer, external data, optional encrypted save / load
  (AES-256-CBC or ChaCha20-Poly1305).
- ``onnx_light::lib_onnx_core`` – implements *all* the generic
  functionalities (runtime value types and execution engine, the
  ``LightOpSchema`` data structures, the symbolic expression engine and
  the kernel / shape-inference dispatch tables) but ships **no** concrete
  operators. The dispatch tables start empty and are filled by the
  extension libraries below.
- ``onnx_light::lib_onnx_op`` – lightweight ``LightOpSchema``
  registrations for ONNX operator domains, with no shape inference.
- ``onnx_light::lib_onnx_manipulations`` – graph-manipulation helpers
  (text parser / printer, attribute and tensor proto helpers, data-type
  name utilities, graph-input collection); depends only on
  ``lib_onnx_proto``.
- ``onnx_light::lib_onnx_lib`` – full ONNX-compatible schemas (with
  history), checker, inliner, shape inference and version converter.
- ``onnx_light::lib_onnx_shape`` – shape-inference dispatch table,
  expression engine and graph optimization helpers.
- ``onnx_light::lib_onnx_patterns`` – concrete ONNX graph-rewriting
  patterns registered through ``onnx_patterns::RegisterPatterns()``; the
  generic pattern-optimization interface and registry remain in
  ``lib_onnx_core``.
- ``onnx_light::lib_onnx_kernels`` – C++ kernels, a C++ reference implementation,
  it is used to generate the expected outputs for the backend test.
- ``onnx_light::lib_onnx_backend_test`` – C++ backend test
  infrastructure and reference operator kernels.
- ``onnx_light::lib_onnx_gradient`` – reverse-mode automatic
  differentiation for ONNX graphs.

In addition, ``onnx_light::onnx_lib`` replicates the current C++ API
from :epkg:`onnx` package.

``onnx_core`` only implements the mechanisms: the actual operator
schemas, kernels, shape-inference and peak-memory functions are
**registered** into the shared dispatch tables owned by ``onnx_core`` by
the extension libraries (``onnx_op``, ``onnx_shapes``, ``onnx_kernels``,
...) through their ``Register*Functions()`` entry points. This keeps the
extensions independent from each other while sharing the same core engine.
See :ref:`l-design-library-split` for the detailed breakdown of each
assembly and :ref:`l-design-cpp-linking` for the matching CMake usage.

Kernels
+++++++

It is a C++ reference implementation and used to generate the expected
outputs for the backend tests. Parallelization is allowed except where it
would change the order of floating-point accumulation: operators that
accumulate internally (reductions, ``MatMul``, ``Gemm``, ``Attention``, ...)
stay sequential on the accumulated axis to enforce reproducibility.
See :ref:`l-design-library-split` for details.

Backend tests
+++++++++++++

They are fully written in C++. They can be called from any language.
Every output is generated with a C++ implementation of the operator.
Kernels can be used without the backend tests but the backend tests rely
on the kernels to produce the expected outputs.

Running models
++++++++++++++

The kernels double as a self-contained C++ **reference runtime** for the ONNX
operator set, so a model can be evaluated in C++ (or from Python) without a
third-party runtime. A ``RuntimeSession`` parses a model once, builds an
execution plan and can then be run repeatedly on runtime ``Tensor`` inputs.
See :ref:`l-design-runtime` for the design and
:ref:`l-example-plot-abs-benchmark` for a benchmark against :epkg:`onnxruntime`.

Graph optimization
++++++++++++++++++

A model is optimized by repeatedly matching small
:class:`~onnx_light.onnx_core.optimization.PatternOptimization` subgraphs and
replacing them with a simplified equivalent. Patterns are implemented in C++,
registered into a shared dispatch table (a downstream project can add its own)
and every applied rewrite is recorded and can be replayed. See
:ref:`l-example-plot-pattern-optimization` for the workflow.

Gradients
+++++++++

Gradients of an ONNX graph can be computed and used to train a model. See
:ref:`l-example-gradient-linear-regression` for an example.

.. toctree::
    :maxdepth: 1
    :caption: Contents

    quick_tour
    getting_started
    design/index
    api/index
    operators/index
    examples
    next_steps/index
    misc/index
