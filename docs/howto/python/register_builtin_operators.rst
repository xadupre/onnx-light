.. _l-howto-register-builtins:
.. _l-howto-register-builtins-python:

:html_theme.sidebar_secondary.remove:

How to register a built-in kernel, test case, shape inference, light op or peak memory function
===============================================================================================

This guide covers the Python interface. For the other language, see
:doc:`../cpp/register_builtin_operators`.

:ref:`l-howto-use-custom-kernel-python` and :ref:`l-howto-use-custom-shape-inference-python`
show how to plug behaviour into the runtime *from outside* the library. This
page is the contributor-facing counterpart: it explains how to add a new
**built-in** operator to *onnx-light* itself. Every operator is made of a few
independent pieces, each registered in its own static table:

* a **kernel** – the C++ code that computes the outputs, in ``lib_onnx_kernels``;
* a **backend test case** – reference inputs/outputs used to validate the kernel;
* a **shape inference** function – propagates dtypes and symbolic shapes, in
  ``lib_onnx_shape``;
* a **light op schema** – the operator definition (inputs, outputs, type
  constraints, documentation), in ``lib_onnx_op``;
* an optional **peak memory** function – estimates the scratch memory the
  kernel needs, also in ``lib_onnx_shape``.

Each piece uses ``Abs`` (an element-wise unary math op) as the running example.
All the source directories are compiled with CMake ``GLOB_RECURSE`` patterns, so
adding a new ``.cc`` file is picked up automatically — no ``CMakeLists.txt`` edit
is required. Only the registration tables need a new entry.

Registration always happens in **C++** (each table is a C++ static). There is
**no separate Python registration step**: the extension exposes generic entry
points that read the same C++ dispatch tables, so once the libraries are rebuilt
(``pip install -e .`` or the CMake build) every piece is reachable from Python
automatically. This page covers the Python usage of those registrations.

.. contents::
    :local:

Register a kernel
-----------------

Kernels live under
``onnx_light/onnx_extensions/kernels/kernels/<category>/`` (``math``, ``nn``,
``tensor``, ...). A kernel is a class deriving from
:cpp:class:`onnx_light::core::runtime::KernelBase` and overriding ``Run``.


Nothing to register on the Python side. Once the C++ libraries are
rebuilt, run a model containing the operator with
:class:`onnx_light.onnx.reference.ReferenceEvaluator`, which drives the C++
``RuntimeSession`` and resolves kernels through ``BuiltinKernelFunctions``:

.. code-block:: python

    import numpy as np
    from onnx_light.onnx_lib import parser
    from onnx_light.onnx.reference import ReferenceEvaluator

    model = parser.parse_model(
        '<ir_version: 10, opset_import: ["" : 18]>'
        'agraph (float[3] x) => (float[3] y) { y = Abs(x) }'
    )
    sess = ReferenceEvaluator(model)
    (y,) = sess.run(None, {"x": np.array([-1.0, 2.0, -3.5], dtype=np.float32)})

Register a test case
--------------------

Backend test cases live under
``onnx_light/onnx_extensions/backend_test/cases/<category>/``. A case pairs a
:cpp:class:`NodeProto` with a lambda that produces the expected inputs and
outputs (typically by calling the kernel itself), through the
:cpp:func:`Expect` helper.


Enumerate the C++ cases with
:func:`onnx_light.onnx.backend.collect_test_cases`; passing the op type
returns just its cases. See :ref:`l-howto-collect-backend-test-cases-python` for
how to enumerate and run cases.

.. code-block:: python

    import onnx_light.onnx.backend as bt

    abs_cases = bt.collect_test_cases("Abs")

Register a shape inference
--------------------------

Shape functions live under
``onnx_light/onnx_extensions/shapes/shapes/<category>/``. A shape function has
signature ``void(ShapesContext &, const NodeProto &)`` and sets the symbolic
:cpp:class:`SymTensor` for every output.


Run the pipeline with
:func:`onnx_light.onnx_core.shape_inference.infer_shapes_model`, which
reads the shared ``core::shapes`` dispatch table populated by
``RegisterShapeFunctions``:

.. code-block:: python

    from onnx_light.onnx_core.shape_inference import infer_shapes_model

    infer_shapes_model(model)  # fills model.graph.value_info / output types

Register a light op
-------------------

Operator schemas (the "light" equivalent of ONNX ``OpSchema``) live under
``onnx_light/onnx_op/``, one ``operator_sets_<domain>.cc`` per domain. A schema
is a :cpp:class:`onnx_light::core::LightOpSchema` describing the inputs, outputs,
type constraints and documentation of one ``since_version`` of an operator.


The new schema shows up in :mod:`onnx_light.onnx.defs` (``get_schema``,
``get_all_schemas_with_history``) and in
``onnx_light.onnx_op.get_all_schemas_with_history``:

.. code-block:: python

    import onnx_light.onnx.defs as defs

    schema = defs.get_schema("Abs")

Register a peak memory function
-------------------------------

A peak-memory function estimates the scratch (non-output) memory a kernel needs,
so a scheduler can size buffers. It is optional — operators without one report a
peak of ``0``. The registry mirrors shape inference. The signature is
``int64_t(Device, const std::vector<SymShape> &)``.


Query the estimate through
:func:`onnx_light.onnx_core.shape_inference.compute_peak_memory`, backed by
``RegisterPeakMemoryFunctions``:

.. code-block:: python

    from onnx_light.onnx_core.shape_inference import compute_peak_memory, Device, SymShape

    peak = compute_peak_memory(
        "ai.onnx", "Attention", Device.kCPU,
        [SymShape([2, 4, 8, 16]), SymShape([2, 4, 8, 16])],
    )

See also
--------

* :ref:`l-howto-use-custom-kernel-python` - plug a kernel in at runtime without
  rebuilding the library.
* :ref:`l-howto-use-custom-shape-inference-python` - register a shape function from
  Python or C++ at runtime.
* :ref:`l-howto-collect-backend-test-cases-python` - enumerate and run backend test
  cases.
* :ref:`l-how-to` - other onnx-light how-to recipes.
