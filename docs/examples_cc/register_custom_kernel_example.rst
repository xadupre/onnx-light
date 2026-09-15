.. _l-cpp-register-custom-kernel-example:

Standalone C++ example: register a new kernel for an existing operator
======================================================================

This page documents ``examples/register_custom_kernel``
(`view on GitHub <https://github.com/xadupre/onnx-light/tree/main/examples/register_custom_kernel>`_),
a self-contained CMake project that shows how to implement a brand-new C++
kernel **class** for an operator that onnx-light already ships, install it into
onnx-light's shared kernel dispatch table, run a model that uses that operator
and verify the new kernel is the one actually executed.

This is exactly the scenario implemented by the companion
`onnx-light-cpu <https://github.com/xadupre/onnx-light-cpu>`_ project, which
ships SIMD-accelerated ``Abs`` / ``Exp`` / ``Log`` / ``Gemm`` / ``Not`` kernels
as :cpp:class:`onnx_light::core::runtime::KernelBase` subclasses and installs
them into onnx-light's dispatch table so *any* model using those operators runs
the optimized kernels instead of the built-in ones. The example implements a
single, self-contained ``Abs`` replacement to keep it short.

How it works
------------

The example has three parts:

* ``ExampleAbsKernel`` — a :cpp:class:`onnx_light::core::runtime::KernelBase`
  subclass computing the element-wise absolute value of a ``FLOAT`` tensor.
  Like every built-in kernel it exposes a
  ``static constexpr const char *name`` identifier
  (``"example:CPU:ai.onnx:Abs"``) following the
  ``"<library>:<device>:<domain>:<op_type>"`` convention used by onnx-light's
  own kernel classes (e.g. ``"onnx_kernels:CPU:ai.onnx:Abs"``). Custom kernels
  use their own library prefix so their name never collides with a built-in
  one.
* ``RegisterExampleAbsKernel`` — installs a factory for the kernel via
  :cpp:func:`onnx_light::core::runtime::RegisterKernelFn` for the CPU device and
  the default ONNX domain, overriding the built-in ``Abs`` entry.
* ``main`` — registers the built-in kernels with
  :cpp:func:`onnx_light::onnx_kernels::RegisterKernelFunctions`, installs the
  override, builds a one-node ``Abs`` graph, runs it through a
  :cpp:class:`onnx_light::core::runtime::RuntimeSession` and checks both that
  the output equals ``|x|`` and that ``ExampleAbsKernel`` — not the built-in —
  produced it (a run counter is bumped on every dispatch).

Registration order does not matter
-----------------------------------

An explicit :cpp:func:`onnx_light::core::runtime::RegisterKernelFn` call
replaces any existing entry for the same ``(domain, op_type, device)``
identifier, while the bulk built-in registration performed by
:cpp:func:`onnx_light::onnx_kernels::RegisterKernelFunctions` never clobbers a
kernel that was already registered (it registers each built-in only when the
slot is still empty). As a result a downstream override wins whether it is
installed before or after the built-ins are registered.

For context-local registration, pass the same factory to
``rt.RegisterKernelFn(domain, op_type, device, factory)`` instead. Local
factories override global ones and propagate into subgraph/function contexts.
Both scopes create one session-owned kernel per resolved node and reuse it
across runs, even when the input shape changes. Register replacements before
resolution: replacing a registration never invalidates an existing kernel.
The original model must outlive the session. Independent sessions must not
share mutable kernel state; concurrent runs on one session are not supported.

Step 1 -- Install the onnx_light C++ library
---------------------------------------------

From the *onnx-light* repository root, build and install the static library and
its public headers (the Python extension is not needed):

.. code-block:: bash

    cmake -S . -B build-install \
          -DCMAKE_BUILD_TYPE=Release \
          -DONNX_LIGHT_BUILD_PYTHON=OFF \
          -DCMAKE_INSTALL_PREFIX=/usr/local
    cmake --build  build-install
    cmake --install build-install

Step 2 -- Build the example
---------------------------

.. code-block:: bash

    cmake -S examples/register_custom_kernel -B build-register-custom-kernel \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_PREFIX_PATH=/usr/local
    cmake --build build-register-custom-kernel

Step 3 -- Run the example
-------------------------

.. code-block:: bash

    ./build-register-custom-kernel/register_custom_kernel

It prints the registered kernel name, the computed output and a ``PASS`` line
confirming the custom kernel ran:

.. code-block:: text

    Registered custom kernel class 'example:CPU:ai.onnx:Abs' for op_type 'Abs' (default domain, CPU device).
    y = [1, 2, 3.5, 0]
    PASS: the custom 'example:CPU:ai.onnx:Abs' kernel ran and produced the expected output.

One-shot script
---------------

To install onnx_light and build the example in one go:

.. code-block:: bash

    bash examples/register_custom_kernel/build.sh

On Windows:

.. code-block:: bat

    examples\register_custom_kernel\build.bat

See also
--------

* :ref:`l-howto-use-custom-kernel` — context-local factories and Python/C++
  callback convenience APIs using the same session-owned kernel lifecycle.
