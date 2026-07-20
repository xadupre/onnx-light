controlflow
===========

Control-flow operator kernels (``If``, ``Loop``, ``Scan``) live in
``core::runtime`` rather than ``onnx_kernels::kernel`` because running
their subgraphs recursively calls :cpp:func:`onnx_light::core::runtime::RunGraph`,
which must live in ``onnx_core``; keeping them in ``onnx_kernels`` would
require ``onnx_core`` to depend on ``onnx_kernels``.

.. toctree::
    :maxdepth: 1

    include_controlflow_kernels
