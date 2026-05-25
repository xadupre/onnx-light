shapes
======

The ``shapes`` sub-namespace of ``onnx_optim`` hosts *shape kernels*:
small objects that perform the shape (and dtype) inference for a single
``NodeProto``. Each kernel consumes one or more
:cpp:class:`OptimTensor` views and produces one or more
:cpp:class:`OptimTensor` views describing the operator outputs.

Kernels are instantiated from a ``NodeProto`` through the
:cpp:func:`MakeShapeKernel` factory, and concrete implementations are
organised per operator domain (currently only ``math``).

.. toctree::
    :maxdepth: 1

    shape_kernel
    math/index
