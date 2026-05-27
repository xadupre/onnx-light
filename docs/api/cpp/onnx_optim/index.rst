onnx_optim
==========

The ``onnx_optim`` library provides lightweight, non-owning data
structures used by ONNX graph optimisation passes. The central types
are :cpp:class:`OptimDim`, :cpp:class:`OptimShape` and
:cpp:class:`OptimTensor`, which together describe a tensor whose shape
may be fully known, fully symbolic, or any mix in between.

Unlike a fully featured tensor type, :cpp:class:`OptimTensor` never
allocates memory: it stores a raw ``void *`` pointer to a buffer owned
by the caller, the element :cpp:type:`TensorType` and an
:cpp:class:`OptimShape`. This makes it suitable for use during
shape-inference and rewrite passes where the underlying constant data
is already materialised elsewhere (initialisers, attribute payloads,
etc.) and only needs to be referenced.

.. toctree::
    :maxdepth: 1

    optim_tensor
    optim_sequence
    expressions
    shapes/index
