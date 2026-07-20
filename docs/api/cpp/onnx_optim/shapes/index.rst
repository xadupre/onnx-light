shapes
======

The ``shapes`` sub-namespace of ``onnx_optim`` hosts the per-operator
shape-inference functions (``ComputeShape*``). Each function consumes
a :cpp:class:`onnx_light::core::shapes::ShapesContext` (a name →
:cpp:class:`SymTensor` map), a ``NodeProto`` and the names of the input
values to read, and writes the descriptors of the node's outputs back into
the context. ``ShapesContext`` itself, along with the generic
node/graph-traversal engine and the dispatch table, lives in ``onnx_core``
(see :doc:`../../onnx_core/shapes/index`); this module only defines the
built-in operator implementations and registers them with that engine's
dispatch table via :doc:`../dispatch_table`.

Concrete functions are organized per operator domain.

.. toctree::
    :maxdepth: 1

    controlflow/index
    generator/index
    image/index
    logical/index
    math/index
    nn/index
    optional/index
    preview/index
    quantization/index
    reduction/index
    rt/index
    sequence/index
    tensor/index
    text/index
    traditionalml/index
    training/index
