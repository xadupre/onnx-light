shapes
======

The ``shapes`` sub-namespace of ``onnx_optim`` hosts the per-operator
shape-inference functions (``ComputeShape*``). Each function consumes
a :cpp:class:`ShapesContext` (a name → :cpp:class:`OptimTensor` map),
a ``NodeProto`` and the names of the input values to read, and writes
the descriptors of the node's outputs back into the context.

Concrete functions are organized per operator domain.

.. toctree::
    :maxdepth: 1

    shapes_context
    shape_broadcast
    shape_check
    shape_inference
    dispatch_table
    controlflow/index
    generator/index
    logical/index
    math/index
    nn/index
    optional/index
    preview/index
    quantization/index
    reduction/index
    sequence/index
    tensor/index
    text/index
    traditionalml/index
    training/index
