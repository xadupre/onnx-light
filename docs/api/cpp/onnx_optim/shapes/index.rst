shapes
======

The ``shapes`` sub-namespace of ``onnx_optim`` hosts the per-operator
shape-inference functions (``ComputeShape*``). Each function consumes
a :cpp:class:`ShapesContext` (a name → :cpp:class:`OptimTensor` map),
a ``NodeProto`` and the names of the input values to read, and writes
the descriptors of the node's outputs back into the context.

Concrete functions are organised per operator domain (currently only
``math``).

.. toctree::
    :maxdepth: 1

    shapes_context
    math/index
