shapes
======

The ``shapes`` sub-namespace of ``onnx_core`` (``core::shapes``) hosts the
generic shape-inference *engine*: :cpp:class:`ShapesContext`, the node/graph
traversal (``ComputeShapeNode``, ``ComputeShapeGraph``,
:cpp:func:`InferShapesModel`), broadcasting and node-checking helpers, and
the dispatch table that maps an operator (domain, op_type) pair to the
function that computes its output shapes.

``onnx_core`` never depends on ``onnx_shapes``, so the dispatch table starts
out empty: it is a mutable registry
(:cpp:func:`RegisterComputeShapeFn`) that ``onnx_shapes`` populates with its
per-operator ``ComputeShape*`` functions (see
:doc:`../../onnx_extensions/onnx_shapes/dispatch_table`) via
:cpp:func:`onnx_light::onnx_shapes::RegisterShapeFunctions`. Any consumer of
the shape-inference engine (Python bindings, tests, examples, ...) must call
that function once before using :cpp:func:`InferShapesModel` or
:cpp:class:`ShapesContext`.

.. toctree::
    :maxdepth: 1

    shapes_context
    shape_broadcast
    shape_check
    shape_inference
    dispatch_table
