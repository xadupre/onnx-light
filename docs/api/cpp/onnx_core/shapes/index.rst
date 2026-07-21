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

Peak-memory estimation
----------------------

A second, parallel registry estimates each operator's peak *computation*
memory rather than its output shapes. Mirroring the shape dispatch table, it
maps an ``(domain, op_type)`` pair to a
:cpp:type:`core::shapes::ComputePeakMemoryFn` — a function that takes the
:cpp:enum:`Device` the operator runs on followed by the
:cpp:class:`SymShape` of each input and returns the estimated scratch memory
in bytes. Functions are registered with
:cpp:func:`RegisterComputePeakMemoryFn` and looked up through
:cpp:func:`ComputePeakMemory`; operators without a registered function report
``0`` by default. ``onnx_shapes`` populates the built-in estimators (for
example ``Attention``) via
:cpp:func:`onnx_light::onnx_shapes::RegisterPeakMemoryFunctions`, and the
Python bindings expose :func:`compute_peak_memory` together with the
``Device`` enum.

.. toctree::
    :maxdepth: 1

    shapes_context
    shape_broadcast
    shape_check
    shape_inference
    dispatch_table
