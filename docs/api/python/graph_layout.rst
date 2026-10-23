Graph and function layout
=========================

The native layout API binds directly to a ``GraphProto`` or ``FunctionProto``.
It preserves semantic node order and maintains scope-local coordinates, stable
IDs and ordered input routes. Python geometry, route and statistics queries
return snapshots; child layouts are live handles that retain the root proto.
The C++ API instead returns borrowed geometry and route views.

See :doc:`../../howto/python/incremental_layout` for persistence, mutation
locality, child-handle invalidation and folded-node rendering rules, and
:ref:`l-example-plot-graph-layout` for a runnable visualization of graph and
function positions.

.. autoclass:: onnx_light.onnx.GraphLayout
    :members:

.. autoclass:: onnx_light.onnx.LayoutNode
    :members:

.. autoclass:: onnx_light.onnx.LayoutEdge
    :members:

.. autoclass:: onnx_light.onnx.LayoutPoint
    :members:

.. autoclass:: onnx_light.onnx.LayoutStats
    :members:

.. autofunction:: onnx_light.onnx.enable_graph_layout

``enable_graph_layout`` chains both ``ParseOptions.graph_callback`` and
``ParseOptions.function_callback``. Binary loading with ``layout=True`` enables
these hooks automatically; textproto loading places every graph and local
function before returning.
