graph_layout.h
==============

``GraphLayout`` binds a live ``GraphProto`` or ``FunctionProto`` without taking
ownership or copying the body. Call ``Load`` before editing and ``Persist``
before serializing. The bound proto must outlive the layout; geometry and route
queries return borrowed views, unlike the Python snapshot API.

See :doc:`../../../howto/python/incremental_layout` for scope coordinates,
incremental guarantees, persisted metadata and folded-node rendering rules.

.. doxygenfile:: onnx_core/graph/graph_layout.h
    :project: onnx-light
