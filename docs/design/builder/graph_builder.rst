.. _l-design-graph-builder:

GraphBuilder
============

``onnx-light`` builds ONNX graphs, models and functions incrementally
through ``core::builder::GraphBuilder`` (C++) and its Python wrapper
:class:`onnx_light.onnx_core.graph_builder.GraphBuilder`. The builder is
the entry point of the *core* pipeline: it accumulates nodes, resolves
their opsets, validates them against the built-in operator schemas, runs
incremental shape inference and finalises everything into a proto.

Overview
--------

A builder starts empty and holds a compute context. It records every
value name it hands out so a name is **never reused**, which keeps the
successor and predecessor maps valid while the graph grows and while
patterns rewrite it (see :ref:`l-design-optimization`). Nodes are added
with :meth:`~onnx_light.onnx_core.graph_builder.GraphBuilder.make_node`,
which:

* resolves the operator opset for the requested domain;
* validates the node against the built-in ONNX operator schemas;
* assigns output names when the caller leaves them empty;
* runs incremental shape inference so every value has an inferred type
  and shape as soon as it is created.

:meth:`~onnx_light.onnx_core.graph_builder.GraphBuilder.to_onnx`
finalises the accumulated graph into a model (default), a graph or a
function, writing back the inferred shapes, the in-place /
release-after metadata, the value tags and the peak-memory estimates.

Typical usage
-------------

.. code-block:: python

    from onnx_light.onnx_core.graph_builder import GraphBuilder
    from onnx_light.onnx_proto import TensorProto

    builder = GraphBuilder("g")
    builder.make_input("x", TensorProto.FLOAT, [2, 3])
    builder.make_input("y", TensorProto.FLOAT, [2, 3])
    (z,) = builder.make_node("Add", ["x", "y"])
    builder.make_output(z)
    model = builder.to_onnx("model")

The same builder can be constructed from an existing ``ModelProto`` to
optimize or extend a model that was produced elsewhere.

Relation to the rest of the core pipeline
-----------------------------------------

The builder is the shared foundation of the other core components:

* **Pattern optimization** rewrites the graph held by a builder; the
  optimizer reuses the builder's shape and type inference, its constant
  knowledge and its cleanup passes instead of duplicating them. See
  :ref:`l-design-optimization`.
* **Shape inference** is the same engine the builder invokes
  incrementally; the standalone entry point is described in
  :ref:`l-design-shape-inference`.
* **Constant folding** replaces subgraphs whose inputs are all constant
  by their computed value, driven by the runtime kernels described in
  :ref:`l-design-runtime`.

API reference
-------------

* **Python API**:
  :class:`onnx_light.onnx_core.graph_builder.GraphBuilder`.
* **C++ API**: :doc:`/api/cpp/onnx_core/builder/index`.

Examples
--------

* :ref:`l-example-plot-pretty-onnx` inspects a model built and rendered
  through the builder.
* :ref:`l-example-plot-compute-context-memory` and
  :ref:`l-example-plot-initializer-statistics` use the builder to report
  the peak-memory and initializer statistics it estimates.

