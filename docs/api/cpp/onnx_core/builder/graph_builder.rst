graph_builder.h
===============

``ApplyRewritings`` applies the ordered ``RewritingProto`` returned by
``GraphGraph::Optimize`` to a root builder and returns a new ``GraphBuilder``.
The source is unchanged. The result owns merged functions and opset imports,
evaluates initializer recipes once, and exposes inferred shapes and memory
metadata immediately. ``BuildGraph()`` preserves recorded node order;
``ToModel()`` finalizes the graph and includes its functions and imports.

.. doxygenfile:: onnx_core/builder/graph_builder.h
   :project: onnx-light
