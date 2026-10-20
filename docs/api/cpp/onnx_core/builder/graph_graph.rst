graph_graph.h
=============

``Replay`` coalesces consecutive node-only edits in the same graph when they
replace original nodes at their first matched position and preserve all
externally used tensor descriptors. It validates each replacement's inference
using only its input descriptors, then applies the combined edit with one graph
analysis. Live-rank queries translate sequential positions without rebuilding
the node order for every record.

Edits consuming previously added nodes or reading values produced by earlier edits,
changing boundary descriptors, modifying initializers or renames, replacing nested
graphs, or calling model-local functions or context-specific inference callbacks
end the batch and retain ordinary sequential validation. This includes replacing
an existing constant under the same name before introducing a new consumer.
Local functions take precedence even when they share a native operator's name.
This keeps subgraph identities and invalid-plan diagnostics while avoiding
repeated full analyses for independent replacements.

.. doxygenfile:: onnx_core/builder/graph_graph.h
   :project: onnx-light
