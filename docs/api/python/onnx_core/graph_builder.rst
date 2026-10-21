onnx_light.onnx_core.graph_builder
==================================

Full builds load the compiled kernel registry when this module is imported,
so constant folding does not depend on importing ``ReferenceEvaluator`` first.
Reduced builds without the kernels extension still support graph authoring.
Required shape constant folding in such builds raises an error identifying the
missing operator kernel; importing the builder does not suppress extension
loading failures when the extension is present.

``GraphBuilder.to_onnx()`` returns a ``ModelProto`` by default. To export another
proto type, use the ``ProtoKind`` enum instead of a string::

    from onnx_light.onnx_core.graph_builder import GraphBuilder, ProtoKind

    builder = GraphBuilder("g")
    model = builder.to_onnx()
    graph = builder.to_onnx(ProtoKind.kGraph)
    function = builder.to_onnx(ProtoKind.kFunction, domain="custom")

``ProtoKind.kModel`` explicitly selects a model. The ``ir_version`` argument
applies to models, and ``domain`` applies to functions. String selectors such as
``"model"``, ``"graph"`` and ``"function"`` are no longer accepted.

The keyword-only ``optimize`` option selects an optimization/export workflow::

    from onnx_light.onnx_core.graph_builder import OptimizationMode

    canonical = builder.to_onnx()
    materialized = builder.to_onnx(optimize=OptimizationMode.kOptimize)
    bundled = builder.to_onnx(optimize=OptimizationMode.kRewritings)
    replayed = GraphBuilder(bundled).apply_rewritings("optimized").to_standard_model()

``OptimizationMode.kNone`` is the default and preserves the existing export
behavior. ``OptimizationMode.kOptimize`` optimizes a copy with the global and
builder-local pattern registrations and returns the optimized result without
rewrite plans. Model exports use ``to_standard_model`` and reject constructs that
standard ONNX cannot represent.
Graph and function exports also accept ``OptimizationMode.kOptimize``.
Function export represents dense initializers as leading ``Constant`` nodes,
including initializers introduced by optimization, without changing the source
builder. Tensor payloads and external-data descriptors are preserved. This
requires a default-domain opset import. Overridable input defaults and extended
initializers cannot be represented this way and are rejected explicitly.

``OptimizationMode.kRewritings`` returns the finalized, non-optimized canonical
model with the captured plan in ``ModelProto.rewritings["optimized"]``. This mode requires
``ProtoKind.kModel``; graphs and functions cannot contain the map. Other attached
plans are retained, but an existing ``"optimized"`` key is an error rather than
being overwritten. Loading a bundle does not execute its plan.
The option accepts only ``OptimizationMode`` members, not booleans, integers,
strings, or members of another enum.

Both optimization modes preserve the source builder's graph and registrations
after ordinary export finalization. The working copy retains each scope's
device, including subgraphs with a different target from their parent and
exported local functions. Both modes keep a canonical copy rather than trying
to invert destructive edits. Builder-local Python patterns remain live
callbacks during capture; only their resulting proto records are serialized.

Load external weights with ``onnx_light.onnx.load(path, load_external_data=True)``
before optimization. Save either exported model with the existing
``onnx_light.onnx.save(..., save_as_external_data=True)`` to write its weights
beside the new model instead of retaining references to the old directory.
Deferred loading through ``load_external_data_for_model`` and streaming
alignment also cover function defaults and rewrite payloads, including tensors
in repeated tensor and graph attributes.
See :ref:`l-example-plot-pattern-replay` for a save/load/replay example.

.. automodule:: onnx_light.onnx_core.graph_builder
    :members:
    :imported-members:
