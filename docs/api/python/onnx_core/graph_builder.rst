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

Builder-local operator registries
--------------------------------

``OperatorRegistry`` supplies local schemas and shape-inference callbacks without
changing process-wide registrations::

    from onnx_light.onnx_core.graph_builder import GraphBuilder, OperatorRegistry

    def infer_custom(ctx, node):
        ctx.set(str(node.output[0]), ctx.get(str(node.input[0])))

    registry = OperatorRegistry()
    registry.register_shape_inference("custom", "CustomOp", infer_custom)
    builder = GraphBuilder(
        "g", opsets={"": 18, "custom": 1}, local_registry=registry
    )

``registry.register_schema(schema)`` adds or replaces a
``LightOpSchema`` identified by domain, name and version. Schema lookup chooses
the newest applicable local schema before consulting the configured global
provider. A local history containing only newer versions falls back to a
compatible global schema; if neither applies, it raises instead of bypassing
validation. The compact ``builder.op`` API uses the same lookup.

Shape resolution prioritizes model-local functions, then local callbacks,
then global inference. A local callback failure propagates without retrying
the global implementation. Callbacks receive the current ``ShapesContext``
and node, so they can inspect the selected opset.

Builders snapshot the registry before importing any nodes. Subgraphs, local
functions, optimization copies and replay/undo copies inherit that snapshot.
Editing the original registry affects only builders constructed afterwards;
``builder.local_registry()`` returns an independent copy. Callback captures
retain ordinary Python callable / C++ ``std::function`` ownership semantics.
Registrations are not serialized into ONNX: pass the registry again when
importing a saved model. The standalone ONNX checker keeps its own schema
registry and is not modified by these builder-local registrations.

.. automodule:: onnx_light.onnx_core.graph_builder
    :members:
    :imported-members:
