.. _l-howto-graph-builder-basics:

:html_theme.sidebar_secondary.remove:

Build and optimize a graph with GraphBuilder
=============================================

:class:`onnx_light.onnx_core.graph_builder.GraphBuilder` incrementally builds
ONNX graphs while resolving operator schemas, inferring shapes, and assigning
unique value names. This walkthrough uses its compact authoring API and then
optimizes a second graph with the standard pattern library.

Create and export a model
-------------------------

Select every opset before adding its nodes. The empty domain is the standard
ONNX domain; a custom operator requires an explicit non-empty domain and an
imported version. It does not require a schema registered on the local machine.

The example includes:

* ``Add`` as a standard operator;
* ``Clip`` with an omitted optional ``min`` input;
* variadic ``Sum`` inputs;
* both outputs of ``TopK``;
* a schema-less ``com.example::CustomNormalize`` operator.

.. runpython::

    from pathlib import Path
    import tempfile

    import numpy as np

    import onnx_light.onnx as onnxl
    import onnx_light.onnx.checker as checker
    from onnx_light.onnx import TensorProto
    from onnx_light.onnx_core.graph_builder import GraphBuilder

    g = GraphBuilder("authoring")
    g.set_opset_version("", 18)
    g.set_opset_version("com.example", 7)

    x = g.inp("X", TensorProto.FLOAT, [4])
    bias = g.init(np.array([1, 2, 3, 4], dtype=np.float32), name="bias")
    added = g.op.Add(x, bias)
    clipped = g.op.Clip(added, None, np.array(6, dtype=np.float32))
    summed = g.op.Sum(clipped, x, bias)
    k = g.init(np.array([2], dtype=np.int64), name="k")
    values, indices = g.op.TopK(
        summed, k, outputs=["values", "indices"], axis=0
    )
    normalized = g.op.CustomNormalize(
        values,
        domain="com.example",
        outputs="Y",
        epsilon=1e-5,
    )
    g.out(normalized, TensorProto.FLOAT, [2])

    model = g.to_onnx("model")
    checker.check_model(model)

    with tempfile.TemporaryDirectory() as temporary_directory:
        path = Path(temporary_directory) / "model.onnx"
        onnxl.save(model, path)
        loaded = onnxl.load(path)
        checker.check_model(loaded)
        assert loaded.SerializeToString() == model.SerializeToString()

    opsets = {opset.domain or "ai.onnx": opset.version for opset in model.opset_import}
    custom = next(node for node in model.graph.node if node.domain == "com.example")
    print(f"opsets: {opsets}")
    print(f"custom node: {custom.domain}::{custom.op_type}")
    print(f"TopK outputs: {values}, {indices}")
    print("model validated and round-tripped")

``g.inp`` declares and returns an input name, ``g.init`` adds a NumPy
initializer, ``g.op.<Operator>`` adds a node, and ``g.out`` declares an output.
Operator inputs can be value names, NumPy arrays, or ``None`` for an omitted
optional input. The ``outputs`` option accepts one name, a sequence of names, or
a positive output count.

The compact helpers delegate to the explicit
:meth:`~onnx_light.onnx_core.graph_builder.GraphBuilder.make_input`,
:meth:`~onnx_light.onnx_core.graph_builder.GraphBuilder.make_initializer`,
:meth:`~onnx_light.onnx_core.graph_builder.GraphBuilder.make_node`, and
:meth:`~onnx_light.onnx_core.graph_builder.GraphBuilder.make_output` methods.
Those ``make_*`` methods are the complete low-level contract for generated code
and advanced authoring.

Optimize and replay a rewrite
-----------------------------

The following model contains a redundant ``Cast`` from ``float`` to ``float``.
Selecting only the standard ``Cast`` pattern makes the result deterministic:
one :class:`~onnx_light.onnx_core.optimization.LocalRewriting` replaces it with
``Identity``.

.. runpython::

    import onnx_light.onnx.checker as checker
    from onnx_light.onnx import TensorProto
    from onnx_light.onnx_core.graph_builder import GraphBuilder
    from onnx_light.onnx_core.optimization import (
        GraphGraph,
        replay,
        standard_patterns,
    )
    source = GraphBuilder("optimization")
    source.set_opset_version("", 18)
    x = source.inp("X", TensorProto.FLOAT, [4])
    y = source.op.Cast(x, outputs="Y", to=TensorProto.FLOAT)
    source.out(y, TensorProto.FLOAT, [4])
    original = source.to_onnx("model")

    builder = GraphBuilder(original)
    graph = GraphGraph(
        builder,
        standard_patterns(["Cast"]),
        use_global_patterns=False,
    )
    rewrites, report = graph.optimize(report=True)
    optimized_graph = builder.build_graph()

    assert len(rewrites) == 1
    assert report.rewrites == 1
    cast_report = next(item for item in report.patterns if item.pattern_name == "Cast")
    print(
        f"Cast: {cast_report.matches} match(es) over "
        f"{cast_report.attempts} attempt(s)"
    )
    print(repr(rewrites[0]))

    replayed_graph = replay(original, rewrites)
    assert replayed_graph.SerializeToString() == optimized_graph.SerializeToString()

    optimized_model = builder.to_onnx("model")
    checker.check_model(optimized_model)
    print("replay reproduced the optimized graph")

``report`` aggregates attempts, matches, rejections, and timings by pattern.
Each returned ``LocalRewriting`` is also a replayable record of the matched and
added nodes, their positions, initializer changes, and value renames.
