onnx_light.onnx_core.optimization
=================================

.. currentmodule:: onnx_light.onnx_core.optimization

Optimization workflow
+++++++++++++++++++++

Optimization always operates on a
:class:`~onnx_light.onnx_core.graph_builder.GraphBuilder` through
:class:`~onnx_light.onnx_core.optimization.GraphGraph`:

.. code-block:: python

    from onnx_light.onnx_core.optimization import GraphBuilder, GraphGraph

    builder = GraphBuilder(model)
    graph = GraphGraph(builder)
    rewrites, report = graph.optimize(report=True)
    optimized_model = builder.to_onnx()

Independent model deltas
++++++++++++++++++++++++

``diff_models(canonical, target, base_dir=..., target_dir=...)`` constructs a
``RewritingProto`` from independently produced models. Comparison and replay
run in C++ (``GraphBuilder::DiffRewritings``); consumers do not implement a graph
diff or a replay engine. The returned plan already contains validated native
``hash_before`` and ``hash_after`` compatibility hashes and can be passed
directly to :func:`apply_rewritings_to_model`.

``source_builder.diff_rewritings(target)`` also accepts a ``ModelProto`` or a
root ``GraphBuilder``. The native builder overload compares both existing
builders directly without exporting and reimporting the target, and preserves
both builders.

**Nested rewrites and reference attributes.**

Nested inference must observe enclosing target values and descriptors together.
The previously fixed parent-initializer/If case remains covered by
``test_nested_edits_use_replaced_parent_initializer_values``. The additional
``test_deep_captures_use_replaced_values_and_descriptors`` regression covers two
If levels, multiple captures, graph-input defaults, unchanged consumers, and
rewritten enclosing producers. Constant replacement and Constant-to-initializer
replacement reproduced stale captured shapes; affected enclosing nodes are now
replaced atomically, and nested builders are imported after the existing
incremental refresh publishes the target producer state. Loop body inputs retain
their inferred descriptors as well. Source and target builders remain unchanged.

Only resolved internal subgraph references are normalized during matching.
Declared semantic STRING/STRINGS attributes remain exact, including Quantize's
``parameter_ref``; generated internal aliases avoid semantic attribute names.
Harmless GRAPH/GRAPHS renaming is still supported. Structural compatibility
hashes intentionally omit ordinary attribute payloads, so equal hashes do not
establish semantic equivalence. Final verification compares materialized node
attributes independently of the matching skeleton. The numerical
``test_model_diff_preserves_semantic_parameter_reference`` regression uses two
valid parameter sets with different scales: the previously empty diff must now
reconstruct the target reference and its different numerical result.

Named plans use the existing ``ModelProto.rewritings`` representation:

.. code-block:: python

    import onnx_light.onnx as onnx
    from onnx_light.onnx_core.optimization import (
        apply_rewritings_to_model,
        diff_models,
    )

    canonical.add_rewriting("cpu", diff_models(canonical, cpu_model))
    canonical.add_rewriting("cuda", diff_models(canonical, cuda_model))
    onnx.save(canonical, "artifacts/model.onnx")
    loaded = onnx.load("artifacts/model.onnx")
    cpu = apply_rewritings_to_model(loaded, loaded.rewritings["cpu"])
    cuda = apply_rewritings_to_model(loaded, loaded.rewritings["cuda"])

No proto fields or separate reference-file format are introduced. A target is
reconstructed from its source model and a list of native rewrites, with normal
pre/post structural-hash validation and no fallback.

Dense initializer additions, removals, same-name replacements and ordering,
node ordering, empty optional output slots, nested graph edits, and local
function additions are supported. Initializer permutations are ordinary rewrites
whose ``initializer_calls`` invoke a local forwarding function and whose
``computed_initializer_positions`` specify the resulting positions. Forwarding
functions with no nodes and outputs referring directly to their inputs reuse
the tensor storage and external byte ranges without materialization. New
forwarding functions needed only during initializer preparation remain in the
rewriting plan, not in the reconstructed model. Functions also used by runtime
nodes are retained.

Two different definitions of the same local-function identity
``(domain, name, overload)`` are unsupported and raise an error. Removing local
functions or reordering graph-input defaults is also unsupported.

Input/output names, shapes, dtypes, effective value tags, persistent bindings,
sparse/extended initializer declarations, model declarations and user metadata
must remain compatible. Existing opset versions cannot be removed or changed;
external sparse, encoded and paged-cache declarations additionally require
matching source storage directories during diff construction.
New required domains are added to the plan. Unsupported differences raise
explicit errors rather than adapting interfaces (in particular, no implicit
FP32/FP16 conversion). Model-declaration errors list every differing
declaration and opset with its source and target values; interface errors name
each mismatched input or output with both tensor types. Producer strings, graph/node names and documentation,
inferred intermediate value-info shapes, and regenerated lifetime, constant,
in-place and peak-memory annotations are nonsemantic diagnostics. Effective
value tags are compared, not ignored. Changes to other intermediate value
metadata are rejected.

Comparison indexes nodes and initializers by identity rather than performing
quadratic sequence alignment. Equal external references with equal metadata
reuse the source storage without opening payload files. ``base_dir`` and
``target_dir`` locate the respective external payloads when comparing different
references, including identical filenames in different directories. Such
comparisons materialize the two compared tensors, not the complete checkpoint.
This storage-aware comparison also covers tensor attributes, local-function
bodies and defaults. Sparse initializer contents must also match; changes are
rejected. Deferred tensors reuse a fast path when their backing-file snapshots
and byte ranges match, even across independent loads. Copied deferred payloads
are loaded and detached so the plan can be serialized independently.
Changed or new dense initializers, tensor attributes
and nested graph weights from another directory are embedded in the plan so
replay cannot accidentally read the source model's file with the same name.
Retained external references preserve their locations, offsets and
lengths; callers must keep their backing files available.
Replay performs the usual graph import, shape inference and validation;
constants needed for inference may therefore require materialization.

Optimization callback
+++++++++++++++++++++

Pass ``callback(graph, match)`` to ``optimize`` to report eligible pattern matches
or reject individual matches before their nodes are reserved:

.. code-block:: python

    def on_match(graph, match):
        print(graph.name, match.pattern.name, len(match.nodes))
        return match.pattern.name != "TransposeTranspose"

    rewrites = graph.optimize(callback=on_match)

Returning ``False`` rejects the match; ``True`` or ``None`` accepts it.
Rejected matches do not block other patterns from matching the same nodes.
The callback also runs in nested subgraphs, receiving the corresponding graph.
It observes matches, not completed rewrites: accepted matches are revalidated
before application and matches may be reconsidered in subsequent iterations.
Cleanup passes are unaffected, and no callback runs when there are no eligible
matches. Callback exceptions propagate.

The callback must not mutate the graph, patterns, or match, or retain borrowed
graph or node references beyond the call. C++ callers can pass an
``OptimizationCallback`` as the fifth argument to ``GraphGraph::Optimize``;
its boolean result accepts or rejects the match.

Parallel matching
+++++++++++++++++

Pass ``parallel=True`` to match explicitly opted-in native patterns concurrently
when at least 32 candidates are eligible. Match acceptance, optimization callbacks
and rewrite application remain sequential and preserve candidate order. Python-defined
patterns are always matched sequentially on the calling thread.
Mixing native and Python-defined patterns does not disable native parallel matching.
The C++ ``parallel`` argument follows the optional optimization callback.

Native patterns are sequential by default. ``CanMatchInParallel`` may return true
only after auditing the matcher's state and every graph query it uses for
concurrent access. Matchers must not mutate the pattern, builder, graph, or shared
caches. ``TransposeTranspose`` opts in because it reads only immutable nodes and
the frozen structural index; unaudited patterns remain sequential.

Pattern registration
++++++++++++++++++++

The available patterns are merged by their stable
:attr:`PatternOptimization.name`. Builder registrations replace global entries
with the same name before selection:

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Scope
     - Registration
     - Selection
   * - Global
     - :func:`register_pattern`
     - Makes a pattern available for selection. The standard ONNX patterns
       are registered globally when this module is imported.
   * - Builder
     - :meth:`GraphBuilder.register_pattern`
     - Overrides a global pattern for optimizers built over that builder.
   * - Graph
     - ``GraphGraph(builder, patterns=[...])``
     - Selects only the given names and instances. Explicit instances override
       earlier entries with the same name and are retained for that optimizer,
       including recursive subgraphs.

:func:`clear_registered_patterns` clears the global
registry; :func:`reset_registered_patterns` restores the standard patterns.

Pattern selection
+++++++++++++++++

``GraphGraph(builder, patterns=...)`` accepts one selector:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Selector
     - Selected patterns
   * - ``None`` (default)
     - All available device-independent patterns, even if ``builder.device``
       is already set.
   * - ``False`` or ``[]``
     - None, including builder registrations. The optimizer's ordinary
       cleanup and constant-folding passes still run.
   * - A concrete ``Device``
     - All device-independent patterns plus those targeting that exact device.
       Also sets ``builder.device`` and subgraph targets. A different,
       already-defined device in any of those builders raises ``ValueError``
       before changing any target.
   * - A regex string or compiled ``re.Pattern[str]``
     - Available names accepted by ``fullmatch``, including device-specific
       patterns. Use ``".*MatMul.*"`` to select names containing ``MatMul``.
   * - An iterable of names and/or pattern instances
     - Only those patterns, including device-specific ones. Names are exact,
       not regexes. Standard names may also be instantiated when absent from
       the global registry.

Regexes and explicit lists do not modify ``builder.device`` or automatically
add global or builder patterns. A regex matching nothing selects no patterns;
invalid regexes and invalid selector types raise exceptions. ``True`` is not
a selector, and ``Device.kUndefined`` is not a concrete device: use ``None``
for the default selection.
During recursive optimization, a subgraph with an undefined device inherits
its parent builder's device; explicitly defined subgraph devices are preserved.

.. code-block:: python

    import re
    from onnx_light.onnx_core.shape_inference import Device

    graph = GraphGraph(builder)                         # device-independent defaults
    graph = GraphGraph(builder, patterns=False)         # cleanup only
    graph = GraphGraph(builder, patterns=Device.kCPU)    # defaults + CPU patterns
    graph = GraphGraph(builder, patterns=r".*MatMul.*")  # full name regex
    graph = GraphGraph(builder, patterns=re.compile(r"Cast.*"))
    graph = GraphGraph(builder, patterns=["Cast", "TransposeMatMul"])

A pattern's ``device`` defaults to ``Device.kUndefined``, meaning
device-independent. Custom Python patterns declare a target with
``super().__init__(name="MyCPUFusion", device=Device.kCPU)``; native patterns
use the corresponding third ``PatternOptimization`` constructor argument.
Device equality is exact, not a GPU-family or execution-provider capability
query. Existing standard patterns remain device-independent, including patterns
whose matching heuristics inspect ``builder.device``.

The former ``use_global_patterns`` argument is removed. Replace an explicit
list plus ``use_global_patterns=False`` with just that list, use ``False`` for
no patterns, or pass ``builder.registered_patterns()`` for builder-only patterns.
To combine defaults with explicit patterns, supply the combined list explicitly,
for example ``[*standard_patterns(), custom_pattern]`` (which intentionally
includes every standard pattern, regardless of device).

Registered standard patterns
++++++++++++++++++++++++++++

The following table lists the standard patterns registered when this module is
imported. It is generated from the live registry, so it always reflects the
currently available patterns.

.. runpython::
    :rst:

    from onnx_light.onnx_core.optimization import render_rst_standard_patterns_table

    print(render_rst_standard_patterns_table())

The runtime list is available through :func:`standard_pattern_names`.
Use :func:`standard_pattern_sets` to select the named ``all``, ``L1``, ``L2``,
and ``Layer`` groups registered by the C++ pattern registry and exposed through
Python. ``L1`` lists the fusion patterns shared with ONNX Runtime;
``L2`` lists onnx-light-only standard patterns, and ``Layer`` selects the
opt-in repeated-layer pattern. ``all`` contains the union of these groups.
Add another group with :func:`register_pattern_set`, add members to any built-in
group with :func:`extend_pattern_set`, and remove custom groups with
:func:`unregister_pattern_set`. Groups can overlap. Members may be standard
patterns or patterns previously added with :func:`register_pattern`.
``standard_patterns(pattern_sets["custom"])`` creates the selected instances.
The :ref:`complete pattern catalogue <l-api-pattern-catalog>` in the
:ref:`ByOp catalogue <l-onnx-operators>` adds the C++ documentation link and
the Before/After rewrite graph for every entry.

See :ref:`l-howto-add-custom-pattern` for a Python/C++ how-to on writing a
custom pattern and choosing its priority, and
:ref:`l-example-plot-pattern-optimization` for a runnable example covering
optimization statistics. Replay is demonstrated separately in
:ref:`l-example-plot-pattern-replay`.

Custom Python pattern
+++++++++++++++++++++

.. code-block:: python

    import onnx_light.onnx.helper as oh
    from onnx_light.onnx_core.optimization import (
        GraphBuilder,
        GraphGraph,
        PatternOptimization,
    )

    class NegNegPattern(PatternOptimization):
        def __init__(self):
            super().__init__(priority=1, name="NegNeg")

        def fast_op_type(self):
            return {"Neg"}

        def match(self, graph, node):
            if node.domain not in ("", "ai.onnx"):
                return self.no_match(node, "the candidate is not in the default ONNX domain")
            previous = graph.node_before(node.input[0])
            if previous is None or previous.op_type != "Neg" or previous.domain not in ("", "ai.onnx"):
                return self.no_match(node, "input is not produced by Neg")
            return self.result([previous, node], insert_at=node)

        def apply(self, graph, nodes):
            previous, node = nodes
            return [
                oh.make_node(
                    "Identity", [previous.input[0]], list(node.output)
                )
            ]

    builder = GraphBuilder(model)
    builder.register_pattern(NegNegPattern())
    graph = GraphGraph(builder)
    rewrites = graph.optimize()

API
+++

.. automodule:: onnx_light.onnx_core.optimization
    :members:
    :imported-members:
