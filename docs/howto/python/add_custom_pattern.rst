:html_theme.sidebar_secondary.remove:

.. _l-howto-add-custom-pattern:
.. _l-howto-add-custom-pattern-python:

How to add a custom graph-rewriting pattern and set its priority
================================================================

This guide covers the Python interface. For the other language, see
:doc:`../cpp/add_custom_pattern`.

This page shows how to write and register a graph-rewriting **pattern** —
the building block of graph optimization
(:cpp:class:`onnx_light::core::builder::PatternOptimization` /
:class:`onnx_light.onnx_core.optimization.PatternOptimization`) — and how
``priority`` orders it relative to the standard patterns. For a runnable,
end-to-end walk-through of optimization statistics see
:ref:`l-example-plot-pattern-optimization`.

A pattern implements three things:

* ``fast_op_type`` / ``FastOpType`` — the set of node ``op_type`` values from
  which the matcher may start (an empty set means "every node"); this keeps
  matching fast by skipping irrelevant candidates.
* ``match`` / ``Match`` — inspects one candidate node (and, through
  ``GraphGraph``, its neighbourhood) and returns either a successful match
  (``self.result(...)`` / a :cpp:class:`MatchResult`) or a rejection with a
  diagnostic (``self.no_match(...)`` / :cpp:func:`PatternOptimization::NoMatch`).
* ``apply`` / ``Apply`` — builds the replacement nodes for one accepted match.

.. contents::
    :local:

Write the pattern
-----------------

The running example collapses two consecutive ``Neg`` nodes into an
``Identity``.


.. code-block:: python

    import onnx_light.onnx.helper as oh
    from onnx_light.onnx_core.optimization import PatternOptimization

    class NegNegPattern(PatternOptimization):
        def __init__(self, priority: int = 1):
            super().__init__(priority=priority, name="NegNeg")

        def fast_op_type(self):
            return {"Neg"}

        def match(self, graph, node):
            if node.domain not in ("", "ai.onnx"):
                return self.no_match(node, "the candidate is not in the default ONNX domain")
            previous = graph.node_before(node.input[0])
            if previous is None or previous.op_type != "Neg" or previous.domain not in ("", "ai.onnx"):
                return self.no_match(node, "the input is not produced by Neg")
            return self.result([previous, node], insert_at=node)

        def apply(self, graph, nodes):
            previous, node = nodes
            return [
                oh.make_node("Identity", [previous.input[0]], list(node.output))
            ]

``self.result(nodes, insert_at=...)`` records the matched nodes (in the
order ``apply`` expects them) and where the replacement should be
inserted; ``self.no_match(candidate, reason)`` records a rejection for
the statistics shown in :ref:`l-example-plot-pattern-optimization`.
A ``None``/``nullptr`` entry may reserve an optional positional role;
the optimizer passes it to ``apply`` but ignores it for positioning,
marking, removal, and replay.

Primitive patterns check the domain and operator type of the candidate and
secondary nodes. Model verification and builder registration reject functions
that reuse a registered operator's domain and name, regardless of ``overload``.
Patterns intentionally matching functions check their expected identities and
semantics before returning the usual ``self.result(...)``.

Define a pattern with two ONNX functions
---------------------------------------

:class:`~onnx_light.onnx_core.optimization.FunctionPattern` builds the matcher
and replacement automatically from two ``FunctionProto`` bodies:

.. code-block:: python

    import onnx_light.onnx.helper as oh
    from onnx_light.onnx_core.optimization import FunctionPattern

    opsets = [oh.make_opsetid("", 18)]
    template = oh.make_function(
        "patterns", "DoubleNeg", ["x"], ["y"],
        [oh.make_node("Neg", ["x"], ["middle"]),
         oh.make_node("Neg", ["middle"], ["y"])],
        opsets,
    )
    replacement = oh.make_function(
        "patterns", "Identity", ["input"], ["output"],
        [oh.make_node("Identity", ["input"], ["output"])],
        opsets,
    )
    pattern = FunctionPattern(template, replacement, name="DoubleNeg")

Function inputs and outputs correspond **by position** and must have equal,
fixed counts. Every node also matches an exact input/output count (including
empty optional slots), domain, overload, literal attributes and opset version.
Input order is significant; commutative permutations are not inferred.

Reference attributes created with ``oh.make_attribute_ref`` capture attributes
instead of comparing a literal value. Declare their ``ref_attr_name`` in the
function's ``attributes`` list. A replacement reference uses the same capture
name and type, but may have a different node-attribute name. Repeated template
references must match equal values. All replacement references must be captured
by the template; missing candidate attributes are not filled from schema defaults.

For example, these nodes capture and reuse a transpose permutation::

    from onnx_light.onnx import AttributeProto

    node = oh.make_node("Transpose", ["x"], ["y"])
    node.attribute.append(
        oh.make_attribute_ref("perm", AttributeProto.INTS, ref_attr_name="permutation")
    )
    # Use attributes=["permutation"] in both function definitions.

Bodies must be topologically sorted, use SSA names, and contain no graph-valued
attributes. Template inputs must be used; its outputs must be distinct node
results. The replacement may return inputs directly or return a value more than
once. Output names are preserved and replacement intermediates get fresh names.
Template nodes must be connected through nonempty values, including shared
function inputs. Multi-output branches are matched by following local producers
and consumers from the last template node; unrelated graph nodes are never
scanned. Templates with completely disconnected components are rejected.
Target nodes with device configurations are rejected rather than losing their
sharding or pipeline-stage assignments.
Removed intermediates cannot be graph outputs, subgraph captures or inputs to
unmatched nodes. Captured node outputs and matches requiring interleaved external
consumers are conservatively rejected, even when the captured value is a template
output.
The target graph must already import the functions' operator opsets; the pattern
does not upgrade opsets or install local function definitions. Forwarding an input
or an aliased output also requires a default ONNX opset for ``Identity``.
As with handwritten patterns, the author is responsible for semantic equivalence,
including supported types, shapes and broadcasting.

This class requires function arguments, so it has no built-in registry factory.
Pass the instance to ``GraphGraph`` or register it with ``register_pattern``.

Select the pattern for one optimizer
------------------------------------

Passing pattern instances directly to ``GraphGraph`` keeps them local to that
optimizer; nothing is shared globally. The list is exclusive: it does not
implicitly include global or builder registrations.


.. code-block:: python

    from onnx_light.onnx_core.optimization import GraphBuilder, GraphGraph

    builder_ = GraphBuilder(model)
    graph = GraphGraph(builder_, patterns=[NegNegPattern()])
    rewrites = graph.optimize()

Register the pattern globally
-----------------------------

A registered pattern becomes available for selection. Device-independent
patterns are included by default, alongside the standard patterns. A
device-specific pattern requires a device selector or an explicit list or regex.


.. code-block:: python

    from onnx_light.onnx_core.optimization import register_pattern, unregister_pattern

    register_pattern(NegNegPattern())
    try:
        graph = GraphGraph(builder_)  # picks up NegNegPattern automatically
        rewrites = graph.optimize()
    finally:
        unregister_pattern("NegNeg")

A pattern can also be registered on one ``GraphBuilder`` only, which
overrides a same-named global pattern for optimizers built over that
builder (``builder_.register_pattern(NegNegPattern())``); see
"Pattern registration" in
:doc:`/api/python/onnx_core/optimization`
for the selection rules and name-override precedence.

Declare a target device
-----------------------

Patterns are device-independent by default. A Python subclass can pass
``device=Device.kCPU`` or ``device=Device.kGPU0`` to ``PatternOptimization.__init__``,
with ``Device`` imported from ``onnx_light.onnx_core.shape_inference``.
In C++, pass the corresponding ``core::symbolic::Device`` as the third
``PatternOptimization`` constructor argument.

``GraphGraph(builder, patterns=Device.kCPU)`` includes independent and CPU-specific
patterns and assigns that target to the builder. An already-defined, different
builder device is an error. A list or regex can deliberately select a
device-specific pattern without modifying the target. Declaring a device controls
automatic selection, not whether a caller may explicitly run the pattern.

Set the priority
----------------

``priority`` (the constructor's first argument, or ``PatternOptimization``'s
public ``priority`` field) controls **when** a pattern is tried relative to
the others sharing the same optimizer: patterns are grouped by priority and
evaluated in **ascending** order, one priority level at a time, restarting
from the lowest whenever a rewrite happens at a higher one. A negative
priority disables the pattern.

The standard patterns illustrate the ordering: ``CastPattern`` runs at
priority ``0`` so a redundant ``Cast`` is simplified to ``Identity`` before
``CastCastPattern`` and the other priority-``1`` patterns look at its
neighbours. Give a custom pattern a lower priority to run it before the
standard patterns, or a higher one to run it only once they have stabilized:


.. code-block:: python

    # Runs before every priority-1 standard pattern.
    early_pattern = NegNegPattern(priority=0)

    # Runs only once every priority-0/1 pattern stops rewriting.
    late_pattern = NegNegPattern(priority=2)

``priority`` can also be changed after construction (Python:
``pattern.priority = 0``; C++: the public ``priority`` field), as long as it
is set before the pattern is passed to ``GraphGraph``.

See also
--------

* :ref:`l-example-plot-pattern-optimization` - runnable optimization and
  statistics example.
* :ref:`l-example-plot-pattern-replay` - dedicated example that captures
  rewrites and replays them from the original model.
* :doc:`/api/python/onnx_core/optimization`
  - Python API reference, including the standard pattern table with their
  priorities.
* :cpp:class:`onnx_light::core::builder::PatternOptimization` - C++ interface
  reference.
* :ref:`l-howto-register-builtins-python` - companion how-to for registering
  built-in kernels, shape functions and schemas (a different extension
  point, unrelated to patterns).
