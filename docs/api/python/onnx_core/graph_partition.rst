onnx_light.onnx_core.graph_partition
====================================

``propose_operator_groups`` identifies bodies suitable for local functions with
small input/output transfers, even when their intermediate tensors are large.
It returns proposals without changing the model or installing functions:

.. code-block:: python

    from onnx_light.onnx_core.graph_partition import propose_operator_groups

    groups = propose_operator_groups(model)
    for group in groups:
        nodes = [model.graph.node[i] for i in group.node_indices]
        print(group.inputs, group.outputs, group.transfer_bytes)

Graph analysis, size accounting and grouping run in C++; the Python facade
validates arguments and returns immutable ``OperatorGroup`` records. Native
callers use :cpp:func:`onnx_light::core::graph::ProposeOperatorGroups`.

Run shape inference before proposing groups when intermediate descriptors are
missing. Symbolic boundaries are supported without a byte cap: costs minimize
the number of unknown transfers before comparing known bytes. No dimension
values are invented, and ``transfer_bytes`` is ``None`` for a group with unknown
boundary sizes. Alternatively, pass ``value_sizes={"name": estimated_bytes}``
to supply explicit byte estimates.

The algorithm minimizes total boundary transfers over bounded topological
intervals; it does not claim optimality over arbitrary subgraphs. The default
search considers up to 128 non-Constant operators per function. Constants and
initializers remain explicit body inputs but contribute no transfer cost, except
for overridable initializers also declared as graph inputs. There is no byte cap
by default. An explicit ``max_boundary_bytes`` requires fully known combined
input/output costs within that cap, not a separate cap per tensor.
Limits and explicit byte estimates must fit in an unsigned 64-bit integer.
Static sizes use the shared native element-width helper, including packed
2-, 4- and 6-bit tensor types.
Static tensor sizes that exceed the native size range are treated as unknown,
not truncated. Accumulated transfer-cost overflow raises ``ValueError``.

``max_constant_inputs`` optionally caps distinct constant boundary inputs before
small-constant internalization. A shared weight counts once; overridable
initializers do not count as constants. The default ``None`` leaves this low-level
limit unset. The ``partition mincut`` CLI supplies
``count_constant_inputs(model) // 20`` to enforce its model-wide 5% weight budget,
rounded down.
``count_constant_inputs`` counts declared non-overridable constant values across
the root graph, local functions and nested graphs, without reading payloads.

To apply the proposals as local functions, explicitly select the
``PartitionGroup`` fusion pattern. It calls the same core algorithm using the
builder's inferred shapes. Materialized constants with strictly fewer than
32 elements are embedded as ``Constant`` nodes; other constants remain explicit
function arguments. A mincut group currently produces a function with one call.
Overridable initializers are never embedded, and external/deferred data is not
loaded for this specialization. An external ``Shape(x)`` is reconstructed inside
the function when ``x`` is already an input, removing the shape-result argument
without freezing dynamic dimensions. Repeated-layer extraction shares this
implementation and requires matching source-input positions and Shape attributes
across all occurrences:

.. code-block:: python

    from onnx_light.onnx_core.graph_builder import GraphBuilder
    from onnx_light.onnx_core.optimization import GraphGraph, PartitionGroupPattern

    builder = GraphBuilder(model)
    optimizer = GraphGraph(
        builder,
        patterns=[PartitionGroupPattern(max_group_nodes=128)],
    )
    optimizer.optimize()
    grouped_model = builder.to_model()

The pattern is not enabled by default. It runs analysis at the first node of
each optimizer scan, not at every candidate. Local-function bodies, previously
generated group calls and device-configured regions are not wrapped.

Generated functions use ``ai.rt::PartitionGroup_<mainop>_<index>``. The shared
core naming implementation also serves repeated-layer extraction. It selects
the first attention/MatMul anchor in source order, including a retained
local-function call, or the first operator when no anchor exists. Indices
start at zero and avoid existing function, value and registered-operator names.

``symbolic_byte_sizes(values)`` reuses native memory-analysis expressions to
describe logical tensor volumes without reading payloads. It preserves symbolic
dimensions and supported sub-byte packing, returning ``None`` for unavailable
shapes, unsupported types or expressions beyond the optional-analysis limit.

``describe_operator_group_calls(model, functions)`` describes actual
call interfaces in the root graph, local functions and nested graphs. It reports
logical and chargeable transfer volumes, excludes non-overridable constants from
transfers, and resolves captured values without mixing independent function
scopes. Each call has two aggregate totals, inputs and outputs, without per-value
details.
These shared reports serve both extraction methods; they describe interfaces
after internalization without changing either selector's objective.

.. automodule:: onnx_light.onnx_core.graph_partition
    :members:
