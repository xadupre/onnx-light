onnx_light.onnx_core.graph_partition
====================================

``propose_operator_groups`` identifies bodies suitable for local functions with
small input/output transfers, even when their intermediate tensors are large.
It returns proposals without changing the model or installing functions:

.. code-block:: python

    from onnx_light.onnx_core.graph_partition import propose_operator_groups

    groups = propose_operator_groups(model, max_boundary_bytes=1024)
    for group in groups:
        nodes = [model.graph.node[i] for i in group.node_indices]
        print(group.inputs, group.outputs, group.transfer_bytes)

Graph analysis, size accounting and grouping run in C++; the Python facade
validates arguments and returns immutable ``OperatorGroup`` records. Native
callers use :cpp:func:`onnx_light::core::graph::ProposeOperatorGroups`.

Run shape inference before proposing groups when intermediate descriptors are
missing. Alternatively, pass ``value_sizes={"name": estimated_bytes}`` for
dynamic boundaries. Unknown sizes are never treated as zero.

The algorithm minimizes total boundary transfers over bounded topological
intervals; it does not claim optimality over arbitrary subgraphs. The default
search considers up to 128 non-Constant operators per function. Constants and
initializers remain explicit body inputs but contribute no transfer cost, except
for overridable initializers also declared as graph inputs. The budget limits
the sum of all input and output bytes, not each tensor separately.
Limits and explicit byte estimates must fit in an unsigned 64-bit integer.
Static sizes use the shared native element-width helper, including packed
2-, 4- and 6-bit tensor types.
Static tensor sizes that exceed the native size range are treated as unknown,
not truncated. Accumulated transfer-cost overflow raises ``ValueError``.

.. automodule:: onnx_light.onnx_core.graph_partition
    :members:
