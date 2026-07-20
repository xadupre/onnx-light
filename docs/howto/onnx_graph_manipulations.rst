.. _l-howto-onnx-graph-manipulations:

:html_theme.sidebar_secondary.remove:

How to use ONNX graph manipulation helpers
==========================================

*onnx-light* ships a small set of graph-analysis utilities that inspect the
data-flow of a list of :class:`~onnx_light.onnx.NodeProto` objects without
running any operator or shape inference.  They are useful for memory-budget
analysis (which tensors must be kept alive at each point in a schedule?),
graph partitioning, and sub-graph extraction.

All three helpers are available both from Python and C++:

* :func:`~onnx_light.onnx.collect_external_inputs` /
  :cpp:func:`onnx_light::CollectExternalInputs` — Returns the names read by
  the nodes that are not produced by any node in the same list.
* :func:`~onnx_light.onnx.collect_remaining_inputs` /
  :cpp:func:`onnx_light::CollectRemainingInputs` — Returns, for every position
  in a topologically-ordered node list, the set of names that must remain in
  memory to produce the requested outputs.
* :cpp:func:`onnx_light::CollectNodeInputs` (C++ only) — Returns the full set
  of inputs (including subgraph captures) for a *single* node.

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

          import onnx_light.onnx as onnxl

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

          #include "onnx_core/graph/graph_manipulations.h"

          using namespace onnx_light;  // CollectExternalInputs, etc.


Collect external inputs
-----------------------

Given a list of nodes, :func:`~onnx_light.onnx.collect_external_inputs`
returns the names that are *consumed* by the nodes but are not *produced*
within the same list.  In a typical ONNX graph those correspond to the graph
inputs and initializers.

The function recurses into ``GRAPH`` / ``GRAPHS`` subgraph attributes so
names captured by ``If``, ``Loop``, or ``Scan`` subgraphs are included when
they are not produced inside the outer node list.

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

          import onnx_light.onnx as onnxl
          import onnx_light.onnx.helper as oh

          nodes = [
              oh.make_node("Mul", ["x", "y"], ["t"]),
              oh.make_node("Sub", ["t", "z"], ["out"]),
              oh.make_node("Add", ["out", "x"], ["final"]),
          ]

          # "t" and "out" are produced inside the list, so only x, y, z appear.
          ext = onnxl.collect_external_inputs(nodes)
          print(ext)  # ["x", "y", "z"]

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

          #include "onnx_core/graph/graph_manipulations.h"
          #include "onnx_proto/onnx_helper.h"

          std::vector<NodeProto> nodes = {
              MakeNode("Mul", {"x", "y"}, {"t"}),
              MakeNode("Sub", {"t", "z"}, {"out"}),
              MakeNode("Add", {"out", "x"}, {"final"}),
          };

          // "t" and "out" are produced inside the list, so only x, y, z appear.
          std::vector<std::string> ext = CollectExternalInputs(nodes);
          // ext == {"x", "y", "z"}

The result preserves first-seen order and contains no duplicates.  Empty
input names (optional inputs) are skipped.


Collect remaining inputs
------------------------

:func:`~onnx_light.onnx.collect_remaining_inputs` performs a backward
reachability analysis: starting from the requested ``outputs``, it determines
which nodes in ``nodes[i:]`` (the suffix from position *i* onward) actually
contribute to producing those outputs, then reports the external names those
relevant nodes need.  The result has exactly ``len(nodes)`` entries — one
per node — and is useful for computing the *live set* just before each node
runs.

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

          import onnx_light.onnx as onnxl
          import onnx_light.onnx.helper as oh

          nodes = [
              oh.make_node("Mul", ["x", "y"], ["t"]),
              oh.make_node("Sub", ["t", "z"], ["out"]),
              oh.make_node("Add", ["out", "x"], ["final"]),
          ]

          live = onnxl.collect_remaining_inputs(nodes, outputs=["final"])
          # live[0]: inputs needed from position 0 onward → ["x", "y", "z"]
          # live[1]: inputs needed from position 1 onward → ["t", "z", "x"]
          # live[2]: inputs needed from position 2 onward → ["out", "x"]
          for i, names in enumerate(live):
              print(f"before node {i}: {names}")

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

          #include "onnx_core/graph/graph_manipulations.h"
          #include "onnx_proto/onnx_helper.h"

          std::vector<NodeProto> nodes = {
              MakeNode("Mul", {"x", "y"}, {"t"}),
              MakeNode("Sub", {"t", "z"}, {"out"}),
              MakeNode("Add", {"out", "x"}, {"final"}),
          };

          std::vector<std::vector<std::string>> live =
              CollectRemainingInputs(nodes, {"final"});
          // live[0] == {"x", "y", "z"}
          // live[1] == {"t", "z", "x"}
          // live[2] == {"out", "x"}

Dead branches are pruned automatically.  If a node does not contribute to any
of the requested ``outputs`` it is excluded from all live sets:

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

          nodes_with_dead = [
              oh.make_node("Mul", ["x", "y"], ["t"]),
              oh.make_node("Sub", ["t", "z"], ["out"]),
              oh.make_node("Neg", ["w"], ["dead"]),   # does not feed "final"
              oh.make_node("Add", ["out", "x"], ["final"]),
          ]

          live = onnxl.collect_remaining_inputs(nodes_with_dead, outputs=["final"])
          # "w" never appears because the Neg node is pruned from all suffixes.
          print(live[0])  # ["x", "y", "z"]
          print(live[2])  # ["out", "x"]   (not ["w", "out", "x"])

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

          std::vector<NodeProto> nodes_with_dead = {
              MakeNode("Mul", {"x", "y"}, {"t"}),
              MakeNode("Sub", {"t", "z"}, {"out"}),
              MakeNode("Neg", {"w"}, {"dead"}),   // does not feed "final"
              MakeNode("Add", {"out", "x"}, {"final"}),
          };

          auto live = CollectRemainingInputs(nodes_with_dead, {"final"});
          // live[0] == {"x", "y", "z"}   ("w" absent because Neg is pruned)
          // live[2] == {"out", "x"}       (not {"w", "out", "x"})


Collect inputs for a single node (C++)
---------------------------------------

:cpp:func:`onnx_light::CollectNodeInputs` is a convenience wrapper that
returns the combined input set for a *single* node, including any names
captured from the outer scope by ``GRAPH`` / ``GRAPHS`` subgraph attributes:

.. code-block:: cpp

    #include "onnx_core/graph/graph_manipulations.h"
    #include "onnx_proto/onnx_helper.h"

    NodeProto node = MakeNode("Add", {"a", "b"}, {"c"});
    std::vector<std::string> inputs = CollectNodeInputs(node);
    // inputs == {"a", "b"}

The Python binding :func:`~onnx_light.onnx.collect_external_inputs` on a
single-element list is equivalent:

.. code-block:: python

    import onnx_light.onnx as onnxl
    import onnx_light.onnx.helper as oh

    node = oh.make_node("Add", ["a", "b"], ["c"])
    inputs = onnxl.collect_external_inputs([node])
    # inputs == ["a", "b"]


See also
--------

* :doc:`/api/cpp/onnx_core/graph_manipulations` — C++ API reference
  for :cpp:func:`onnx_light::CollectExternalInputs`,
  :cpp:func:`onnx_light::CollectRemainingInputs` and
  :cpp:func:`onnx_light::CollectNodeInputs`.
* :ref:`l-how-to` — other onnx-light how-to recipes.
