Incremental graph layout
========================

Layout is native C++ presentation state, not a viewer or a numerical graph
transformation. It never sorts nodes, renames tensors, changes execution order,
or modifies operator attributes merely to compute positions.

See :ref:`l-example-plot-graph-layout` for a runnable gallery example that draws
native node rectangles and port routes, applies local edits, and reloads
persisted positions.

Loading and editing
-------------------

.. code-block:: python

    import onnx_light.onnx as onnx
    import onnx_light.onnx.helper as oh

    model = onnx.load("model.onnx", layout=True)
    layout = onnx.GraphLayout(model.graph)
    layout.load()  # Reconstructs indexes and reuses persisted positions.
    node_id = layout.append(oh.make_node("Relu", ["existing_value"], ["new_value"]))
    layout.set_size(node_id, 240, 96)
    layout.set_position(node_id, -100, -80)
    layout.persist()
    onnx.save(model, "positioned.onnx")

``layout=False`` is the default: loading neither computes positions nor modifies
layout or unrelated metadata. With ``layout=True``, binary loading installs
``ParseOptions.graph_callback`` and ``ParseOptions.function_callback``.
The callbacks run as each graph or function finishes parsing,
after its node callbacks and all metadata fields, with children before parents.
This ordering matters because ONNX may serialize graph metadata after its nodes.
Local function bodies in ``model.functions`` receive their own layout, including
their nested graph attributes. There is no separate post-model layout pass.
Textproto loading uses an iterative child-first traversal, loading and persisting
each scope nonrecursively before returning the parsed model. Diagnostics are
emitted as each scope is processed, without retaining a tree of live layouts.

Low-level Python callers can call ``onnx.enable_graph_layout(options)`` on their
``ParseOptions``. This chains existing graph and function callbacks rather than
replacing them. C++ callers can use the same hooks directly:

.. code-block:: cpp

    using namespace onnx_light;
    ParseOptions options;
    options.graph_callback = [](GraphProto &graph) {
      core::graph::GraphLayout layout(graph);
      layout.Load(false);
      layout.Persist(false);
    };
    options.function_callback = [](FunctionProto &function) {
      core::graph::GraphLayout layout(function, false);
      layout.Persist(false);
    };
    ModelProto model;
    model.ParseFromFile("model.onnx", options);

``GraphLayout`` binds directly to a live ``GraphProto`` or ``FunctionProto``,
without converting or copying function bodies. For example,
``onnx.GraphLayout(model.functions[0])`` edits that function's actual nodes and
metadata. Function formal inputs are local source endpoints, not outer captures;
graph and function scopes have independent ID spaces.
Call ``load()`` once, then perform
edits through ``append``, ``insert``, ``erase``, ``replace``, ``rewire_input``,
``set_size`` and ``set_position``. Replacement keeps the original layout ID;
insertion and deletion affect the requested semantic node-list positions.
``append`` and ``insert`` discard incoming node IDs, positions, sizing flags,
ranks and signatures, assigning fresh layout state in the destination scope.
They preserve unrelated node metadata and layout metadata inside child graphs.
``replace`` also discards incoming node layout metadata while retaining the
destination node's ID and live positioning/sizing state.
Direct modifications to the bound proto bypass the layout indexes; call
``load()`` after such modifications. Reloading an attached child updates its
owning container and affected consumers if its bounds or captures change.
Call ``persist()`` before
saving subsequent edits; after child edits, call it on the root layout so the
updated container state is saved as well. Saving itself does not optimize or
normalize layout.

``order`` contains stable IDs in semantic node-list order, ``get(id)`` returns
dimensions and coordinates, and ``edges(consumer)`` returns ordered port routes.
Scope-local IDs are not node names or list indices. Rewrites preserving node
metadata retain identity; rewrites constructing new nodes should use
``replace`` to explicitly transfer identity. A rewrite dropping layout metadata
cannot preserve an unknown placement.

Coordinates and ONNX presentation
--------------------------------

Coordinates are left-to-right, with increasing y downward, in arbitrary visual
units. Positions refer to the top-left corner of a node rectangle. Negative
coordinates are valid and are never globally translated to become nonnegative.
Dimensions account for ordered input/output ports. Repeated inputs have distinct
target ports even when their source value is the same.

Initializers are folded into consumer input ports instead of creating one huge
initializer column. ``Constant`` nodes have a compact folded presentation.
Folded nodes are virtual content intended for rendering inline at consumer input
ports, not independent rectangles. Their IDs and semantic order remain public,
but their coordinates may overlap; their boxes do not occupy space or contribute
to layout bounds. Renderers should skip folded boxes and render folded edges as
port content rather than drawing their ordinary routes.
Graph inputs and captures are source endpoints, not invented operator nodes.
Graph outputs retain their tensor names and refer to existing value endpoints.
Layout does not add proxy nodes or tensors to the model.

Every ``GRAPH`` and ``GRAPHS`` attribute is traversed, independently of the
operator name. Each child has its own coordinate system, metadata and ID space;
equal IDs or tensor names in different graphs do not identify the same object.
Child coordinates are relative to the interior of their owning container, not
absolute parent coordinates. Renderers choose the container padding/origin.
Outer-scope captures constrain the owning container's dependencies without
creating a local producer. ``children(container_id)`` exposes child layouts for
incremental edits; their container effects propagate through the owning node.
Python child handles keep their root graph alive. After ``load``, ``replace``
or ``erase`` on any handle, reacquire child handles with ``children()``; stale
handles raise an exception rather than accessing freed native state. Geometry
and edge queries return snapshots that remain safe after subsequent edits.
``persist()`` only writes metadata and never rebuilds child layouts, so existing
child handles remain valid for both recursive and nonrecursive persistence.

Incremental guarantees and costs
-------------------------------

The engine maintains producer/consumer indexes, dependency layers, geometry and
occupied layer intervals. A work queue propagates changed constraints through
the affected dependency region. Vertical placement uses port alignment and local
collision resolution. Routes are refreshed for affected endpoints only.
The spatial interval index uses AVL balancing, so persisted IDs and coordinates
cannot force linear tree depth during insertion, deletion, queries or destruction.
Unrelated components retain their coordinates; ordinary edits do not globally
compact, renumber, translate, or optimize the layout.

Initialization and reconstruction of indexes necessarily read the graph.
Ordinary mutations do not call ``load()`` or whole-graph shape inference.
``stats`` reports the nodes, edges and graph scopes processed by the last edit,
allowing tests and applications to observe locality. A producer near the start
of a long chain, or growth of a crowded container, can affect a large region.
Collision neighborhoods can also be large. Updates therefore are not guaranteed
constant-time; insertion/deletion additionally incurs the underlying node-list
storage cost. Stability takes precedence over globally minimal area.

Persistence and validation
--------------------------

Layout uses ONNX ``metadata_props`` rather than changing tensor names or node
names. Keys are namespaced with ``onnx_light.layout.`` and the graph-level
``onnx_light.layout.version`` identifies schema version ``1``. Node metadata
contains ``onnx_light.layout.id`` and ``onnx_light.layout.position``. Positions
are locale-independent decimal x, y, width and height, preserving negative
coordinates and double precision. Graph-level state supports resuming local
placement; adjacency and routes are reconstructed from actual ONNX values.
Unrelated metadata keys and values are preserved.

The node keys ``onnx_light.layout.automatic`` and
``onnx_light.layout.fixed_size`` use ``"0"``/``"1"`` values to distinguish
automatic placement from manual positions, and inferred dimensions from
explicit dimensions. An absent ``fixed_size`` key selects inferred dimensions.
``onnx_light.layout.scope`` stores a bounded scope-local
geometry/capture summary so a parent finishing its parse can use child state
without rebuilding every child layout. Capture names use quoted, escaped
strings, not whitespace-delimited tensor names.
Persistence omits summaries exceeding the size or capture-count limits and
reports the limit through diagnostics. An absent summary triggers direct child
loading without an invalid-summary diagnostic.
The graph/function key ``onnx_light.layout.next_id`` persists the monotonic ID allocator;
node keys ``onnx_light.layout.rank`` and ``onnx_light.layout.signature`` persist
the dependency rank and a semantic signature used to reject stale geometry.
IDs are positive decimal integers, limited to ``2**53 - 1``. The next-ID key
also accepts ``2**53`` as an exhausted-allocator sentinel, preserved across
reloads even after the final node is erased. Further insertion throws
``std::overflow_error`` without changing the node list. Ranks are
decimal integers in ``[0, node_count - 1]`` and are validated against the actual
dependency constraints. Signatures are change detectors, not cryptographic authentication.
Child scope summaries larger than 1 MiB are ignored when computing signatures,
matching the summary parser's size limit; their raw bytes are not hashed.
If duplicate scope summaries exist, nonrecursive loading skips malformed entries
until a valid summary is found, diagnosing rejected entries. Node signatures
track all bounded summaries so changes to a later usable entry invalidate stale
container geometry.

Missing placement state is computed locally, reserving valid saved rectangles
so missing or stale geometry does not displace unrelated saved nodes.
Invalid or duplicate IDs are
reassigned, stale geometry is repaired, and malformed or unsupported-version
geometry is ignored rather than trusted. ``diagnostics`` reports recoveries;
loading with layout enabled emits ``RuntimeWarning`` diagnostics. Metadata is
bounded before parsing, numeric values must be finite, coordinate magnitudes
are limited to one billion visual units, and dimensions are positive and
bounded. Nested layout has a depth limit. Layout validation is not a replacement
for the ONNX model checker.
