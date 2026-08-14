"""Convert an ONNX model or graph to a standalone `SVG <https://www.w3.org/Graphics/SVG/>`_
image.

The resulting string is a self-contained ``<svg>`` document that can be
written to a ``.svg`` file or embedded directly in an HTML page::

    from onnx_light.tools import to_svg

    with open("model.svg", "w", encoding="utf-8") as f:
        f.write(to_svg(model))

Unlike :func:`onnx_light.tools.to_mermaid`, which produces source that
still needs a Mermaid renderer, :func:`to_svg` performs a simple layered
layout itself and emits ready-to-display SVG.  The converter is
implemented in pure Python and only depends on the attributes of the
standard ONNX message types (``ModelProto``, ``GraphProto``,
``NodeProto``, ``ValueInfoProto``, ``TensorProto`` and
``TensorShapeProto``).  It therefore works both with messages built by
:mod:`onnx_light` and with messages built by the upstream :mod:`onnx`
package.
"""

from __future__ import annotations

import math
from typing import Any

from ._proto_utils import (
    NODE_TAG_METADATA_KEY,
    VALUE_TAG_COLORS,
    _graph_value_tags,
    _dtype_name,
    _extract_graph,
    _format_inplace_reuse,
    _format_release_after,
    _format_shape,
    _iter,
    _looks_like_graph,
    _node_metadata_value,
    _short_display_name,
    _s,
)

# ---------------------------------------------------------------------------
# Geometry constants
# ---------------------------------------------------------------------------

_FONT_SIZE = 12
_CHAR_WIDTH = 7.0  # Approximate width of a monospace-ish character in px.
_LINE_HEIGHT = 16.0
_BOX_PAD_X = 12.0
_BOX_PAD_Y = 8.0
_LAYER_GAP = 60.0  # Gap between successive layers.
_SIBLING_GAP = 24.0  # Gap between boxes within the same layer.
_MARGIN = 20.0  # Outer margin around the whole drawing.
# Number of barycenter sweeps used to reduce edge crossings; a handful of
# passes is enough to converge for the small graphs rendered here.
_CROSSING_SWEEPS = 4
_EDGE_LABEL_STAGGER = 8.0  # Per-label orthogonal offset to reduce text collisions.
# Edge labels are rendered small and unobtrusive (no highlight halo) so they do
# not dominate the diagram.
_EDGE_FONT_SIZE = _FONT_SIZE - 3
_EDGE_LABEL_COLOR = "#888888"

# Styling per kind of box: ``fill``, ``stroke`` and ``dashed`` flag.
_STYLES = {
    "input": {"fill": "#cde4ff", "stroke": "#3a6ea5", "dashed": False, "rounded": True},
    "output": {"fill": "#ffe1b3", "stroke": "#a35a00", "dashed": False, "rounded": True},
    "initializer": {"fill": "#eeeeee", "stroke": "#888888", "dashed": True, "rounded": False},
    "op": {"fill": "#d4ecd4", "stroke": "#3a8c3a", "dashed": False, "rounded": False},
}


# ---------------------------------------------------------------------------
# XML escaping
# ---------------------------------------------------------------------------

_XML_ESCAPE = {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&apos;"}


def _escape_xml(text: str) -> str:
    """Escapes characters that are special in XML text and attributes."""
    return "".join(_XML_ESCAPE.get(ch, ch) for ch in text)


def _edge_label(name: str, shape: str) -> str:
    """Returns the SVG label for an edge.

    Parameters:
        name: The tensor name carried by the edge.
        shape: The optional tensor shape annotation for the edge.

    Returns:
        A compact label containing the shortened tensor name, optionally
        followed by the shape annotation.
    """
    display_name = _short_display_name(name)
    if display_name and shape:
        return f"{display_name} · {shape}"
    return display_name or shape


# ---------------------------------------------------------------------------
# Internal box representation
# ---------------------------------------------------------------------------


class _Box:
    """A single rectangle in the rendered diagram."""

    def __init__(self, box_id: int, kind: str, lines: list[str], tag: str = "") -> None:
        self.id = box_id
        self.kind = kind
        self.tag = tag
        self.lines = [line for line in lines if line] or [""]
        text_width = max((len(line) for line in self.lines), default=0) * _CHAR_WIDTH
        self.width = text_width + 2 * _BOX_PAD_X
        self.height = len(self.lines) * _LINE_HEIGHT + 2 * _BOX_PAD_Y
        self.layer = 0
        # Position within the layer, filled in by the crossing-reduction pass.
        self.order = 0
        # Top-left corner, filled in during layout.
        self.x = 0.0
        self.y = 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def to_svg(
    model_or_graph: Any,
    *,
    direction: str = "TB",
    layout: str = "layered",
    include_initializers: bool = True,
    include_shapes: bool = True,
    include_attributes: bool = False,
    include_inplace: bool = False,
    include_release: bool = False,
) -> str:
    """Renders an ONNX ``ModelProto`` or ``GraphProto`` as an SVG image.

    Args:
        model_or_graph: A ``ModelProto`` or ``GraphProto`` instance.  Both
            :mod:`onnx_light` and :mod:`onnx` messages are accepted.
        direction: Layout direction; ``"TB"`` (or ``"TD"``) lays the graph
            out top-to-bottom and ``"LR"`` left-to-right.  Defaults to
            ``"TB"``.
        layout: Positioning strategy for the boxes.  ``"layered"`` (the
            default) uses the built-in layered layout with barycenter
            crossing reduction.  ``"umap"`` instead derives the box
            positions from a two-dimensional `UMAP
            <https://umap-learn.readthedocs.io/>`_ embedding of the graph's
            connectivity, which requires the optional ``umap-learn`` package
            to be installed.
        include_initializers: When :data:`True`, initializers are rendered
            as separate (dashed) boxes connected to their consumers.  When
            :data:`False`, initializer tensors are not shown.
        include_shapes: When :data:`True`, tensor type/shape information
            available in graph inputs, outputs, ``value_info`` and
            initializers is appended to the corresponding box labels.
        include_attributes: When :data:`True`, node attribute names are
            listed inside the operator label.
        include_inplace: When :data:`True`, the in-place reuse opportunities
            recorded in each node's ``metadata_props`` (under the
            ``onnx_light.inplace_reuse`` key) are appended to the operator
            label, for example ``inplace: out0=in1(equal)``.
        include_release: When :data:`True`, the post-execution release hints
            recorded in each node's ``metadata_props`` (under the
            ``onnx_light.release_after`` key, and optional last-use hints
            under ``onnx_light.not_used_after``) are appended to the operator
            label, for example ``release: A, B; not used after: X``.

    Returns:
        A self-contained SVG document as a single ``str``.

    Raises:
        TypeError: If ``model_or_graph`` is neither a ``ModelProto`` nor
            a ``GraphProto``.
        ValueError: If ``direction`` or ``layout`` is not supported.

    The example below builds a small ``Abs`` chain, runs shape inference and
    records the in-place reuse opportunities into the graph metadata with
    :func:`onnx_light.onnx_core.shape_inference.write_inplace_reuse_to_metadata`,
    then renders the annotated diagram with ``include_inplace=True``:

    .. runpython::
        :rst:

        from onnx_light.onnx_lib import TensorProto
        from onnx_light.onnx.helper import (
            make_graph,
            make_model,
            make_node,
            make_opsetid,
            make_tensor_value_info,
        )
        from onnx_light.onnx_core import shape_inference
        from onnx_light.tools import to_svg

        X = make_tensor_value_info("X", TensorProto.FLOAT, [3, 4])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [3, 4])
        graph = make_graph(
            [
                make_node("Abs", ["X"], ["A"]),
                make_node("Abs", ["A"], ["B"]),
                make_node("Abs", ["B"], ["Y"]),
            ],
            "example",
            [X],
            [Y],
        )
        model = make_model(graph, opset_imports=[make_opsetid("", 18)])
        model.ir_version = 8

        ctx = shape_inference.ShapesContext()
        shape_inference.compute_shape_model(ctx, model)
        shape_inference.write_inplace_reuse_to_metadata(ctx, model.graph)

        print(".. raw:: html")
        print()
        for line in to_svg(model, include_inplace=True).split("\\n"):
            print("    " + line)
    """
    valid_directions = {"TB", "TD", "LR"}
    if direction not in valid_directions:
        raise ValueError(
            f"Unsupported SVG direction {direction!r}; "
            f"expected one of {sorted(valid_directions)}."
        )

    graph = _extract_graph(model_or_graph)
    return to_svg_graph(
        graph,
        direction=direction,
        layout=layout,
        include_initializers=include_initializers,
        include_shapes=include_shapes,
        include_attributes=include_attributes,
        include_inplace=include_inplace,
        include_release=include_release,
    )


def to_svg_graph(
    graph: Any,
    *,
    direction: str = "TB",
    layout: str = "layered",
    include_initializers: bool = True,
    include_shapes: bool = True,
    include_attributes: bool = False,
    include_inplace: bool = False,
    include_release: bool = False,
) -> str:
    """Renders a ``GraphProto`` as an SVG image.

    See :func:`to_svg` for the meaning of every parameter.
    """
    if not _looks_like_graph(graph):
        raise TypeError(
            "to_svg_graph expected a GraphProto-like object "
            f"with 'node', 'input' and 'output' fields, got {type(graph).__name__}."
        )

    valid_layouts = {"layered", "umap"}
    if layout not in valid_layouts:
        raise ValueError(
            f"Unsupported SVG layout {layout!r}; expected one of {sorted(valid_layouts)}."
        )

    horizontal = direction == "LR"

    boxes: list[_Box] = []
    # Maps an ONNX tensor name to the box that produces it.
    producer: dict[str, int] = {}
    # Edges as ordered (source box id, target box id, label) tuples.
    edges: list[tuple[int, int, str]] = []

    value_tags = _graph_value_tags(graph)

    def new_box(kind: str, lines: list[str], tag: str = "") -> _Box:
        box = _Box(len(boxes), kind, lines, tag=tag)
        boxes.append(box)
        return box

    # Shape lookup from inputs, outputs and value_info.
    shape_lookup: dict[str, str] = {}
    if include_shapes:
        for collection in ("input", "output", "value_info"):
            for value_info in _iter(getattr(graph, collection, ())):
                shape_lookup[_s(value_info.name)] = _format_shape(
                    getattr(value_info, "type", None)
                )

    initializer_names: set[str] = set()
    initializer_shapes: dict[str, str] = {}
    for init in _iter(getattr(graph, "initializer", ())):
        name = _s(init.name)
        initializer_names.add(name)
        if include_shapes:
            dims = ",".join(str(int(d)) for d in getattr(init, "dims", ()))
            dtype = _dtype_name(int(getattr(init, "data_type", 0)))
            initializer_shapes[name] = f"{dtype}[{dims}]" if dims else dtype

    # Input boxes.
    input_names = [_s(v.name) for v in _iter(getattr(graph, "input", ()))]
    input_name_set = set(input_names)
    for name in input_names:
        if not include_initializers and name in initializer_names:
            continue
        lines = [_short_display_name(name) or "(unnamed)"]
        shape = shape_lookup.get(name, "")
        if shape:
            lines.append(shape)
        box = new_box("input", lines, value_tags.get(name, ""))
        producer[name] = box.id

    # Initializer boxes (skip those already shown as inputs).
    if include_initializers:
        for name in sorted(initializer_names):
            if name in input_name_set:
                continue
            lines = [_short_display_name(name) or "(unnamed)"]
            shape = initializer_shapes.get(name, "")
            if shape:
                lines.append(shape)
            box = new_box("initializer", lines, value_tags.get(name, ""))
            producer[name] = box.id

    # Operator boxes (recorded first so producers exist before edges).
    op_boxes: list[tuple[int, Any]] = []
    for node in _iter(getattr(graph, "node", ())):
        op_type = _s(getattr(node, "op_type", "")) or "Op"
        lines = [op_type]
        if include_attributes:
            attr_names = sorted(_s(a.name) for a in _iter(getattr(node, "attribute", ())))
            if attr_names:
                lines.append(", ".join(attr_names))
        if include_inplace:
            inplace_label = _format_inplace_reuse(node)
            if inplace_label:
                lines.append(inplace_label)
        if include_release:
            release_label = _format_release_after(node)
            if release_label:
                lines.append(release_label)
        node_tag = _s(_node_metadata_value(node, NODE_TAG_METADATA_KEY)).lower()
        box = new_box("op", lines, node_tag)
        op_boxes.append((box.id, node))
        for out in _iter(getattr(node, "output", ())):
            out_name = _s(out)
            if out_name:
                producer[out_name] = box.id

    # Edges for operator inputs.
    for box_id, node in op_boxes:
        for inp in _iter(getattr(node, "input", ())):
            inp_name = _s(inp)
            if not inp_name:
                continue
            if not include_initializers and inp_name in initializer_names:
                continue
            src = producer.get(inp_name)
            if src is None:
                continue
            shape = shape_lookup.get(inp_name) or initializer_shapes.get(inp_name, "")
            label = _edge_label(inp_name, shape if include_shapes else "")
            edges.append((src, box_id, label))

    # Output boxes and their incoming edges.
    for value_info in _iter(getattr(graph, "output", ())):
        name = _s(value_info.name)
        if not name:
            continue
        shape = shape_lookup.get(name, "")
        lines = [shape or "output"]
        box = new_box("output", lines, value_tags.get(name, ""))
        src = producer.get(name)
        if src is not None:
            edges.append((src, box.id, _edge_label(name, shape if include_shapes else "")))

    if layout == "umap":
        width, height = _layout_umap(boxes, edges, horizontal)
    else:
        _assign_layers(boxes, edges)
        _minimize_crossings(boxes, edges)
        width, height = _layout(boxes, horizontal)

    return _render_svg(boxes, edges, width, height, horizontal)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def _assign_layers(boxes: list[_Box], edges: list[tuple[int, int, str]]) -> None:
    """Assigns a layer index to every box.

    The layers are first computed with longest-path layering and then pure
    source boxes (graph inputs and initializers, which have no incoming
    edges) are pulled down to sit just above their earliest consumer.  This
    keeps inputs that feed deep operators from being stranded on the first
    row, which shortens the corresponding edges and yields a more compact
    drawing.
    """
    # Relax edges until the layering is stable; an ONNX graph is a DAG so
    # this converges in at most ``len(boxes)`` passes.
    for _ in range(len(boxes)):
        changed = False
        for src, dst, _label in edges:
            if boxes[dst].layer < boxes[src].layer + 1:
                boxes[dst].layer = boxes[src].layer + 1
                changed = True
        if not changed:
            break

    if not edges:
        return

    successors: dict[int, list[int]] = {box.id: [] for box in boxes}
    predecessors: dict[int, list[int]] = {box.id: [] for box in boxes}
    for src, dst, _label in edges:
        successors[src].append(dst)
        predecessors[dst].append(src)

    # Pull pure sources down to just above their earliest consumer.  A source
    # has no predecessors, so lowering it only shortens its outgoing edges and
    # can never lengthen another edge.
    for box in boxes:
        if not predecessors[box.id] and successors[box.id]:
            box.layer = min(boxes[succ].layer for succ in successors[box.id]) - 1


def _minimize_crossings(boxes: list[_Box], edges: list[tuple[int, int, str]]) -> None:
    """Orders boxes within each layer to reduce the number of edge crossings.

    Uses the iterated barycenter heuristic: every box is repeatedly placed at
    the average position of its neighbours in the adjacent layers, alternating
    downward and upward sweeps.  Boxes without neighbours keep their current
    position so the layout stays stable.

    Long-range edges (skip connections) that span more than one layer are
    handled by inserting dummy nodes at every intermediate layer before the
    sweeps begin.  Each original edge thereby becomes a chain of unit-length
    edges, so the barycenter of every real node is always computed relative to
    the single adjacent layer — no positions from non-comparable layers are
    ever mixed.  The dummy nodes are discarded after the sweeps; they are
    never written back to the caller's ``boxes`` list.
    """
    if not edges:
        return

    layers: dict[int, list[_Box]] = {}
    for box in boxes:
        layers.setdefault(box.layer, []).append(box)
    sorted_layers = sorted(layers)
    layer_rank: dict[int, int] = {layer: i for i, layer in enumerate(sorted_layers)}
    box_by_id: dict[int, _Box] = {box.id: box for box in boxes}

    # Insert dummy nodes at intermediate layers for each long-range edge so
    # that every edge in the augmented graph spans exactly one layer.
    next_id = max(box_by_id, default=-1) + 1
    real_ids: set[int] = set(box_by_id)
    aug_edges: list[tuple[int, int, str]] = []
    for src_id, dst_id, label in edges:
        src_rank = layer_rank[box_by_id[src_id].layer]
        dst_rank = layer_rank[box_by_id[dst_id].layer]
        if dst_rank - src_rank <= 1:
            aug_edges.append((src_id, dst_id, label))
            continue
        # Break the long-range edge into a chain through one dummy per intermediate layer.
        prev_id = src_id
        for ri in range(src_rank + 1, dst_rank):
            inter_layer = sorted_layers[ri]
            # "dummy" is an internal-only kind; these boxes never reach the renderer.
            dummy = _Box(next_id, "dummy", [])
            dummy.layer = inter_layer
            box_by_id[next_id] = dummy
            layers[inter_layer].append(dummy)
            aug_edges.append((prev_id, next_id, ""))
            next_id += 1
            prev_id = dummy.id
        aug_edges.append((prev_id, dst_id, ""))

    # Build adjacency maps from the augmented edge set.
    successors: dict[int, list[int]] = {bid: [] for bid in box_by_id}
    predecessors: dict[int, list[int]] = {bid: [] for bid in box_by_id}
    for src, dst, _ in aug_edges:
        successors[src].append(dst)
        predecessors[dst].append(src)

    # Position of every box (real and dummy) within its layer.
    position: dict[int, int] = {}
    for layer_boxes in layers.values():
        for index, box in enumerate(layer_boxes):
            position[box.id] = index

    def reorder(layer_index: int, neighbours: dict[int, list[int]]) -> None:
        layer_boxes = layers[layer_index]

        def barycenter(box: _Box) -> float:
            ids = neighbours[box.id]
            if not ids:
                return float(position[box.id])
            return sum(position[i] for i in ids) / len(ids)

        layer_boxes.sort(key=barycenter)
        for index, box in enumerate(layer_boxes):
            position[box.id] = index

    for _ in range(_CROSSING_SWEEPS):
        for layer_index in sorted_layers[1:]:
            reorder(layer_index, predecessors)
        for layer_index in reversed(sorted_layers[:-1]):
            reorder(layer_index, successors)

    # Write the final within-layer order back to real boxes only.
    for layer_boxes in layers.values():
        for index, box in enumerate(layer_boxes):
            if box.id in real_ids:
                box.order = index


def _layout(boxes: list[_Box], horizontal: bool) -> tuple[float, float]:
    """Places every box and returns the overall ``(width, height)``."""
    if not boxes:
        return (2 * _MARGIN, 2 * _MARGIN)

    layers: dict[int, list[_Box]] = {}
    for box in boxes:
        layers.setdefault(box.layer, []).append(box)

    # Honor the order computed by the crossing-reduction pass.
    for layer_boxes in layers.values():
        layer_boxes.sort(key=lambda box: (box.order, box.id))

    # ``cross`` is the axis along which siblings spread, ``depth`` the axis
    # along which layers stack.
    cross_extent = 0.0
    depth_cursor = _MARGIN
    for layer_index in sorted(layers):
        layer_boxes = layers[layer_index]
        depth_size = max((box.width if horizontal else box.height) for box in layer_boxes)
        cross_cursor = _MARGIN
        for box in layer_boxes:
            if horizontal:
                box.x = depth_cursor
                box.y = cross_cursor
                cross_cursor += box.height + _SIBLING_GAP
            else:
                box.x = cross_cursor
                box.y = depth_cursor
                cross_cursor += box.width + _SIBLING_GAP
        cross_extent = max(cross_extent, cross_cursor - _SIBLING_GAP)
        depth_cursor += depth_size + _LAYER_GAP

    depth_extent = depth_cursor - _LAYER_GAP + _MARGIN

    # Centre each layer along the cross axis for a tidier picture.
    for layer_boxes in layers.values():
        if horizontal:
            used = sum(box.height for box in layer_boxes) + _SIBLING_GAP * (len(layer_boxes) - 1)
            offset = (cross_extent - _MARGIN - used) / 2.0
            for box in layer_boxes:
                box.y += offset
        else:
            used = sum(box.width for box in layer_boxes) + _SIBLING_GAP * (len(layer_boxes) - 1)
            offset = (cross_extent - _MARGIN - used) / 2.0
            for box in layer_boxes:
                box.x += offset

    if horizontal:
        return (depth_extent, cross_extent + _MARGIN)
    return (cross_extent + _MARGIN, depth_extent)


# Minimum number of boxes required for a meaningful UMAP embedding; below this
# the layered layout is used because UMAP needs a handful of neighbours.
_UMAP_MIN_BOXES = 4
# Target neighbourhood size handed to UMAP; capped to the number of other boxes.
_UMAP_DEFAULT_NEIGHBORS = 15
# Fixed seed so the UMAP embedding (and therefore the drawing) is reproducible.
_UMAP_RANDOM_STATE = 42


def _graph_distances(box_count: int, edges: list[tuple[int, int, str]]) -> list[list[float]]:
    """Computes an all-pairs shortest-path distance matrix over the boxes.

    Edges are treated as undirected and unit length.  Pairs of boxes that are
    not connected receive a distance one greater than the graph diameter so
    that disconnected components are pushed apart by the embedding.

    Returns:
        A symmetric ``box_count`` by ``box_count`` matrix of distances.
    """
    from collections import deque

    adjacency: list[set[int]] = [set() for _ in range(box_count)]
    for src, dst, _label in edges:
        adjacency[src].add(dst)
        adjacency[dst].add(src)

    distances = [[0.0] * box_count for _ in range(box_count)]
    diameter = 0
    for start in range(box_count):
        seen = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbour in adjacency[node]:
                if neighbour not in seen:
                    seen[neighbour] = seen[node] + 1
                    queue.append(neighbour)
        for target, hops in seen.items():
            distances[start][target] = float(hops)
            diameter = max(diameter, hops)

    far = float(diameter + 1)
    for i in range(box_count):
        for j in range(box_count):
            if i != j and distances[i][j] == 0.0:
                distances[i][j] = far
    return distances


def _layout_umap(
    boxes: list[_Box], edges: list[tuple[int, int, str]], horizontal: bool
) -> tuple[float, float]:
    """Places every box using a two-dimensional UMAP embedding of the graph.

    The graph connectivity is turned into an all-pairs shortest-path distance
    matrix which is embedded into two dimensions with UMAP (``metric``
    ``"precomputed"``).  The embedding is scaled to pixel space and, so the
    dominant axis matches ``direction``, optionally rotated by ninety degrees.
    Graphs with fewer than :data:`_UMAP_MIN_BOXES` boxes fall back to the
    layered layout because UMAP needs a handful of neighbours to converge.

    Returns:
        The overall ``(width, height)`` of the drawing in pixels.

    Raises:
        ImportError: If the optional ``umap-learn`` package is not installed.
    """
    if len(boxes) < _UMAP_MIN_BOXES:
        _assign_layers(boxes, edges)
        _minimize_crossings(boxes, edges)
        return _layout(boxes, horizontal)

    try:
        import numpy
        import umap
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "The 'umap' layout requires the optional 'umap-learn' package; "
            "install it with 'pip install umap-learn'."
        ) from exc

    distances = numpy.asarray(_graph_distances(len(boxes), edges), dtype=numpy.float64)
    reducer = umap.UMAP(
        n_components=2,
        metric="precomputed",
        n_neighbors=min(_UMAP_DEFAULT_NEIGHBORS, len(boxes) - 1),
        random_state=_UMAP_RANDOM_STATE,
    )
    embedding = numpy.asarray(reducer.fit_transform(distances), dtype=numpy.float64)

    # Orient the embedding so its most spread-out axis runs along the layer
    # (depth) direction requested by the caller.
    spread = embedding.max(axis=0) - embedding.min(axis=0)
    depth_axis = 1 if horizontal else 0
    if spread[depth_axis] < spread[1 - depth_axis]:
        embedding = embedding[:, ::-1]

    lo = embedding.min(axis=0)
    hi = embedding.max(axis=0)
    span = numpy.where(hi - lo > 0.0, hi - lo, 1.0)
    normalized = (embedding - lo) / span

    # Size the canvas so the boxes have room to spread without heavy overlap.
    columns = math.ceil(math.sqrt(len(boxes)))
    max_width = max(box.width for box in boxes)
    max_height = max(box.height for box in boxes)
    canvas_width = columns * (max_width + _SIBLING_GAP)
    canvas_height = columns * (max_height + _SIBLING_GAP)

    for index, box in enumerate(boxes):
        center_x = _MARGIN + max_width / 2.0 + normalized[index, 0] * canvas_width
        center_y = _MARGIN + max_height / 2.0 + normalized[index, 1] * canvas_height
        box.x = center_x - box.width / 2.0
        box.y = center_y - box.height / 2.0

    width = 2 * _MARGIN + canvas_width + max_width
    height = 2 * _MARGIN + canvas_height + max_height
    return (width, height)


# ---------------------------------------------------------------------------
# SVG rendering
# ---------------------------------------------------------------------------


def _render_svg(
    boxes: list[_Box],
    edges: list[tuple[int, int, str]],
    width: float,
    height: float,
    horizontal: bool,
) -> str:
    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{_round(width)}" height="{_round(height)}" '
        f'viewBox="0 0 {_round(width)} {_round(height)}" '
        f'font-family="sans-serif" font-size="{_FONT_SIZE}">'
    )
    parts.append(
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" '
        'refX="8" refY="3" orient="auto" markerUnits="strokeWidth">'
        '<path d="M0,0 L8,3 L0,6 Z" fill="#555555"/></marker></defs>'
    )

    # Draw edges first so boxes sit on top of the lines.
    labeled_edge_index = 0
    for src, dst, label in edges:
        label_shift = 0.0
        if label:
            if labeled_edge_index > 0:
                # Staggers as 0, +d, -d, +2d, -2d, ... to reduce label collisions.
                stagger_multiplier = (labeled_edge_index + 1) // 2
                sign = 1.0 if labeled_edge_index % 2 == 1 else -1.0
                label_shift = sign * stagger_multiplier * _EDGE_LABEL_STAGGER
            labeled_edge_index += 1
        parts.append(_render_edge(boxes[src], boxes[dst], label, horizontal, label_shift))

    for box in boxes:
        parts.append(_render_box(box))

    parts.append("</svg>")
    return "\n".join(parts)


def _render_box(box: _Box) -> str:
    style = _STYLES[box.kind]
    if box.tag in {"shape", "axes", "weight", "ambiguous"}:
        colors = VALUE_TAG_COLORS[box.tag]
        style = {
            "fill": colors["fill"],
            "stroke": colors["stroke"],
            "dashed": style["dashed"] if box.tag == "weight" else False,
            "rounded": True,
        }
    rx = 12 if style["rounded"] else 4
    dash = ' stroke-dasharray="4 3"' if style["dashed"] else ""
    out = [
        (
            f'<rect x="{_round(box.x)}" y="{_round(box.y)}" '
            f'width="{_round(box.width)}" height="{_round(box.height)}" '
            f'rx="{rx}" ry="{rx}" fill="{style["fill"]}" '
            f'stroke="{style["stroke"]}"{dash}/>'
        )
    ]
    cx = box.x + box.width / 2.0
    text_top = box.y + _BOX_PAD_Y + _LINE_HEIGHT * 0.75
    for i, line in enumerate(box.lines):
        y = text_top + i * _LINE_HEIGHT
        out.append(
            f'<text x="{_round(cx)}" y="{_round(y)}" '
            f'text-anchor="middle" fill="#000000">{_escape_xml(line)}</text>'
        )
    return "".join(out)


def _render_edge(
    src: _Box, dst: _Box, label: str, horizontal: bool, label_shift: float = 0.0
) -> str:
    """Renders one edge and shifts its label orthogonally to avoid overlaps."""
    if horizontal:
        x1 = src.x + src.width
        y1 = src.y + src.height / 2.0
        x2 = dst.x
        y2 = dst.y + dst.height / 2.0
    else:
        x1 = src.x + src.width / 2.0
        y1 = src.y + src.height
        x2 = dst.x + dst.width / 2.0
        y2 = dst.y
    out = [
        (
            f'<line x1="{_round(x1)}" y1="{_round(y1)}" '
            f'x2="{_round(x2)}" y2="{_round(y2)}" '
            f'stroke="#555555" marker-end="url(#arrow)"/>'
        )
    ]
    if label:
        mx = (x1 + x2) / 2.0
        my = (y1 + y2) / 2.0
        if horizontal:
            my += label_shift
        else:
            mx += label_shift
        out.append(
            f'<text x="{_round(mx)}" y="{_round(my)}" text-anchor="middle" '
            f'fill="{_EDGE_LABEL_COLOR}" font-size="{_EDGE_FONT_SIZE}">'
            f"{_escape_xml(label)}</text>"
        )
    return "".join(out)


def _round(value: float) -> str:
    """Formats a coordinate with at most two decimals, dropping trailing zeros."""
    return f"{value:.2f}".rstrip("0").rstrip(".")


# Re-exported for callers that want the iterable helper.
__all__ = ["to_svg", "to_svg_graph"]
