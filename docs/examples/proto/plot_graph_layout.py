"""
.. _l-example-plot-graph-layout:

Plotting incremental ONNX graph positions
=========================================

The native :class:`onnx_light.onnx.GraphLayout` computes node rectangles and
ordered input/output port routes without changing the graph's execution order.
This example draws those coordinates directly with Matplotlib, then resizes a
node and appends a branch without recomputing the whole layout.

Coordinates describe the top-left corner of each rectangle, with y increasing
downward. Graph inputs are edge endpoints, not additional operator nodes.
Finally, the example saves positions in ONNX metadata and reloads them.
No external model, graph-rendering executable, or network access is needed.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import onnx_light.onnx as onnx
import onnx_light.onnx.helper as oh
from onnx_light.onnx.checker import check_model

# %%
# Build a branching graph
# -----------------------
#
# Both branches consume ``X`` and meet at ``Add``. Node names are only labels:
# layout IDs remain stable independently of those names.

model = oh.make_model(
    oh.make_graph(
        [
            oh.make_node("Relu", ["X"], ["positive"], name="positive branch"),
            oh.make_node("Neg", ["X"], ["negative"], name="negative branch"),
            oh.make_node("Add", ["positive", "negative"], ["sum"], name="join"),
            oh.make_node("Sigmoid", ["sum"], ["Y"], name="output"),
        ],
        "layout_demo",
        [oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 4])],
        [oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [2, 4])],
    ),
    opset_imports=[oh.make_opsetid("", 18)],
    ir_version=10,
)
check_model(model)
model = onnx.load(model.SerializeToString(), layout=True)
layout = onnx.GraphLayout(model.graph)
layout.load()

# %%
# Draw native rectangles and port routes
# -------------------------------------
#
# The renderer does not run its own layout algorithm. Each edge follows
# ``LayoutEdge.points``; the final point is the target input port.


def draw_layout(
    ax, graph: onnx.GraphProto | onnx.FunctionProto, placement: onnx.GraphLayout, title: str
):
    """Draws native node positions and routed tensor edges on an axes."""
    for node_id, node in zip(placement.order, graph.node):
        box = placement.get(node_id)
        if box.folded:
            continue
        for edge in placement.edges(node_id):
            xs = [point.x for point in edge.points]
            ys = [point.y for point in edge.points]
            if edge.folded:
                ax.text(xs[-1] - 4, ys[-1], edge.value, ha="right", va="center", fontsize=8)
                continue
            ax.plot(xs, ys, color="0.45", linewidth=1, zorder=1)
            ax.annotate(
                "",
                xy=(xs[-1], ys[-1]),
                xytext=(xs[-2], ys[-2]),
                arrowprops={"arrowstyle": "->", "color": "0.45"},
            )
            if edge.source == 0:
                ax.text(xs[0] - 4, ys[0], edge.value, ha="right", va="center", fontsize=8)
        ax.add_patch(
            Rectangle(
                (box.x, box.y),
                box.width,
                box.height,
                facecolor="#e0efff",
                edgecolor="#285780",
                zorder=2,
            )
        )
        ax.text(
            box.x + box.width / 2,
            box.y + box.height / 2,
            f"{node.name}\n{node.op_type} / ID {node_id}\n({box.x:g}, {box.y:g})",
            ha="center",
            va="center",
            fontsize=8,
            zorder=3,
        )
    ax.autoscale_view()
    ax.margins(0.15)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xlabel("x (visual units)")
    ax.set_ylabel("y (visual units, downward)")
    ax.set_title(title)
    ax.grid(alpha=0.2)


figure, axes = plt.subplots(2, 1, figsize=(12, 7), layout="constrained")
draw_layout(axes[0], model.graph, layout, "Initial native layout")
for node_id in layout.order:
    box = layout.get(node_id)
    print(f"ID {node_id}: x={box.x:g}, y={box.y:g}, size={box.width:g} x {box.height:g}")

# %%
# Apply local edits
# -----------------
#
# Widening the positive branch shifts its consumers to the right. The negative
# branch is independent of this edit and retains its coordinates. Appending an
# inspection node adds another consumer of ``positive``; ``Y`` remains the
# graph output. All semantic node edits go through the live layout object.

positive_id, negative_id, _, _ = layout.order
negative_before = layout.get(negative_id)
assert layout.set_size(positive_id, 280, 110)
inspection_id = layout.append(
    oh.make_node("Identity", ["positive"], ["inspection"], name="inspection branch")
)
negative_after = layout.get(negative_id)
assert (negative_after.x, negative_after.y) == (negative_before.x, negative_before.y)
print(f"Appended ID {inspection_id}; last edit visited {layout.stats.nodes} nodes.")
draw_layout(axes[1], model.graph, layout, "After resizing and appending a branch")
check_model(model)

# %%
# Persist and reload coordinates
# ------------------------------
#
# ``persist()`` writes namespaced metadata, not operator attributes. Serializing
# to bytes keeps the example self-contained; the same metadata survives
# ``onnx.save(model, "positioned.onnx")`` followed by ``onnx.load(..., layout=True)``.

layout.persist()
restored_model = onnx.load(model.SerializeToString(), layout=True)
restored_layout = onnx.GraphLayout(restored_model.graph)
restored_layout.load()
assert restored_layout.order == layout.order
for node_id in layout.order:
    before = layout.get(node_id)
    after = restored_layout.get(node_id)
    assert (before.x, before.y, before.width, before.height) == (
        after.x,
        after.y,
        after.width,
        after.height,
    )
print("Stable IDs and all node rectangles survived the ONNX round trip.")

# %%
# Lay out a local function body
# -----------------------------
#
# The same API binds directly to a ``FunctionProto``. Function inputs are local
# source endpoints, and IDs are independent of those in the main graph.
# ``onnx.load(..., layout=True)`` also places every local function in a model.

function = oh.make_function(
    "demo",
    "Activation",
    ["X"],
    ["Y"],
    [
        oh.make_node("Relu", ["X"], ["positive"], name="function activation"),
        oh.make_node("Neg", ["positive"], ["Y"], name="function output"),
    ],
    opset_imports=[oh.make_opsetid("", 18)],
)
function_layout = onnx.GraphLayout(function)
function_layout.load()
function_layout.persist()
function_figure, function_axes = plt.subplots(figsize=(10, 3), layout="constrained")
draw_layout(function_axes, function, function_layout, "Native FunctionProto layout")
plt.show()
