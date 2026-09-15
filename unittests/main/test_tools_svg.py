# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`onnx_light.tools.svg`.

The SVG converter is duck-typed against the ONNX message API so the
tests build small graphs out of :class:`types.SimpleNamespace` objects
to avoid pulling in the (compiled) ``onnx_light`` extensions or the
upstream :mod:`onnx` package.
"""

from __future__ import annotations

import unittest
import xml.dom.minidom
from types import SimpleNamespace

from onnx_light.ext_test_case import ExtTestCase, HAS_OPTIM_EXT
from onnx_light.tools import to_svg, to_svg_graph
from onnx_light.tools.svg import _escape_xml


def _vi(name: str, elem_type: int = 1, dims: tuple = ()) -> SimpleNamespace:
    """Build a minimal ValueInfoProto-like object."""
    dim = [
        (
            SimpleNamespace(dim_value=int(d), dim_param="")
            if isinstance(d, int)
            else SimpleNamespace(dim_value=0, dim_param=str(d))
        )
        for d in dims
    ]
    return SimpleNamespace(
        name=name,
        metadata_props=[],
        type=SimpleNamespace(
            tensor_type=SimpleNamespace(elem_type=elem_type, shape=SimpleNamespace(dim=dim)),
            sequence_type=None,
            optional_type=None,
            map_type=None,
        ),
    )


def _node(
    op_type: str,
    inputs: list,
    outputs: list,
    name: str = "",
    attributes: list | None = None,
    metadata: dict | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        op_type=op_type,
        input=list(inputs),
        output=list(outputs),
        name=name,
        attribute=[SimpleNamespace(name=a) for a in (attributes or [])],
        metadata_props=[SimpleNamespace(key=k, value=v) for k, v in (metadata or {}).items()],
    )


def _init(name: str, dims: tuple = (), data_type: int = 1) -> SimpleNamespace:
    return SimpleNamespace(name=name, dims=list(dims), data_type=data_type, metadata_props=[])


def _graph(
    nodes: list,
    inputs: list,
    outputs: list,
    initializers: list | None = None,
    value_info: list | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        node=nodes,
        input=inputs,
        output=outputs,
        initializer=initializers or [],
        value_info=value_info or [],
        metadata_props=[],
    )


def _model(graph: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(graph=graph)


def _count_crossings(boxes: list, edges: list, by_order: bool) -> int:
    """Counts edge crossings using the within-layer position of each box."""
    layers: dict[int, list] = {}
    for box in boxes:
        layers.setdefault(box.layer, []).append(box)
    position: dict[int, int] = {}
    for layer_boxes in layers.values():
        if by_order:
            ordered = sorted(layer_boxes, key=lambda box: (box.order, box.id))
        else:
            ordered = sorted(layer_boxes, key=lambda box: box.id)
        for index, box in enumerate(ordered):
            position[box.id] = index
    crossings = 0
    for i, (src1, dst1, _l1) in enumerate(edges):
        for src2, dst2, _l2 in edges[i + 1 :]:
            # Only edges spanning the same pair of layers can be compared.
            if boxes[src1].layer != boxes[src2].layer:
                continue
            if boxes[dst1].layer != boxes[dst2].layer:
                continue
            top = position[src1] - position[src2]
            bottom = position[dst1] - position[dst2]
            if top * bottom < 0:
                crossings += 1
    return crossings


class TestSvg(ExtTestCase):
    def _assert_valid_svg(self, text: str) -> None:
        self.assertTrue(text.startswith("<svg "))
        self.assertTrue(text.rstrip().endswith("</svg>"))
        # The result must be well-formed XML.
        xml.dom.minidom.parseString(text)

    def test_simple_model(self) -> None:
        g = _graph(
            nodes=[
                _node("Add", ["X", "Y"], ["T"], name="add0"),
                _node("Mul", ["T", "X"], ["Z"], name="mul0"),
            ],
            inputs=[_vi("X", dims=(1, 3)), _vi("Y", dims=(1, 3))],
            outputs=[_vi("Z", dims=(1, 3))],
        )
        text = to_svg(_model(g))
        self._assert_valid_svg(text)
        self.assertIn("Add", text)
        self.assertIn("Mul", text)
        self.assertNotIn("add0", text)
        # Distinct fill colours are used per kind of box.
        self.assertIn("#cde4ff", text)  # input
        self.assertIn("#ffe1b3", text)  # output
        self.assertIn("#d4ecd4", text)  # operator
        # Edges and the arrow marker are emitted.
        self.assertIn("<line", text)
        self.assertIn('marker-end="url(#arrow)"', text)

    def test_direction_validation(self) -> None:
        g = _graph([_node("Identity", ["X"], ["Y"])], [_vi("X")], [_vi("Y")])
        for direction in ("TB", "TD", "LR"):
            text = to_svg(_model(g), direction=direction)
            self._assert_valid_svg(text)
        with self.assertRaises(ValueError):
            to_svg(_model(g), direction="XX")

    def test_layout_validation(self) -> None:
        g = _graph([_node("Identity", ["X"], ["Y"])], [_vi("X")], [_vi("Y")])
        # The default "layered" layout is accepted and matches the default.
        self.assertEqual(to_svg(_model(g)), to_svg(_model(g), layout="layered"))
        with self.assertRaises(ValueError):
            to_svg(_model(g), layout="spring")
        with self.assertRaises(ValueError):
            to_svg_graph(g, layout="spring")

    def test_umap_layout_small_graph_falls_back(self) -> None:
        # Fewer than the minimum number of boxes falls back to the layered
        # layout, so no optional dependency is needed and the SVG is valid.
        g = _graph([_node("Identity", ["X"], ["Y"])], [_vi("X")], [_vi("Y")])
        text = to_svg(_model(g), layout="umap")
        self._assert_valid_svg(text)

    def test_umap_layout_requires_dependency(self) -> None:
        import importlib.util

        g = _graph(
            nodes=[
                _node("Add", ["X", "Y"], ["A"]),
                _node("Mul", ["A", "X"], ["B"]),
                _node("Sub", ["B", "Y"], ["Z"]),
            ],
            inputs=[_vi("X"), _vi("Y")],
            outputs=[_vi("Z")],
        )
        if importlib.util.find_spec("umap") is None or importlib.util.find_spec("numpy") is None:
            with self.assertRaises(ImportError):
                to_svg(_model(g), layout="umap")
        else:
            self._assert_valid_svg(to_svg(_model(g), layout="umap"))

    def test_accepts_graph(self) -> None:
        g = _graph([_node("Identity", ["X"], ["Y"])], [_vi("X")], [_vi("Y")])
        text_from_model = to_svg(_model(g))
        text_from_graph = to_svg(g)
        self.assertEqual(text_from_model, text_from_graph)

    def test_rejects_unknown_input(self) -> None:
        with self.assertRaises(TypeError):
            to_svg(42)
        with self.assertRaises(TypeError):
            to_svg_graph(42)

    def test_initializers(self) -> None:
        g = _graph(
            nodes=[_node("Mul", ["X", "W"], ["Y"], name="mul0")],
            inputs=[_vi("X", dims=(4,))],
            outputs=[_vi("Y", dims=(4,))],
            initializers=[_init("W", dims=(4,))],
        )
        text = to_svg(_model(g))
        self._assert_valid_svg(text)
        # The initializer box uses the dashed style.
        self.assertIn("stroke-dasharray", text)
        self.assertIn("#eeeeee", text)
        self.assertIn("W", text)

        text_no_init = to_svg(_model(g), include_initializers=False)
        self._assert_valid_svg(text_no_init)
        self.assertNotIn("stroke-dasharray", text_no_init)

    def test_optional_input_skipped(self) -> None:
        g = _graph(nodes=[_node("Foo", ["X", ""], ["Y"])], inputs=[_vi("X")], outputs=[_vi("Y")])
        text = to_svg(_model(g))
        self._assert_valid_svg(text)
        # One operator + one input + one output -> exactly two edges.
        self.assertEqual(text.count("<line"), 2)

    def test_shapes_in_labels(self) -> None:
        g = _graph(
            nodes=[_node("Identity", ["X"], ["Y"])],
            inputs=[_vi("X", elem_type=7, dims=(2, "N"))],
            outputs=[_vi("Y", elem_type=7, dims=(2, "N"))],
        )
        text = to_svg(_model(g), include_shapes=True)
        self.assertIn("int64[2,N]", text)
        text_no_shapes = to_svg(_model(g), include_shapes=False)
        self.assertNotIn("int64[2,N]", text_no_shapes)

    def test_attributes(self) -> None:
        g = _graph(
            nodes=[
                _node("Conv", ["X", "W"], ["Y"], name="c0", attributes=["kernel_shape", "pads"])
            ],
            inputs=[_vi("X"), _vi("W")],
            outputs=[_vi("Y")],
        )
        text = to_svg(_model(g), include_attributes=True)
        self.assertIn("kernel_shape", text)
        self.assertIn("pads", text)
        self.assertNotIn(">c0<", text)
        text_no_attr = to_svg(_model(g), include_attributes=False)
        self.assertNotIn("kernel_shape", text_no_attr)

    def test_edge_labels_use_tensor_names(self) -> None:
        g = _graph(
            nodes=[
                _node(
                    "Identity",
                    ["very_long_input_name_abcdef"],
                    ["very_long_output_name_uvwxyz"],
                    name="very_long_node_name_123456789",
                )
            ],
            inputs=[_vi("very_long_input_name_abcdef", dims=(2,))],
            outputs=[_vi("very_long_output_name_uvwxyz", dims=(2,))],
        )
        text = to_svg(_model(g), include_shapes=True)
        self._assert_valid_svg(text)
        self.assertIn("put_name_abcdef · float[2]", text)
        self.assertIn("put_name_uvwxyz · float[2]", text)
        self.assertNotIn(">very_long_output_name_uvwxyz<", text)
        self.assertNotIn(">very_long_node_name_123456789<", text)

    def test_edge_labels_are_staggered_to_reduce_collisions(self) -> None:
        from onnx_light.tools.svg import _Box, _render_svg

        src = _Box(0, "input", ["X"])
        dst = _Box(1, "op", ["Add"])
        src.x, src.y = 10.0, 10.0
        dst.x, dst.y = 10.0, 120.0
        svg = _render_svg(
            [src, dst], [(0, 1, "L0"), (0, 1, "L1")], 200.0, 220.0, horizontal=False
        )
        doc = xml.dom.minidom.parseString(svg)
        label_positions = {
            node.firstChild.nodeValue: float(node.getAttribute("x"))
            for node in doc.getElementsByTagName("text")
            if node.firstChild is not None and node.firstChild.nodeValue in {"L0", "L1"}
        }
        self.assertEqual(set(label_positions), {"L0", "L1"})
        self.assertNotEqual(label_positions["L0"], label_positions["L1"])

    def test_edge_labels_are_small_and_not_highlighted(self) -> None:
        g = _graph(
            nodes=[_node("Identity", ["X"], ["Y"])],
            inputs=[_vi("X", dims=(2,))],
            outputs=[_vi("Y", dims=(2,))],
        )
        text = to_svg(_model(g), include_shapes=True)
        # Edge labels must be rendered small and without a highlight halo,
        # i.e. no white stroke / paint-order on the label text.
        self.assertNotIn('paint-order="stroke"', text)
        self.assertNotIn('stroke="#ffffff"', text)
        self.assertIn('font-size="9"', text)

    def test_inplace(self) -> None:
        g = _graph(
            nodes=[
                _node("Add", ["X", "Y"], ["T"], name="add0"),
                _node(
                    "Relu",
                    ["T"],
                    ["Z"],
                    name="relu0",
                    metadata={"onnx_light.inplace_reuse": "0:0:equal"},
                ),
            ],
            inputs=[_vi("X"), _vi("Y")],
            outputs=[_vi("Z")],
        )
        text = to_svg(_model(g), include_inplace=True)
        self._assert_valid_svg(text)
        self.assertIn("inplace: out0=in0(equal)", text)
        text_off = to_svg(_model(g), include_inplace=False)
        self.assertNotIn("inplace", text_off)

    def test_release(self) -> None:
        g = _graph(
            nodes=[
                _node("Add", ["X", "Y"], ["T"], name="add0"),
                _node(
                    "Relu",
                    ["T"],
                    ["Z"],
                    name="relu0",
                    metadata={"onnx_light.release_after": "T;X"},
                ),
            ],
            inputs=[_vi("X"), _vi("Y")],
            outputs=[_vi("Z")],
        )
        text = to_svg(_model(g), include_release=True)
        self._assert_valid_svg(text)
        self.assertIn("release: T, X", text)
        text_off = to_svg(_model(g), include_release=False)
        self.assertNotIn("release:", text_off)

    @unittest.skipUnless(HAS_OPTIM_EXT, "requires onnx_light C++ shape_inference bindings")
    def test_tagged_colors(self) -> None:
        import onnx_light.onnx.helper as oh
        from onnx_light.onnx import TensorProto
        from onnx_light.tools import write_value_and_node_tags_to_metadata

        g = oh.make_graph(
            [
                oh.make_node("Shape", ["X"], ["S"], name="shape0"),
                oh.make_node("Reshape", ["X", "S"], ["Y"], name="reshape0"),
            ],
            "g",
            [oh.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2])],
        )
        write_value_and_node_tags_to_metadata(g)
        text = to_svg(oh.make_model(g))
        self.assertIn("#f4d6ff", text)

    def test_special_chars_escaped(self) -> None:
        g = _graph(
            nodes=[_node("Add", ["a<b", "c&d"], ["e>f"], name='my"node')],
            inputs=[_vi("a<b"), _vi("c&d")],
            outputs=[_vi("e>f")],
        )
        text = to_svg(_model(g))
        # The raw special characters must not appear unescaped in text nodes.
        self.assertIn("a&lt;b", text)
        self.assertIn("c&amp;d", text)
        self.assertIn("e&gt;f", text)
        self._assert_valid_svg(text)

    def test_escape_xml(self) -> None:
        self.assertEqual(_escape_xml("a<b>c"), "a&lt;b&gt;c")
        self.assertEqual(_escape_xml("x&y"), "x&amp;y")
        self.assertEqual(_escape_xml('q"z'), "q&quot;z")
        self.assertEqual(_escape_xml("float[1,3]"), "float[1,3]")

    def test_empty_graph(self) -> None:
        g = _graph(nodes=[], inputs=[], outputs=[])
        text = to_svg(_model(g))
        self._assert_valid_svg(text)

    def test_input_pulled_down_to_consumer(self) -> None:
        # ``W`` only feeds the last operator, so it must not stay stranded on
        # the first layer next to ``X`` and ``A``.
        from onnx_light.tools.svg import _assign_layers, _Box

        # 0:X 1:A 2:W (inputs) 3:Add 4:Relu 5:Mul 6:Z (output)
        boxes = [
            _Box(0, "input", ["X"]),
            _Box(1, "input", ["A"]),
            _Box(2, "input", ["W"]),
            _Box(3, "op", ["Add"]),
            _Box(4, "op", ["Relu"]),
            _Box(5, "op", ["Mul"]),
            _Box(6, "output", ["Z"]),
        ]
        edges = [(0, 3, ""), (1, 3, ""), (3, 4, ""), (4, 5, ""), (2, 5, ""), (5, 6, "")]
        _assign_layers(boxes, edges)
        # Inputs feeding the first operator stay on the first layer.
        self.assertEqual(boxes[0].layer, 0)
        self.assertEqual(boxes[1].layer, 0)
        # ``W`` is pulled down to just above its consumer ``Mul``.
        self.assertEqual(boxes[2].layer, boxes[5].layer - 1)
        self.assertGreater(boxes[2].layer, 0)

    def test_crossings_are_reduced(self) -> None:
        from onnx_light.tools.svg import _assign_layers, _Box, _minimize_crossings

        # Two parallel chains whose operators are declared in the opposite
        # order to their inputs, which makes the edges cross under the naive
        # insertion order.  The crossing-reduction pass should untangle them.
        boxes = [
            _Box(0, "input", ["A"]),
            _Box(1, "input", ["B"]),
            _Box(2, "op", ["rb"]),
            _Box(3, "op", ["ra"]),
            _Box(4, "output", ["Yb"]),
            _Box(5, "output", ["Ya"]),
        ]
        # Chain A: A(0) -> ra(3) -> Ya(5) ; Chain B: B(1) -> rb(2) -> Yb(4).
        edges = [(0, 3, ""), (3, 5, ""), (1, 2, ""), (2, 4, "")]
        _assign_layers(boxes, edges)
        before = _count_crossings(boxes, edges, by_order=False)
        _minimize_crossings(boxes, edges)
        after = _count_crossings(boxes, edges, by_order=True)
        self.assertGreater(before, 0)
        self.assertLess(after, before)
        self.assertEqual(after, 0)

    def test_crossings_are_eliminated_with_skip_edge(self) -> None:
        from onnx_light.tools.svg import _assign_layers, _Box, _minimize_crossings

        # X feeds three operators (adjacent edges) and also directly Out3, a
        # long skip edge spanning two layers.  Out3 is declared first so the
        # naive insertion order produces crossings.  All crossings must be
        # resolved after the crossing-reduction pass.
        #
        #   X (layer 0)
        #   ├──→ Op1 → Out1 (layer 2)
        #   ├──→ Op2 → Out2
        #   ├──→ Op3 → Out3
        #   └─────────→ Out3  ← skip edge
        boxes = [
            _Box(0, "input", ["X"]),
            _Box(1, "op", ["Op1"]),
            _Box(2, "op", ["Op2"]),
            _Box(3, "op", ["Op3"]),
            _Box(4, "output", ["Out3"]),  # declared first → wrong initial order
            _Box(5, "output", ["Out1"]),
            _Box(6, "output", ["Out2"]),
        ]
        edges = [
            (0, 1, ""),
            (0, 2, ""),
            (0, 3, ""),
            (1, 5, ""),
            (2, 6, ""),
            (3, 4, ""),
            (0, 4, ""),  # skip edge: X -> Out3
        ]
        _assign_layers(boxes, edges)
        before = _count_crossings(boxes, edges, by_order=False)
        _minimize_crossings(boxes, edges)
        after = _count_crossings(boxes, edges, by_order=True)
        self.assertGreater(before, 0)
        self.assertEqual(after, 0)


if __name__ == "__main__":
    unittest.main()
