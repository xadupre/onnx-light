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
        type=SimpleNamespace(
            tensor_type=SimpleNamespace(elem_type=elem_type, shape=SimpleNamespace(dim=dim)),
            sequence_type=None,
            optional_type=None,
            map_type=None,
        ),
    )


def _node(
    op_type: str, inputs: list, outputs: list, name: str = "", attributes: list | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        op_type=op_type,
        input=list(inputs),
        output=list(outputs),
        name=name,
        attribute=[SimpleNamespace(name=a) for a in (attributes or [])],
    )


def _init(name: str, dims: tuple = (), data_type: int = 1) -> SimpleNamespace:
    return SimpleNamespace(name=name, dims=list(dims), data_type=data_type)


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
    )


def _model(graph: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(graph=graph)


class TestSvg(unittest.TestCase):
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
        self.assertIn("add0", text)
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
        text_no_attr = to_svg(_model(g), include_attributes=False)
        self.assertNotIn("kernel_shape", text_no_attr)

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
        self.assertIn("my&quot;node", text)
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


if __name__ == "__main__":
    unittest.main()
