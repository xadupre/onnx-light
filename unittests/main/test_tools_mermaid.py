# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`onnx_light.tools.mermaid`.

The Mermaid converter is duck-typed against the ONNX message API so the
tests build small graphs out of :class:`types.SimpleNamespace` objects
to avoid pulling in the (compiled) ``onnx_light`` extensions or the
upstream :mod:`onnx` package.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from onnx_light.tools import to_mermaid, to_mermaid_graph
from onnx_light.tools.mermaid import _escape_label, _format_shape


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
    g = SimpleNamespace(
        node=nodes,
        input=inputs,
        output=outputs,
        initializer=initializers or [],
        value_info=value_info or [],
    )
    return g


def _model(graph: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(graph=graph)


class TestMermaid(unittest.TestCase):
    def test_simple_model(self) -> None:
        g = _graph(
            nodes=[
                _node("Add", ["X", "Y"], ["T"], name="add0"),
                _node("Mul", ["T", "X"], ["Z"], name="mul0"),
            ],
            inputs=[_vi("X", dims=(1, 3)), _vi("Y", dims=(1, 3))],
            outputs=[_vi("Z", dims=(1, 3))],
        )
        text = to_mermaid(_model(g))
        self.assertTrue(text.startswith("flowchart TB\n"))
        self.assertIn("Add", text)
        self.assertIn("Mul", text)
        # Inputs and outputs are styled differently.
        self.assertIn(":::onnxInput", text)
        self.assertIn(":::onnxOutput", text)
        self.assertIn(":::onnxOp", text)
        # Class definitions are emitted.
        self.assertIn("classDef onnxOp", text)
        # Edges connect tensors and operators.
        self.assertRegex(text, r"t_X -->.* n_add0")
        self.assertRegex(text, r"n_add0 -->.* t_T")
        self.assertRegex(text, r"n_mul0 -->.* t_Z")

    def test_direction_validation(self) -> None:
        g = _graph([_node("Identity", ["X"], ["Y"])], [_vi("X")], [_vi("Y")])
        for direction in ("TB", "TD", "BT", "LR", "RL"):
            text = to_mermaid(_model(g), direction=direction)
            self.assertTrue(text.startswith(f"flowchart {direction}\n"))
        with self.assertRaises(ValueError):
            to_mermaid(_model(g), direction="XX")

    def test_accepts_graph(self) -> None:
        g = _graph([_node("Identity", ["X"], ["Y"])], [_vi("X")], [_vi("Y")])
        text_from_model = to_mermaid(_model(g))
        text_from_graph = to_mermaid(g)
        self.assertEqual(text_from_model, text_from_graph)

    def test_rejects_unknown_input(self) -> None:
        with self.assertRaises(TypeError):
            to_mermaid(42)
        with self.assertRaises(TypeError):
            to_mermaid_graph(42)

    def test_initializers(self) -> None:
        g = _graph(
            nodes=[_node("Mul", ["X", "W"], ["Y"], name="mul0")],
            inputs=[_vi("X", dims=(4,))],
            outputs=[_vi("Y", dims=(4,))],
            initializers=[_init("W", dims=(4,))],
        )
        text = to_mermaid(_model(g))
        self.assertIn(":::onnxInitializer", text)
        # The initializer is connected as an input to the operator.
        self.assertRegex(text, r"t_W -->.* n_mul0")

        text_no_init = to_mermaid(_model(g), include_initializers=False)
        self.assertNotIn(":::onnxInitializer", text_no_init)
        self.assertNotIn("t_W", text_no_init)

    def test_optional_input_skipped(self) -> None:
        g = _graph(nodes=[_node("Foo", ["X", ""], ["Y"])], inputs=[_vi("X")], outputs=[_vi("Y")])
        text = to_mermaid(_model(g))
        # The empty input must not introduce an edge.
        self.assertNotIn(" -->  ", text)

    def test_attributes(self) -> None:
        g = _graph(
            nodes=[
                _node("Conv", ["X", "W"], ["Y"], name="c0", attributes=["kernel_shape", "pads"])
            ],
            inputs=[_vi("X"), _vi("W")],
            outputs=[_vi("Y")],
        )
        text = to_mermaid(_model(g), include_attributes=True)
        self.assertIn("kernel_shape", text)
        self.assertIn("pads", text)
        text_no_attr = to_mermaid(_model(g), include_attributes=False)
        self.assertNotIn("kernel_shape", text_no_attr)

    def test_node_name_collision(self) -> None:
        """Two nodes with the same op_type and no name must get distinct ids."""
        g = _graph(
            nodes=[_node("Add", ["X", "Y"], ["A"]), _node("Add", ["A", "Y"], ["B"])],
            inputs=[_vi("X"), _vi("Y")],
            outputs=[_vi("B")],
        )
        text = to_mermaid(_model(g))
        # Both Add operators should appear with distinct identifiers.
        op_lines = [
            line
            for line in text.splitlines()
            if line.strip().startswith("n_") and ":::onnxOp" in line
        ]
        self.assertEqual(len(op_lines), 2)
        ids = {line.strip().split("[", 1)[0] for line in op_lines}
        self.assertEqual(len(ids), 2)

    def test_special_chars_in_name(self) -> None:
        g = _graph(
            nodes=[_node("Add", ["a/b", "c d"], ["e:f"], name="my.node")],
            inputs=[_vi("a/b"), _vi("c d")],
            outputs=[_vi("e:f")],
        )
        text = to_mermaid(_model(g))
        # Names are sanitised for Mermaid identifiers.
        self.assertIn("t_a_b", text)
        self.assertIn("t_c_d", text)
        self.assertIn("t_e_f", text)
        self.assertIn("n_my_node", text)
        # Original names still appear inside the (escaped) labels.
        self.assertIn("a/b", text)

    def test_format_shape(self) -> None:
        ti = SimpleNamespace(
            tensor_type=SimpleNamespace(
                elem_type=7,
                shape=SimpleNamespace(
                    dim=[
                        SimpleNamespace(dim_value=2, dim_param=""),
                        SimpleNamespace(dim_value=0, dim_param="N"),
                        SimpleNamespace(dim_value=0, dim_param=""),
                    ]
                ),
            ),
            sequence_type=None,
            optional_type=None,
            map_type=None,
        )
        self.assertEqual(_format_shape(ti), "int64[2,N,?]")
        self.assertEqual(_format_shape(None), "")

    def test_escape_label(self) -> None:
        self.assertEqual(_escape_label("a<b>c"), "a#lt;b#gt;c")
        self.assertEqual(_escape_label('x"y'), "x#quot;y")
        self.assertEqual(_escape_label("line1\nline2"), "line1<br/>line2")
        # Brackets and pipes are allowed verbatim inside a quoted label.
        self.assertEqual(_escape_label("float[1,3]"), "float[1,3]")


if __name__ == "__main__":
    unittest.main()
