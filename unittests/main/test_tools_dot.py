# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`onnx_light.tools.dot`.

The DOT converter is duck-typed against the ONNX message API so the
tests build small graphs out of :class:`types.SimpleNamespace` objects
to avoid pulling in the (compiled) ``onnx_light`` extensions or the
upstream :mod:`onnx` package.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from onnx_light.ext_test_case import HAS_OPTIM_EXT
from onnx_light.tools import to_dot, to_dot_graph
from onnx_light.tools.dot import _escape_dot_label


def _vi(name: str, elem_type: int = 1, dims: tuple = ()) -> SimpleNamespace:
    """Builds a minimal ValueInfoProto-like object."""
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


class TestDot(unittest.TestCase):
    def test_simple_model(self) -> None:
        g = _graph(
            nodes=[
                _node("Add", ["X", "Y"], ["T"], name="add0"),
                _node("Mul", ["T", "X"], ["Z"], name="mul0"),
            ],
            inputs=[_vi("X", dims=(1, 3)), _vi("Y", dims=(1, 3))],
            outputs=[_vi("Z", dims=(1, 3))],
        )
        text = to_dot(_model(g))
        self.assertTrue(text.startswith("digraph onnx {"))
        self.assertIn("Add", text)
        self.assertIn("Mul", text)
        # Inputs are ellipse, ops are box.
        self.assertIn("shape=ellipse", text)
        self.assertIn("shape=box", text)
        # Edges connect tensors and operators.
        self.assertIn('"t_X" -> "n_add0"', text)
        self.assertIn('"n_add0" -> "t_T"', text)
        self.assertIn('"n_mul0" -> "t_Z"', text)
        self.assertTrue(text.endswith("}"))

    def test_direction_validation(self) -> None:
        g = _graph([_node("Identity", ["X"], ["Y"])], [_vi("X")], [_vi("Y")])
        for direction in ("TB", "BT", "LR", "RL"):
            text = to_dot(_model(g), direction=direction)
            self.assertIn(f'rankdir="{direction}"', text)
        with self.assertRaises(ValueError):
            to_dot(_model(g), direction="XX")

    def test_accepts_graph(self) -> None:
        g = _graph([_node("Identity", ["X"], ["Y"])], [_vi("X")], [_vi("Y")])
        text_from_model = to_dot(_model(g))
        text_from_graph = to_dot(g)
        self.assertEqual(text_from_model, text_from_graph)

    def test_rejects_unknown_input(self) -> None:
        with self.assertRaises(TypeError):
            to_dot(42)
        with self.assertRaises(TypeError):
            to_dot_graph(42)

    def test_initializers(self) -> None:
        g = _graph(
            nodes=[_node("Mul", ["X", "W"], ["Y"], name="mul0")],
            inputs=[_vi("X", dims=(4,))],
            outputs=[_vi("Y", dims=(4,))],
            initializers=[_init("W", dims=(4,))],
        )
        text = to_dot(_model(g))
        self.assertIn("shape=cylinder", text)
        # The initializer is connected as an input to the operator.
        self.assertIn('"t_W" -> "n_mul0"', text)

        text_no_init = to_dot(_model(g), include_initializers=False)
        self.assertNotIn("shape=cylinder", text_no_init)
        self.assertNotIn("t_W", text_no_init)

    def test_optional_input_skipped(self) -> None:
        g = _graph(nodes=[_node("Foo", ["X", ""], ["Y"])], inputs=[_vi("X")], outputs=[_vi("Y")])
        text = to_dot(_model(g))
        # The empty input must not introduce an edge to an empty-named node.
        self.assertNotIn('"" ->', text)

    def test_attributes(self) -> None:
        g = _graph(
            nodes=[
                _node("Conv", ["X", "W"], ["Y"], name="c0", attributes=["kernel_shape", "pads"])
            ],
            inputs=[_vi("X"), _vi("W")],
            outputs=[_vi("Y")],
        )
        text = to_dot(_model(g), include_attributes=True)
        self.assertIn("kernel_shape", text)
        self.assertIn("pads", text)
        text_no_attr = to_dot(_model(g), include_attributes=False)
        self.assertNotIn("kernel_shape", text_no_attr)

    def test_long_names_are_shortened(self) -> None:
        g = _graph(
            nodes=[
                _node(
                    "Identity",
                    ["very_long_input_name_abcdef"],
                    ["very_long_output_name_uvwxyz"],
                    name="very_long_node_name_123456789",
                )
            ],
            inputs=[_vi("very_long_input_name_abcdef")],
            outputs=[_vi("very_long_output_name_uvwxyz")],
        )
        text = to_dot(_model(g))
        self.assertIn('label="Identity\\n_name_123456789"', text)
        self.assertIn('label="put_name_uvwxyz\\nfloat"', text)

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
        text = to_dot(_model(g), include_inplace=True)
        self.assertIn("inplace: out0=in0(equal)", text)
        text_off = to_dot(_model(g), include_inplace=False)
        self.assertNotIn("inplace", text_off)

    def test_shapes_in_labels(self) -> None:
        g = _graph(
            nodes=[_node("Identity", ["X"], ["Y"])],
            inputs=[_vi("X", elem_type=1, dims=(3, 4))],
            outputs=[_vi("Y", elem_type=1, dims=(3, 4))],
        )
        text = to_dot(_model(g), include_shapes=True)
        self.assertIn("float[3,4]", text)

        text_no_shapes = to_dot(_model(g), include_shapes=False)
        self.assertNotIn("float[3,4]", text_no_shapes)

    def test_node_name_collision(self) -> None:
        """Two nodes with the same op_type and no name must get distinct ids."""
        g = _graph(
            nodes=[_node("Add", ["X", "Y"], ["A"]), _node("Add", ["A", "Y"], ["B"])],
            inputs=[_vi("X"), _vi("Y")],
            outputs=[_vi("B")],
        )
        text = to_dot(_model(g))
        op_lines = [line for line in text.splitlines() if '"n_' in line and "shape=box" in line]
        self.assertEqual(len(op_lines), 2)
        ids = {line.strip().split('"')[1] for line in op_lines}
        self.assertEqual(len(ids), 2)

    def test_special_chars_in_name(self) -> None:
        g = _graph(
            nodes=[_node("Add", ["a/b", "c d"], ["e:f"], name="my.node")],
            inputs=[_vi("a/b"), _vi("c d")],
            outputs=[_vi("e:f")],
        )
        text = to_dot(_model(g))
        # Names are sanitised for internal identifiers.
        self.assertIn("t_a_b", text)
        self.assertIn("t_c_d", text)
        self.assertIn("t_e_f", text)
        self.assertIn("n_my_node", text)
        # Original names still appear inside the (escaped) labels.
        self.assertIn("a/b", text)

    def test_escape_dot_label(self) -> None:
        self.assertEqual(_escape_dot_label('say "hello"'), 'say \\"hello\\"')
        self.assertEqual(_escape_dot_label("line1\nline2"), "line1\\nline2")
        self.assertEqual(_escape_dot_label("a<b>c"), "a\\<b\\>c")
        self.assertEqual(_escape_dot_label("{key|val}"), "\\{key\\|val\\}")
        self.assertEqual(_escape_dot_label("back\\slash"), "back\\\\slash")
        # Combined special characters.
        self.assertEqual(_escape_dot_label('a\\"b\nc<d>{e|f}'), 'a\\\\\\"b\\nc\\<d\\>\\{e\\|f\\}')
        # Consecutive backslashes.
        self.assertEqual(_escape_dot_label("\\\\"), "\\\\\\\\")

    @unittest.skipUnless(HAS_OPTIM_EXT, "requires onnx_light C++ shape_inference bindings")
    def test_tagged_style(self) -> None:
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
        text = to_dot(oh.make_model(g))
        # Tagged tensor S should have the shape tag fill colour.
        self.assertIn("#f4d6ff", text)


if __name__ == "__main__":
    unittest.main()
