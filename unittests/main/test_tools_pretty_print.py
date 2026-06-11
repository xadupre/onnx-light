# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`onnx_light.tools.pretty_print`.

The pretty-printer is duck-typed against the ONNX message API so the
tests build small graphs out of :class:`types.SimpleNamespace` objects
to avoid pulling in the (compiled) ``onnx_light`` extensions or the
upstream :mod:`onnx` package.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from onnx_light.tools import pretty_print, pretty_print_graph, pretty_print_node


def _vi(name: str, elem_type: int = 1, dims: tuple = ()) -> SimpleNamespace:
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


def _attr(name: str, **kwargs) -> SimpleNamespace:
    base = {
        "name": name,
        "type": 0,
        "f": 0.0,
        "i": 0,
        "s": b"",
        "t": None,
        "floats": [],
        "ints": [],
        "strings": [],
        "tensors": [],
        "graphs": [],
        "sparse_tensors": [],
        "type_protos": [],
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def _node(
    op_type: str,
    inputs: list,
    outputs: list,
    name: str = "",
    attributes: list | None = None,
    domain: str = "",
) -> SimpleNamespace:
    return SimpleNamespace(
        op_type=op_type,
        input=list(inputs),
        output=list(outputs),
        name=name,
        domain=domain,
        attribute=list(attributes or []),
    )


def _init(name: str, dims: tuple = (), data_type: int = 1) -> SimpleNamespace:
    return SimpleNamespace(name=name, dims=list(dims), data_type=data_type)


def _graph(
    nodes: list, inputs: list, outputs: list, initializers: list | None = None, name: str = "main"
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        node=nodes,
        input=inputs,
        output=outputs,
        initializer=initializers or [],
        value_info=[],
    )


def _model(graph: SimpleNamespace, opsets: list | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        ir_version=8,
        producer_name="onnx_light_tests",
        producer_version="",
        domain="",
        model_version=0,
        graph=graph,
        opset_import=opsets or [SimpleNamespace(domain="", version=18)],
        functions=[],
    )


class TestPrettyPrint(unittest.TestCase):
    def test_node_basic(self) -> None:
        node = _node("Add", ["X", "Y"], ["Z"], name="add0")
        text = pretty_print_node(node)
        self.assertIn("Z = Add(X, Y)", text)
        self.assertIn("# add0", text)

    def test_node_with_attributes(self) -> None:
        attrs = [
            _attr("alpha", type=1, f=0.5),
            _attr("axis", type=2, i=3),
            _attr("axes", type=7, ints=[1, 2]),
            _attr("mode", type=3, s=b"constant"),
        ]
        node = _node("MyOp", ["A"], ["B"], attributes=attrs, domain="custom")
        text = pretty_print_node(node)
        self.assertIn("custom.MyOp", text)
        self.assertIn("alpha = 0.5", text)
        self.assertIn("axis = 3", text)
        self.assertIn("axes = [1, 2]", text)
        self.assertIn('mode = "constant"', text)

    def test_graph(self) -> None:
        g = _graph(
            nodes=[
                _node("Add", ["X", "Y"], ["T"], name="add0"),
                _node("Mul", ["T", "X"], ["Z"], name="mul0"),
            ],
            inputs=[_vi("X", dims=(1, 3)), _vi("Y", dims=(1, 3))],
            outputs=[_vi("Z", dims=(1, 3))],
            initializers=[_init("W", dims=(3,))],
        )
        text = pretty_print_graph(g)
        self.assertTrue(text.startswith("graph main ("))
        self.assertIn("float[1,3] X", text)
        self.assertIn("=> (float[1,3] Z)", text)
        self.assertIn("# initializers", text)
        self.assertIn("float[3] W", text)
        self.assertIn("T = Add(X, Y)", text)
        self.assertIn("Z = Mul(T, X)", text)
        self.assertTrue(text.rstrip().endswith("}"))

    def test_model(self) -> None:
        g = _graph(
            [_node("Identity", ["X"], ["Y"], name="id0")],
            [_vi("X", dims=("N",))],
            [_vi("Y", dims=("N",))],
        )
        text = pretty_print(_model(g))
        self.assertIn("ir_version: 8", text)
        self.assertIn('producer_name: "onnx_light_tests"', text)
        self.assertIn('opset_import: "ai.onnx" : 18', text)
        self.assertIn("graph main", text)
        self.assertIn("Y = Identity(X)", text)

    def test_accepts_graph_and_node(self) -> None:
        node = _node("Relu", ["X"], ["Y"])
        self.assertEqual(pretty_print(node), pretty_print_node(node))
        g = _graph([node], [_vi("X")], [_vi("Y")])
        self.assertEqual(pretty_print(g), pretty_print_graph(g))

    def test_unknown_input_fallback(self) -> None:
        # Objects that don't look like ONNX messages fall back to str().
        self.assertEqual(pretty_print(42), "42")


if __name__ == "__main__":
    unittest.main()
