# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for :func:`onnx_light.tools.pretty_onnx`.

The pretty-printer is duck-typed against the ONNX message API so the
tests build small protos out of :class:`types.SimpleNamespace` objects
(matching the pattern used by :mod:`test_tools_mermaid`).
"""

from __future__ import annotations

import unittest

from onnx_light.ext_test_case import ExtTestCase
from types import SimpleNamespace

from onnx_light.tools import pretty_onnx


def _vi(name: str, elem_type: int = 1, dims: tuple = ()) -> SimpleNamespace:
    """Returns a minimal ValueInfoProto-like object."""
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
    """Returns a minimal AttributeProto-like object with all list fields preset."""
    base = {
        "name": name,
        "type": 0,
        "f": 0.0,
        "i": 0,
        "s": b"",
        "t": None,
        "tp": None,
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
    metadata: dict | None = None,
) -> SimpleNamespace:
    """Returns a minimal NodeProto-like object."""
    return SimpleNamespace(
        op_type=op_type,
        input=list(inputs),
        output=list(outputs),
        name=name,
        domain=domain,
        attribute=list(attributes or []),
        metadata_props=[SimpleNamespace(key=k, value=v) for k, v in (metadata or {}).items()],
    )


def _init(name: str, dims: tuple = (), data_type: int = 1) -> SimpleNamespace:
    """Returns a minimal TensorProto initializer."""
    return SimpleNamespace(name=name, dims=list(dims), data_type=data_type)


def _graph(
    nodes: list, inputs: list, outputs: list, initializers: list | None = None, name: str = ""
) -> SimpleNamespace:
    """Returns a minimal GraphProto-like object."""
    return SimpleNamespace(
        name=name,
        node=nodes,
        input=inputs,
        output=outputs,
        initializer=initializers or [],
        value_info=[],
    )


def _model(graph: SimpleNamespace, opsets: list | None = None) -> SimpleNamespace:
    """Returns a minimal ModelProto-like object."""
    return SimpleNamespace(
        graph=graph, opset_import=opsets or [SimpleNamespace(domain="", version=18)], functions=[]
    )


def _function(
    name: str, inputs: list, outputs: list, nodes: list, domain: str = ""
) -> SimpleNamespace:
    """Returns a minimal FunctionProto-like object."""
    return SimpleNamespace(
        name=name, domain=domain, input=list(inputs), output=list(outputs), node=nodes
    )


class TestPrettyOnnx(ExtTestCase):
    def test_value_info(self) -> None:
        self.assertEqual(pretty_onnx(_vi("X", dims=(1, 3))), "float[1,3] X")
        self.assertEqual(pretty_onnx(_vi("Y", dims=("N", 4))), "float[N,4] Y")

    def test_attribute_scalars(self) -> None:
        self.assertEqual(pretty_onnx(_attr("axis", type=2, i=3)), "axis=3")
        self.assertEqual(pretty_onnx(_attr("alpha", type=1, f=0.5)), "alpha=0.5")
        self.assertEqual(pretty_onnx(_attr("mode", type=3, s=b"constant")), "mode='constant'")

    def test_attribute_lists(self) -> None:
        self.assertEqual(pretty_onnx(_attr("axes", type=7, ints=[1, 2])), "axes=[1, 2]")
        rendered = pretty_onnx(_attr("vals", type=6, floats=[0.5, 1.5]))
        self.assertTrue(rendered.startswith("vals=["))
        self.assertIn("0.5", rendered)
        self.assertEqual(
            pretty_onnx(_attr("kinds", type=8, strings=[b"a", b"b"])), "kinds=['a', 'b']"
        )

    def test_node_basic(self) -> None:
        node = _node("Add", ["X", "Y"], ["Z"], name="add0")
        self.assertEqual(pretty_onnx(node), "Add(X, Y) -> Z")

    def test_node_with_domain(self) -> None:
        node = _node("MyOp", ["A"], ["B"], domain="custom")
        self.assertEqual(pretty_onnx(node), "custom.MyOp(A) -> B")

    def test_node_with_attributes_single(self) -> None:
        node = _node("Mul", ["A", "B"], ["C"], attributes=[_attr("axis", type=2, i=1)])
        self.assertEqual(pretty_onnx(node, with_attributes=True), "Mul(A, B) -> C  ---  axis=1")

    def test_node_with_attributes_multi(self) -> None:
        node = _node(
            "Op", ["A"], ["B"], attributes=[_attr("a", type=2, i=1), _attr("b", type=2, i=2)]
        )
        text = pretty_onnx(node, with_attributes=True)
        self.assertEqual(text, "Op(A) -> B\n    a=1\n    b=2")

    def test_highlight(self) -> None:
        node = _node("Add", ["X", "Y"], ["Z"])
        self.assertEqual(pretty_onnx(node, highlight={"Y"}), "Add(X, **Y**) -> Z")

    def test_tensor_proto(self) -> None:
        # data_type=1 (FLOAT), shape 2x3
        self.assertEqual(pretty_onnx(_init("W", dims=(2, 3))), "onnx.TensorProto:1:2x3:W")

    def test_graph_simple_text_plot(self) -> None:
        g = _graph(
            nodes=[
                _node("Add", ["X", "Y"], ["T"], name="add0"),
                _node("Mul", ["T", "W"], ["Z"], name="mul0"),
            ],
            inputs=[_vi("X", dims=(1, 3)), _vi("Y", dims=(1, 3))],
            outputs=[_vi("Z", dims=(1, 3))],
            initializers=[_init("W", dims=(3,))],
            name="g",
        )
        text = pretty_onnx(g)
        self.assertIn("graph: name='g'", text)
        self.assertIn("input: float[1,3] X", text)
        self.assertIn("init: float[3] W", text)
        self.assertIn("0: Add(X, Y) -> T", text)
        self.assertIn("1: Mul(T, W) -> Z", text)
        self.assertIn("output: float[1,3] Z", text)

    def test_model(self) -> None:
        g = _graph(
            [_node("Identity", ["X"], ["Y"])], [_vi("X", dims=("N",))], [_vi("Y", dims=("N",))]
        )
        text = pretty_onnx(_model(g))
        self.assertIn("opset: domain='' version=18", text)
        self.assertIn("input: float[N] X", text)
        self.assertIn("0: Identity(X) -> Y", text)
        self.assertIn("output: float[N] Y", text)

    def test_model_function_node_indexes(self) -> None:
        g = _graph([], [], [])
        model = _model(g)
        model.functions = [
            _function("f", ["X"], ["Y"], [_node("Identity", ["X"], ["Y"])], "custom")
        ]
        text = pretty_onnx(model)
        self.assertIn("function: f[custom]", text)
        self.assertIn("0: Identity(X) -> Y", text)

    def test_graph_node_index_with_multiline_attributes(self) -> None:
        g = _graph(
            [
                _node(
                    "Mul",
                    ["A", "B"],
                    ["C"],
                    attributes=[_attr("axis", type=2, i=1), _attr("beta", type=2, i=2)],
                )
            ],
            [_vi("A")],
            [_vi("C")],
        )
        text = pretty_onnx(g, with_attributes=True)
        self.assertIn("0: Mul(A, B) -> C", text)
        self.assertIn("   axis=1", text)
        self.assertIn("   beta=2", text)

    def test_assert_none(self) -> None:
        with self.assertRaises(AssertionError):
            pretty_onnx(None)

    def test_node_tag_shape(self) -> None:
        node = _node("Shape", ["X"], ["S"], metadata={"onnx_light.node_tag": "shape"})
        self.assertIn("[shape]", pretty_onnx(node, include_node_tags=True))
        self.assertNotIn("[shape]", pretty_onnx(node, include_node_tags=False))

    def test_node_tag_axes(self) -> None:
        node = _node("Gather", ["X", "I"], ["Y"], metadata={"onnx_light.node_tag": "axes"})
        self.assertIn("[axes]", pretty_onnx(node, include_node_tags=True))

    def test_node_tag_weight(self) -> None:
        node = _node("Constant", [], ["C"], metadata={"onnx_light.node_tag": "weight"})
        self.assertIn("[weight]", pretty_onnx(node, include_node_tags=True))

    def test_node_tag_unknown_not_shown(self) -> None:
        node = _node("Add", ["X", "Y"], ["Z"], metadata={"onnx_light.node_tag": "other"})
        text = pretty_onnx(node, include_node_tags=True)
        self.assertNotIn("[other]", text)

    def test_node_tag_absent_not_shown(self) -> None:
        node = _node("Add", ["X", "Y"], ["Z"])
        text = pretty_onnx(node, include_node_tags=True)
        self.assertNotIn("[", text)

    def test_inplace(self) -> None:
        node = _node("Relu", ["T"], ["Z"], metadata={"onnx_light.inplace_reuse": "0:0:equal"})
        text = pretty_onnx(node, include_inplace=True)
        self.assertIn("inplace: out0=in0(equal)", text)
        text_off = pretty_onnx(node, include_inplace=False)
        self.assertNotIn("inplace", text_off)

    def test_inplace_multiple(self) -> None:
        node = _node(
            "Foo",
            ["A", "B"],
            ["C", "D"],
            metadata={"onnx_light.inplace_reuse": "0:0:equal;1:1:greater"},
        )
        text = pretty_onnx(node, include_inplace=True)
        self.assertIn("inplace: out0=in0(equal), out1=in1(greater)", text)

    def test_inplace_absent_not_shown(self) -> None:
        node = _node("Add", ["X", "Y"], ["Z"])
        text = pretty_onnx(node, include_inplace=True)
        self.assertNotIn("inplace", text)

    def test_graph_with_tags_and_inplace(self) -> None:
        g = _graph(
            nodes=[
                _node(
                    "Shape",
                    ["X"],
                    ["S"],
                    name="shape0",
                    metadata={"onnx_light.node_tag": "shape"},
                ),
                _node(
                    "Relu",
                    ["X"],
                    ["Y"],
                    name="relu0",
                    metadata={"onnx_light.inplace_reuse": "0:0:equal"},
                ),
            ],
            inputs=[_vi("X")],
            outputs=[_vi("Y")],
        )
        text = pretty_onnx(g, include_node_tags=True, include_inplace=True)
        self.assertIn("[shape] Shape", text)
        self.assertIn("inplace: out0=in0(equal)", text)
        # Tags and inplace off by default.
        text_off = pretty_onnx(g)
        self.assertNotIn("[shape]", text_off)
        self.assertNotIn("inplace", text_off)

    def test_release(self) -> None:
        node = _node("Abs", ["A"], ["B"], metadata={"onnx_light.release_after": "A"})
        text = pretty_onnx(node, include_release=True)
        self.assertIn("release: A", text)
        text_off = pretty_onnx(node, include_release=False)
        self.assertNotIn("release", text_off)

    def test_release_multiple(self) -> None:
        node = _node("Add", ["A", "B"], ["C"], metadata={"onnx_light.release_after": "A;B"})
        text = pretty_onnx(node, include_release=True)
        self.assertIn("release: A, B", text)

    def test_release_absent_not_shown(self) -> None:
        node = _node("Add", ["X", "Y"], ["Z"])
        text = pretty_onnx(node, include_release=True)
        self.assertNotIn("release", text)

    def test_graph_with_inplace_and_release(self) -> None:
        g = _graph(
            nodes=[
                _node("Abs", ["X"], ["A"], name="abs0"),
                _node(
                    "Abs",
                    ["A"],
                    ["Y"],
                    name="abs1",
                    metadata={
                        "onnx_light.inplace_reuse": "0:0:equal",
                        "onnx_light.release_after": "A",
                    },
                ),
            ],
            inputs=[_vi("X")],
            outputs=[_vi("Y")],
        )
        text = pretty_onnx(g, include_inplace=True, include_release=True)
        self.assertIn("inplace: out0=in0(equal)", text)
        self.assertIn("release: A", text)
        # Off by default.
        text_off = pretty_onnx(g)
        self.assertNotIn("inplace", text_off)
        self.assertNotIn("release", text_off)


if __name__ == "__main__":
    unittest.main()
