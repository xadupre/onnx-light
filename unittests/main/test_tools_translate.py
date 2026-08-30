# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for :func:`onnx_light.tools.translate`.

The translator is duck-typed against the ONNX message API so the textual
tests build small protos out of :class:`types.SimpleNamespace` objects
(matching the pattern used by :mod:`test_tools_pretty_print`).  The
round-trip tests instead build real protos with
:mod:`onnx_light.onnx.helper` and execute the generated code.
"""

from __future__ import annotations

import math
import unittest

from onnx_light.ext_test_case import ExtTestCase
from types import SimpleNamespace

import numpy as np

from onnx_light.tools import translate, translate_header


def _vi(name: str, elem_type: int = 1, dims: tuple = ()) -> SimpleNamespace:
    """Returns a minimal ValueInfoProto-like object."""
    dim = [
        (
            SimpleNamespace(dim_value=int(d), dim_param="")
            if isinstance(d, int)
            else SimpleNamespace(dim_value=0, dim_param=(str(d) if d is not None else ""))
        )
        for d in dims
    ]
    return SimpleNamespace(
        name=name,
        type=SimpleNamespace(
            tensor_type=SimpleNamespace(elem_type=elem_type, shape=SimpleNamespace(dim=dim))
        ),
    )


def _attr(name: str, **kwargs) -> SimpleNamespace:
    """Returns a minimal AttributeProto-like object with all list fields preset."""
    base = {
        "name": name,
        "type": 0,
        "ref_attr_name": "",
        "f": 0.0,
        "i": 0,
        "s": b"",
        "t": None,
        "g": None,
        "floats": [],
        "ints": [],
        "strings": [],
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
    """Returns a minimal NodeProto-like object."""
    return SimpleNamespace(
        op_type=op_type,
        input=list(inputs),
        output=list(outputs),
        name=name,
        domain=domain,
        attribute=list(attributes or []),
    )


def _tensor(array: np.ndarray, name: str) -> SimpleNamespace:
    """Returns a minimal TensorProto-like object storing ``array`` as raw_data."""
    dtype_map = {"float32": 1, "int64": 7, "int32": 6, "float64": 11, "bool": 9}
    return SimpleNamespace(
        name=name,
        data_type=dtype_map[str(array.dtype)],
        dims=list(array.shape),
        raw_data=array.tobytes(),
    )


def _graph(
    nodes: list, inputs: list, outputs: list, initializers: list | None = None, name: str = ""
) -> SimpleNamespace:
    """Returns a minimal GraphProto-like object."""
    return SimpleNamespace(
        name=name,
        node=list(nodes),
        input=list(inputs),
        output=list(outputs),
        initializer=list(initializers or []),
    )


def _model(graph: SimpleNamespace, opsets: list | None = None, ir_version: int = 0):
    """Returns a minimal ModelProto-like object."""
    return SimpleNamespace(
        graph=graph,
        opset_import=opsets or [SimpleNamespace(domain="", version=18)],
        ir_version=ir_version,
    )


class TestTranslateText(ExtTestCase):
    """Textual (duck-typed) checks that do not require the C++ extension."""

    def _simple_model(self):
        graph = _graph(
            nodes=[
                _node("Reshape", ["X", "shape"], ["reshaped"]),
                _node(
                    "Transpose",
                    ["reshaped"],
                    ["Y"],
                    attributes=[_attr("perm", type=7, ints=[1, 0])],
                ),
            ],
            inputs=[_vi("X", dims=(None, None))],
            outputs=[_vi("Y", dims=(None, None))],
            initializers=[_tensor(np.array([-1, 1], dtype=np.int64), "shape")],
            name="simple",
        )
        return _model(graph, opsets=[SimpleNamespace(domain="", version=17)], ir_version=8)

    def test_header_compact(self) -> None:
        header = translate_header("onnx-compact")
        self.assertIn("import onnx_light.onnx as onnx", header)
        self.assertIn("import onnx_light.onnx.helper as oh", header)
        self.assertIn("import onnx_light.onnx.numpy_helper as onh", header)

    def test_header_builder(self) -> None:
        header = translate_header("builder")
        self.assertIn("from onnx_light.onnx_core.graph_builder import GraphBuilder", header)

    def test_header_cpp(self) -> None:
        header = translate_header("cpp")
        self.assertIn('#include "onnx_core/builder/graph_builder.h"', header)
        self.assertIn('#include "onnx_op/operator_sets.h"', header)

    def test_header_unknown(self) -> None:
        with self.assertRaises(ValueError):
            translate_header("does-not-exist")

    def test_translate_unknown_api(self) -> None:
        with self.assertRaises(ValueError):
            translate(self._simple_model(), api="does-not-exist")

    def test_compact_structure(self) -> None:
        code = translate(self._simple_model(), api="onnx-compact")
        self.assertIn("model = oh.make_model(", code)
        self.assertIn("oh.make_graph(", code)
        self.assertIn("oh.make_node('Reshape', ['X', 'shape'], ['reshaped'])", code)
        self.assertIn("perm=[1, 0]", code)
        self.assertIn("oh.make_tensor_value_info('X', onnx.TensorProto.FLOAT", code)
        self.assertIn("onh.from_array(np.array([-1, 1], dtype=np.int64), name='shape')", code)
        self.assertIn("opset_imports=[oh.make_opsetid('', 17)]", code)
        self.assertIn("ir_version=8", code)

    def test_builder_structure(self) -> None:
        code = translate(self._simple_model(), api="builder")
        self.assertIn("g = GraphBuilder('simple')", code)
        self.assertIn("g.set_opset_version('', 17)", code)
        self.assertIn("g.inp('X', onnx.TensorProto.FLOAT, (None, None))", code)
        self.assertIn("g.init(np.array([-1, 1], dtype=np.int64), name='shape')", code)
        self.assertIn("g.op.Reshape('X', 'shape', outputs=['reshaped'])", code)
        self.assertIn("perm=[1, 0]", code)
        self.assertIn("g.out('Y', onnx.TensorProto.FLOAT, (None, None))", code)
        self.assertIn("model = g.to_onnx('model', ir_version=8)", code)

    def test_cpp_structure(self) -> None:
        code = translate(self._simple_model(), api="cpp")
        self.assertIn("onnx_light::ModelProto BuildModel()", code)
        self.assertIn('g.SetOpsetVersion("", 17);', code)
        self.assertIn('g.MakeInput("X", onnx_light::core::symbolic::TensorType::kFloat', code)
        self.assertIn('initializer_0.set_name("shape");', code)
        self.assertIn("static_cast<char>(static_cast<unsigned char>(255))", code)
        self.assertIn('g.MakeNode("Reshape", {"X", "shape"}, {"reshaped"});', code)
        self.assertIn("attribute_1_0.add_ints(1);", code)
        self.assertIn("attribute_1_0.add_ints(0);", code)
        self.assertIn('g.MakeOutput("Y", onnx_light::core::symbolic::TensorType::kFloat', code)
        self.assertIn("return g.ToModel(8);", code)

    def test_cpp_preserves_custom_domain_attributes(self) -> None:
        graph = _graph(
            nodes=[
                _node(
                    "CustomOp",
                    ["X"],
                    ["Y"],
                    attributes=[_attr("alpha", type=1, f=0.5)],
                    domain="com.example",
                    name="custom",
                )
            ],
            inputs=[_vi("X", dims=(1,))],
            outputs=[_vi("Y", dims=(1,))],
        )
        code = translate(
            _model(graph, opsets=[SimpleNamespace(domain="com.example", version=1)]), api="cpp"
        )
        self.assertIn(
            'g.MakeNode("CustomOp", {"X"}, {"Y"}, "com.example", "custom", attributes_0);', code
        )

    def test_cpp_formats_non_finite_float_attributes(self) -> None:
        graph = _graph(
            nodes=[
                _node(
                    "CustomOp",
                    ["X"],
                    ["Y"],
                    attributes=[
                        _attr("nan", type=1, f=math.nan),
                        _attr("inf", type=1, f=math.inf),
                    ],
                )
            ],
            inputs=[_vi("X", dims=(1,))],
            outputs=[_vi("Y", dims=(1,))],
        )
        code = translate(
            _model(graph, opsets=[SimpleNamespace(domain="custom", version=1)]), api="cpp"
        )
        self.assertIn("std::numeric_limits<float>::quiet_NaN()", code)
        self.assertIn("std::numeric_limits<float>::infinity()", code)

    def test_cpp_rejects_unsupported_attribute_type(self) -> None:
        graph = _graph(
            nodes=[_node("CustomOp", ["X"], ["Y"], attributes=[_attr("value", type=4)])],
            inputs=[_vi("X", dims=(1,))],
            outputs=[_vi("Y", dims=(1,))],
        )
        with self.assertRaisesRegex(NotImplementedError, "attribute 'value' with type 4"):
            translate(
                _model(graph, opsets=[SimpleNamespace(domain="custom", version=1)]), api="cpp"
            )

    def test_builder_skips_initializer_input(self) -> None:
        # ``shape`` is declared both as an initializer and as a graph input.
        graph = _graph(
            nodes=[_node("Reshape", ["X", "shape"], ["Y"])],
            inputs=[_vi("X", dims=(2, 2)), _vi("shape", elem_type=7, dims=(2,))],
            outputs=[_vi("Y", dims=(None,))],
            initializers=[_tensor(np.array([-1, 1], dtype=np.int64), "shape")],
            name="g",
        )
        code = translate(_model(graph), api="builder")
        self.assertNotIn("g.inp('shape'", code)
        self.assertIn("g.init(", code)

    def test_builder_preserves_non_identifier_attribute_names(self) -> None:
        graph = _graph(
            nodes=[
                _node(
                    "CustomOp", ["X"], ["Y"], attributes=[_attr("not-an-identifier", type=2, i=1)]
                )
            ],
            inputs=[_vi("X", dims=(1,))],
            outputs=[_vi("Y", dims=(1,))],
        )
        code = translate(
            _model(graph, opsets=[SimpleNamespace(domain="custom", version=1)]), api="builder"
        )
        self.assertIn("g.op.CustomOp('X', outputs=['Y'], **{'not-an-identifier': 1})", code)

    def test_builder_preserves_non_identifier_operator_names(self) -> None:
        graph = _graph(
            nodes=[_node("Custom-Op", ["X"], ["Y"])],
            inputs=[_vi("X", dims=(1,))],
            outputs=[_vi("Y", dims=(1,))],
        )
        code = translate(
            _model(graph, opsets=[SimpleNamespace(domain="custom", version=1)]), api="builder"
        )
        self.assertIn("getattr(g.op, 'Custom-Op')('X', outputs=['Y'])", code)

    def test_graph_input(self) -> None:
        # A bare GraphProto is accepted (no model wrapper).
        graph = _graph(
            nodes=[_node("Identity", ["X"], ["Y"])],
            inputs=[_vi("X", dims=("N",))],
            outputs=[_vi("Y", dims=("N",))],
            name="idg",
        )
        code = translate(graph, api="onnx-compact")
        self.assertIn("oh.make_node('Identity', ['X'], ['Y'])", code)
        self.assertIn("'idg'", code)


class TestTranslateRoundTrip(ExtTestCase):
    """Round-trip tests: execute the generated code and compare the models."""

    def _build_reference(self):
        import onnx_light.onnx.helper as oh
        import onnx_light.onnx.numpy_helper as onh
        from onnx_light.onnx import TensorProto

        node1 = oh.make_node("Add", ["X", "A"], ["T"])
        node2 = oh.make_node("Transpose", ["T"], ["Y"], perm=[1, 0])
        graph = oh.make_graph(
            [node1, node2],
            "roundtrip",
            [oh.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 2])],
            [onh.from_array(np.arange(6).reshape(2, 3).astype(np.float32), name="A")],
        )
        return oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)], ir_version=9)

    def _exec(self, code: str):
        namespace: dict = {}
        exec(code, namespace)  # noqa: S102
        return namespace["model"]

    def test_roundtrip_compact(self) -> None:
        model = self._build_reference()
        code = translate_header("onnx-compact") + translate(model, api="onnx-compact")
        rebuilt = self._exec(code)
        self.assertEqual(len(rebuilt.graph.node), 2)
        self.assertEqual([n.op_type for n in rebuilt.graph.node], ["Add", "Transpose"])
        self.assertEqual([o.name for o in rebuilt.graph.output], ["Y"])
        self.assertEqual(len(rebuilt.graph.initializer), 1)

    def test_roundtrip_builder(self) -> None:
        model = self._build_reference()
        code = translate_header("builder") + translate(model, api="builder")
        rebuilt = self._exec(code)
        self.assertEqual([n.op_type for n in rebuilt.graph.node], ["Add", "Transpose"])
        self.assertEqual([i.name for i in rebuilt.graph.input], ["X"])
        self.assertEqual([o.name for o in rebuilt.graph.output], ["Y"])
        self.assertEqual(len(rebuilt.graph.initializer), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
