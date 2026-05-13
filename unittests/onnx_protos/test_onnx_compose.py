# source: https://github.com/onnx/onnx/blob/main/onnx/test/compose_test.py
from __future__ import annotations

import unittest
from typing import TYPE_CHECKING

import numpy as np

import onnx_light.onnx as onnxl
import onnx_light.onnx.compose as compose
import onnx_light.onnx.helper as oh
import onnx_light.onnx.pychecker as pychecker

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def _make_model_m1() -> onnxl.ModelProto:
    """Builds the equivalent of M1_DEF: agraph (A0, A1, _A) => (B00, B10, B20).

    ``_A`` is an intentionally unused input present in the original M1_DEF.
    """
    A0 = oh.make_tensor_value_info("A0", onnxl.TensorProto.FLOAT, ["N", "M"])
    A1 = oh.make_tensor_value_info("A1", onnxl.TensorProto.FLOAT, ["N", "M"])
    _A = oh.make_tensor_value_info("_A", onnxl.TensorProto.FLOAT, ["N", "M"])
    B00 = oh.make_tensor_value_info("B00", onnxl.TensorProto.FLOAT, ["N", "M"])
    B10 = oh.make_tensor_value_info("B10", onnxl.TensorProto.FLOAT, ["N", "M"])
    B20 = oh.make_tensor_value_info("B20", onnxl.TensorProto.FLOAT, ["N", "M"])
    nodes = [
        oh.make_node("Add", ["A0", "A1"], ["B00"]),
        oh.make_node("Sub", ["A0", "A1"], ["B10"]),
        oh.make_node("Mul", ["A0", "A1"], ["B20"]),
    ]
    graph = oh.make_graph(nodes, "agraph", [A0, A1, _A], [B00, B10, B20])
    ops = [oh.make_opsetid("", 10), oh.make_opsetid("com.microsoft", 1)]
    return oh.make_model(graph, ir_version=7, opset_imports=ops)


def _make_model_m2() -> onnxl.ModelProto:
    """Builds the equivalent of M2_DEF: agraph (B01, B11, B21) => (D0,)."""
    B01 = oh.make_tensor_value_info("B01", onnxl.TensorProto.FLOAT, ["N", "M"])
    B11 = oh.make_tensor_value_info("B11", onnxl.TensorProto.FLOAT, ["N", "M"])
    B21 = oh.make_tensor_value_info("B21", onnxl.TensorProto.FLOAT, ["N", "M"])
    D0 = oh.make_tensor_value_info("D0", onnxl.TensorProto.FLOAT, ["N", "M"])
    nodes = [
        oh.make_node("Add", ["B01", "B11"], ["C0"]),
        oh.make_node("Sub", ["B11", "B21"], ["C1"]),
        oh.make_node("Mul", ["C0", "C1"], ["D0"]),
    ]
    graph = oh.make_graph(nodes, "agraph", [B01, B11, B21], [D0])
    ops = [oh.make_opsetid("", 10), oh.make_opsetid("com.microsoft", 1)]
    return oh.make_model(graph, ir_version=7, opset_imports=ops)


def _prefixed(prefix: str, s: str) -> str:
    """Prefixes a string (if not empty)."""
    return prefix + s if len(s) > 0 else s


def _get_shape(value_info: onnxl.ValueInfoProto) -> list[int]:
    """Returns a list of integers representing the shape of the ValueInfoProto."""
    return [
        value_info.type.tensor_type.shape.dim[d].dim_value
        for d in range(len(value_info.type.tensor_type.shape.dim))
    ]


def _make_sparse_tensor(name: str) -> onnxl.SparseTensorProto:
    dense_shape = [3, 3]
    linear_indices = [2, 3, 5]
    sparse_values = [1.7, 0.4, 0.9]
    values_tensor = oh.make_tensor(
        name=name + "_values",
        data_type=onnxl.TensorProto.FLOAT,
        dims=[len(sparse_values)],
        vals=np.array(sparse_values).astype(np.float32),
        raw=False,
    )
    indices_tensor = oh.make_tensor(
        name=name + "_idx",
        data_type=onnxl.TensorProto.INT64,
        dims=[len(linear_indices)],
        vals=np.array(linear_indices).astype(np.int64),
        raw=False,
    )
    return oh.make_sparse_tensor(values_tensor, indices_tensor, dense_shape)


class TestComposeFunctions(unittest.TestCase):
    def _test_merge_models(
        self,
        m1: onnxl.ModelProto,
        m2: onnxl.ModelProto,
        io_map: list[tuple[str, str]],
        check_expectations: Callable[
            [onnxl.GraphProto, onnxl.GraphProto, onnxl.GraphProto], None
        ],
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        prefix1: str | None = None,
        prefix2: str | None = None,
    ) -> None:
        g3 = compose.merge_graphs(
            m1.graph,
            m2.graph,
            io_map=io_map,
            inputs=inputs,
            outputs=outputs,
            prefix1=prefix1,
            prefix2=prefix2,
        )
        pychecker.check_graph(g3)
        check_expectations(m1.graph, m2.graph, g3)
        m3 = compose.merge_models(
            m1,
            m2,
            io_map=io_map,
            inputs=inputs,
            outputs=outputs,
            prefix1=prefix1,
            prefix2=prefix2,
        )
        pychecker.check_model(m3)
        check_expectations(m1.graph, m2.graph, m3.graph)

    def test_case_connect_all_no_name_collision(self) -> None:
        """Tests merging two models without overlapping names connecting all outputs to inputs."""

        def check_expectations(
            g1: onnxl.GraphProto, g2: onnxl.GraphProto, g3: onnxl.GraphProto
        ) -> None:
            self.assertEqual(list(g3.input), list(g1.input))
            self.assertEqual(list(g3.output), list(g2.output))
            self.assertEqual(
                ["Add", "Sub", "Mul", "Add", "Sub", "Mul"], [item.op_type for item in g3.node]
            )

        io_map = [("B00", "B01"), ("B10", "B11"), ("B20", "B21")]
        m1, m2 = _make_model_m1(), _make_model_m2()
        self._test_merge_models(m1, m2, io_map, check_expectations)

    def test_case_connect_same_output_twice(self) -> None:
        """Tests merging by connecting one output to all inputs of the second model."""

        def check_expectations(
            g1: onnxl.GraphProto, g2: onnxl.GraphProto, g3: onnxl.GraphProto
        ) -> None:
            del g2  # Unused
            self.assertEqual(list(g3.input), list(g1.input))
            self.assertEqual(["B10", "B20", "D0"], [elem.name for elem in g3.output])
            self.assertEqual(
                ["Add", "Sub", "Mul", "Add", "Sub", "Mul"], [item.op_type for item in g3.node]
            )

        io_map = [("B00", "B01"), ("B00", "B11"), ("B00", "B21")]
        m1, m2 = _make_model_m1(), _make_model_m2()
        self._test_merge_models(m1, m2, io_map, check_expectations)

    def test_case_connect_same_output_drop_outputs(self) -> None:
        """Tests merging while dropping outputs not in the outputs list."""

        def check_expectations(
            g1: onnxl.GraphProto, g2: onnxl.GraphProto, g3: onnxl.GraphProto
        ) -> None:
            del g2  # Unused
            self.assertEqual(list(g3.input), list(g1.input))
            self.assertEqual(["D0"], [elem.name for elem in g3.output])
            self.assertEqual(["Add", "Add", "Sub", "Mul"], [item.op_type for item in g3.node])

        io_map = [("B00", "B01"), ("B00", "B11"), ("B00", "B21")]
        outputs = ["D0"]
        m1, m2 = _make_model_m1(), _make_model_m2()
        self._test_merge_models(m1, m2, io_map, check_expectations, outputs=outputs)

    def test_case_connect_same_input_output_name(self) -> None:
        """Tests merging when the connected input/output share the same name."""
        A = oh.make_tensor_value_info("A", onnxl.TensorProto.FLOAT, ["N", "M"])
        B_in = oh.make_tensor_value_info("B", onnxl.TensorProto.FLOAT, ["N", "M"])
        B_out = oh.make_tensor_value_info("B", onnxl.TensorProto.FLOAT, ["N", "M"])
        C = oh.make_tensor_value_info("C", onnxl.TensorProto.FLOAT, ["N", "M"])
        ops = [oh.make_opsetid("", 10)]
        m1 = oh.make_model(
            oh.make_graph([oh.make_node("Add", ["A", "A"], ["B"])], "agraph", [A], [B_out]),
            ir_version=7,
            opset_imports=ops,
        )
        m2 = oh.make_model(
            oh.make_graph([oh.make_node("Add", ["B", "B"], ["C"])], "agraph", [B_in], [C]),
            ir_version=7,
            opset_imports=ops,
        )
        io_map = [("B", "B")]

        def check_expectations(
            g1: onnxl.GraphProto, g2: onnxl.GraphProto, g3: onnxl.GraphProto
        ) -> None:
            del g1, g2  # Unused
            self.assertEqual(["A"], [elem.name for elem in g3.input])
            self.assertEqual(["C"], [elem.name for elem in g3.output])

        self._test_merge_models(m1, m2, io_map, check_expectations)

    def test_case_drop_inputs_outputs(self) -> None:
        """Tests merging while not including some of the inputs/outputs."""
        ops = [oh.make_opsetid("", 10)]
        A0 = oh.make_tensor_value_info("A0", onnxl.TensorProto.FLOAT, ["N"])
        B0 = oh.make_tensor_value_info("B0", onnxl.TensorProto.FLOAT, ["N"])
        A1 = oh.make_tensor_value_info("A1", onnxl.TensorProto.FLOAT, ["N"])
        B1 = oh.make_tensor_value_info("B1", onnxl.TensorProto.FLOAT, ["N"])
        m1 = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["A0", "A0"], ["A1"]),
                    oh.make_node("Sub", ["B0", "B0"], ["B1"]),
                ],
                "agraph",
                [A0, B0],
                [A1, B1],
            ),
            ir_version=7,
            opset_imports=ops,
        )
        A2 = oh.make_tensor_value_info("A2", onnxl.TensorProto.FLOAT, ["N"])
        B2 = oh.make_tensor_value_info("B2", onnxl.TensorProto.FLOAT, ["N"])
        A3 = oh.make_tensor_value_info("A3", onnxl.TensorProto.FLOAT, ["N"])
        B3 = oh.make_tensor_value_info("B3", onnxl.TensorProto.FLOAT, ["N"])
        m2 = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["A2", "A2"], ["A3"]),
                    oh.make_node("Sub", ["B2", "B2"], ["B3"]),
                ],
                "agraph",
                [A2, B2],
                [A3, B3],
            ),
            ir_version=7,
            opset_imports=ops,
        )
        io_map = [("A1", "B2")]

        def check_expectations(
            g1: onnxl.GraphProto, g2: onnxl.GraphProto, g3: onnxl.GraphProto
        ) -> None:
            del g1, g2  # Unused
            self.assertEqual(["A0"], [elem.name for elem in g3.input])
            self.assertEqual(["B3"], [elem.name for elem in g3.output])
            self.assertEqual(["Add", "Sub"], [elem.op_type for elem in g3.node])

        self._test_merge_models(m1, m2, io_map, check_expectations, inputs=["A0"], outputs=["B3"])

    def test_case_name_collision_prefix(self) -> None:
        """Tests merging models with name collisions avoided via prefixes."""
        ops = [oh.make_opsetid("", 10)]
        A = oh.make_tensor_value_info("A", onnxl.TensorProto.FLOAT, ["N"])
        B = oh.make_tensor_value_info("B", onnxl.TensorProto.FLOAT, ["N"])
        C = oh.make_tensor_value_info("C", onnxl.TensorProto.FLOAT, ["N"])
        m1 = oh.make_model(
            oh.make_graph([oh.make_node("Add", ["A", "B"], ["C"])], "agraph", [A, B], [C]),
            ir_version=7,
            opset_imports=ops,
        )
        io_map = [("C", "A")]

        def check_expectations(
            g1: onnxl.GraphProto, g2: onnxl.GraphProto, g3: onnxl.GraphProto
        ) -> None:
            del g1, g2  # Unused
            self.assertEqual(["m1/A", "m1/B", "m2/B"], [elem.name for elem in g3.input])
            self.assertEqual(["m2/C"], [elem.name for elem in g3.output])
            self.assertEqual(["Add", "Add"], [item.op_type for item in g3.node])

        self._test_merge_models(m1, m1, io_map, check_expectations, prefix1="m1/", prefix2="m2/")

    def test_case_connect_partially_no_name_collision(self) -> None:
        """Tests partial connection: remaining inputs/outputs appear in the combined graph."""

        def check_expectations(
            g1: onnxl.GraphProto, g2: onnxl.GraphProto, g4: onnxl.GraphProto
        ) -> None:
            del g1, g2  # Unused
            self.assertEqual(["A0", "A1", "_A", "B21"], [elem.name for elem in g4.input])
            self.assertEqual(["B20", "D0"], [elem.name for elem in g4.output])

        io_map = [("B00", "B01"), ("B10", "B11")]
        m1, m2 = _make_model_m1(), _make_model_m2()
        self._test_merge_models(m1, m2, io_map, check_expectations)

    def test_merge_models_with_metadata_props(self) -> None:
        m1, m2 = _make_model_m1(), _make_model_m2()
        oh.set_model_props(m1, {"p1": "v1", "p2": "v2"})
        oh.set_model_props(m2, {"p3": "v3", "p4": "v4"})

        io_map = [("B00", "B01")]
        m3 = compose.merge_models(m1, m2, io_map=io_map)
        assert len(m3.metadata_props) == 4

        # Overlap with same value — fine
        oh.set_model_props(m2, {"p1": "v1", "p4": "v4"})
        m3 = compose.merge_models(m1, m2, io_map=io_map)
        assert len(m3.metadata_props) == 3

        # Same key but different value — error
        oh.set_model_props(m2, {"p1": "v5", "p4": "v4"})
        self.assertRaises(ValueError, compose.merge_models, m1, m2, io_map=io_map)

    def test_error_wrong_input_output_name(self) -> None:
        """Tests that a non-existing output/input name in io_map raises ValueError."""
        m1, m2 = _make_model_m1(), _make_model_m2()

        self.assertRaises(
            ValueError,
            compose.merge_models,
            m1,
            m2,
            io_map=[("wrong_outname", "B01"), ("B10", "B11"), ("B20", "B21")],
        )

        self.assertRaises(
            ValueError,
            compose.merge_models,
            m1,
            m2,
            io_map=[("B00", "wrong_input"), ("B10", "B11"), ("B20", "B21")],
        )

    def test_error_ir_version_mismatch(self) -> None:
        ops = [oh.make_opsetid("", 13)]
        X0 = oh.make_tensor_value_info("X0", onnxl.TensorProto.FLOAT, ["N", "M"])
        Y0 = oh.make_tensor_value_info("Y0", onnxl.TensorProto.FLOAT, ["N", "M"])
        X1 = oh.make_tensor_value_info("X1", onnxl.TensorProto.FLOAT, ["N", "M"])
        Y1 = oh.make_tensor_value_info("Y1", onnxl.TensorProto.FLOAT, ["N", "M"])
        m1 = oh.make_model(
            oh.make_graph([oh.make_node("Add", ["X0", "X0"], ["Y0"])], "g", [X0], [Y0]),
            ir_version=7,
            opset_imports=ops,
        )
        m2 = oh.make_model(
            oh.make_graph([oh.make_node("Add", ["X1", "X1"], ["Y1"])], "g", [X1], [Y1]),
            ir_version=6,
            opset_imports=ops,
        )
        self.assertRaises(ValueError, compose.merge_models, m1, m2, io_map=[("Y0", "X1")])

    def test_error_opset_import_mismatch(self) -> None:
        """Tests that models with different operator set versions raise ValueError."""
        m1, m2 = _make_model_m1(), _make_model_m2()
        ops10 = [oh.make_opsetid("", 10)]
        ops15 = [oh.make_opsetid("", 15)]
        m1 = oh.make_model(m1.graph, ir_version=7, opset_imports=ops10)
        m2 = oh.make_model(m2.graph, ir_version=7, opset_imports=ops15)
        io_map = [("B00", "B01"), ("B10", "B11"), ("B20", "B21")]
        self.assertRaises(ValueError, compose.merge_models, m1, m2, io_map)

    def test_add_prefix_to_inputs_outputs(self) -> None:
        """Tests prefixing inputs and outputs nodes."""
        X = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, ["N", 128])
        W = oh.make_tensor_value_info("W", onnxl.TensorProto.FLOAT, [128, 10])
        B = oh.make_tensor_value_info("B", onnxl.TensorProto.FLOAT, [10])
        C = oh.make_tensor_value_info("C", onnxl.TensorProto.FLOAT, ["N", 10])
        ops = [oh.make_opsetid("", 13)]
        input_model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("MatMul", ["X", "W"], ["T"]),
                    oh.make_node("Add", ["T", "B"], ["S"]),
                    oh.make_node("Softmax", ["S"], ["C"]),
                ],
                "agraph",
                [X, W, B],
                [C],
            ),
            ir_version=6,
            opset_imports=ops,
        )
        prefix = "pre_"
        prefixed_model = compose.add_prefix(
            model=input_model,
            prefix=prefix,
            rename_inputs=True,
            rename_outputs=True,
            rename_edges=True,
        )

        pychecker.check_graph(prefixed_model.graph)

        for i in prefixed_model.graph.input:
            self.assertTrue(i.name.startswith(prefix))
        for o in prefixed_model.graph.output:
            self.assertTrue(o.name.startswith(prefix))

    def test_add_prefix_wo_inputs_outputs(self) -> None:
        """Tests prefixing when renaming of inputs/outputs is deactivated."""
        A = oh.make_tensor_value_info("A", onnxl.TensorProto.FLOAT, [2, 2])
        B = oh.make_tensor_value_info("B", onnxl.TensorProto.FLOAT, [2, 2])
        ops = [oh.make_opsetid("", 13)]
        input_model = oh.make_model(
            oh.make_graph([oh.make_node("Identity", ["A"], ["B"])], "agraph", [A], [B]),
            ir_version=6,
            opset_imports=ops,
        )
        prefix = "pre_"
        prefixed_model = compose.add_prefix(
            model=input_model,
            prefix=prefix,
            rename_inputs=False,
            rename_outputs=False,
            rename_edges=True,
        )

        pychecker.check_graph(prefixed_model.graph)
        self.assertEqual(list(input_model.graph.input), list(prefixed_model.graph.input))
        self.assertEqual(list(input_model.graph.output), list(prefixed_model.graph.output))

    def _test_add_prefix(
        self,
        rename_nodes: bool = False,
        rename_edges: bool = False,
        rename_inputs: bool = False,
        rename_outputs: bool = False,
        rename_initializers: bool = False,
        rename_value_infos: bool = False,
        inplace: bool = False,
    ) -> None:
        m1 = _make_model_m1()
        prefix = "pre/"

        if inplace:
            m2 = onnxl.ModelProto()
            m2.CopyFrom(m1)
            compose.add_prefix(
                m2,
                prefix,
                rename_nodes=rename_nodes,
                rename_edges=rename_edges,
                rename_inputs=rename_inputs,
                rename_outputs=rename_outputs,
                rename_initializers=rename_initializers,
                rename_value_infos=rename_value_infos,
                inplace=True,
            )
        else:
            m2 = compose.add_prefix(
                m1,
                prefix,
                rename_nodes=rename_nodes,
                rename_edges=rename_edges,
                rename_inputs=rename_inputs,
                rename_outputs=rename_outputs,
                rename_initializers=rename_initializers,
                rename_value_infos=rename_value_infos,
            )
        g_in = m1.graph
        g_out = m2.graph

        if (
            rename_edges
            or rename_inputs
            or rename_outputs
            or rename_initializers
            or rename_value_infos
        ):
            name_mapping: dict[str, str] = {}

            if rename_edges:
                for n in g_in.node:
                    for e in n.input:
                        name_mapping[e] = _prefixed(prefix, e)
                    for e in n.output:
                        name_mapping[e] = _prefixed(prefix, e)
            if rename_inputs:
                for elem in g_in.input:
                    name_mapping[elem.name] = _prefixed(prefix, elem.name)
            if rename_outputs:
                for elem in g_in.output:
                    name_mapping[elem.name] = _prefixed(prefix, elem.name)

            if rename_initializers:
                for init in g_in.initializer:
                    name_mapping[init.name] = _prefixed(prefix, init.name)
                for sparse_init in g_in.sparse_initializer:
                    name_mapping[sparse_init.values.name] = _prefixed(
                        prefix, sparse_init.values.name
                    )
                    name_mapping[sparse_init.indices.name] = _prefixed(
                        prefix, sparse_init.indices.name
                    )

            if rename_value_infos:
                for value_info in g_in.output:
                    name_mapping[value_info.name] = _prefixed(prefix, value_info.name)

            for n1, n0 in zip(g_out.node, g_in.node, strict=True):
                for e1, e0 in zip(n1.input, n0.input, strict=True):
                    self.assertEqual(name_mapping.get(e0, e0), e1)
                for e1, e0 in zip(n1.output, n0.output, strict=True):
                    self.assertEqual(name_mapping.get(e0, e0), e1)
            for i1, i0 in zip(g_out.input, g_in.input, strict=True):
                self.assertEqual(name_mapping.get(i0.name, i0.name), i1.name)
            for o1, o0 in zip(g_out.output, g_in.output, strict=True):
                self.assertEqual(name_mapping.get(o0.name, o0.name), o1.name)

            for init1, init0 in zip(g_out.initializer, g_in.initializer, strict=True):
                self.assertEqual(name_mapping.get(init0.name, init0.name), init1.name)

            for sparse_init1, sparse_init0 in zip(
                g_out.sparse_initializer, g_in.sparse_initializer, strict=True
            ):
                self.assertEqual(
                    name_mapping.get(sparse_init0.values.name, sparse_init0.values.name),
                    sparse_init1.values.name,
                )
                self.assertEqual(
                    name_mapping.get(sparse_init0.indices.name, sparse_init0.indices.name),
                    sparse_init1.indices.name,
                )

            for vi1, vi0 in zip(g_out.value_info, g_in.value_info, strict=True):
                self.assertEqual(name_mapping.get(vi0.name, vi0.name), vi1.name)

            if rename_nodes:
                for n1, n0 in zip(g_out.node, g_in.node, strict=True):
                    self.assertEqual(_prefixed(prefix, n0.name), n1.name)

    def test_add_prefix_nodes(self) -> None:
        """Tests renaming nodes only."""
        self._test_add_prefix(rename_nodes=True)

    def test_add_prefix_edges(self) -> None:
        """Tests prefixing node edges (also renames inputs/outputs sharing those names)."""
        self._test_add_prefix(rename_edges=True, rename_inputs=True, rename_outputs=True)

    def test_add_prefix_inputs(self) -> None:
        """Tests prefixing graph inputs only."""
        self._test_add_prefix(rename_inputs=True)

    def test_add_prefix_outputs(self) -> None:
        """Tests prefixing graph outputs only."""
        self._test_add_prefix(rename_outputs=True)

    def test_add_prefix_attribute_subgraph(self) -> None:
        """Tests prefixing attribute subgraphs."""
        C = oh.make_tensor_value_info("C", onnxl.TensorProto.BOOL, [1])
        X = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [None, 1])
        Y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [None, 1])
        Z = oh.make_tensor_value_info("Z", onnxl.TensorProto.FLOAT, [None, 1])
        Out = oh.make_tensor_value_info("Out", onnxl.TensorProto.FLOAT, [None, 1])

        XY = oh.make_node("Mul", inputs=["X", "Y"], outputs=["XY"])
        add = oh.make_node("Add", inputs=["XY", "Z"], outputs=["Out"])
        sub = oh.make_node("Sub", inputs=["XY", "Z"], outputs=["Out"])

        cond = oh.make_node(
            "If",
            inputs=["C"],
            outputs=["Out"],
            then_branch=oh.make_graph(nodes=[add], name="then", inputs=[], outputs=[Out]),
            else_branch=oh.make_graph(nodes=[sub], name="else", inputs=[], outputs=[Out]),
        )
        graph = oh.make_graph(nodes=[XY, cond], name="graph", inputs=[C, X, Y, Z], outputs=[Out])
        prefix = "prefix."
        prefixed_graph = compose.add_prefix_graph(graph, prefix)
        pychecker.check_graph(prefixed_graph)
        for n1, n0 in zip(prefixed_graph.node, graph.node, strict=True):
            self.assertEqual(_prefixed(prefix, n0.name), n1.name)
            for attribute1, attribute0 in zip(n1.attribute, n0.attribute, strict=True):
                if attribute1.has_g():
                    for subgraph_n1, subgraph_n0 in zip(
                        attribute1.g.node, attribute0.g.node, strict=True
                    ):
                        for input_n1, input_n0 in zip(
                            subgraph_n1.input, subgraph_n0.input, strict=True
                        ):
                            self.assertEqual(_prefixed(prefix, input_n0), input_n1)
                        for output_n1, output_n0 in zip(
                            subgraph_n1.output, subgraph_n0.output, strict=True
                        ):
                            self.assertEqual(_prefixed(prefix, output_n0), output_n1)

    def test_add_prefix_attribute_multiple_subgraphs(self) -> None:
        """Tests prefixing attribute repeated graphs field."""
        X = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [1])
        Out = oh.make_tensor_value_info("Out", onnxl.TensorProto.FLOAT, [1])

        g1 = oh.make_graph(
            [oh.make_node("Identity", ["X"], ["Out"], name="id1")], "g1", [X], [Out]
        )
        g2 = oh.make_graph(
            [oh.make_node("Identity", ["X"], ["Out"], name="id2")], "g2", [X], [Out]
        )
        node = oh.make_node("CustomOp", ["X"], ["Out"], name="custom", bodies=[g1, g2])
        graph = oh.make_graph([node], "graph", [X], [Out])

        prefix = "pfx."
        prefixed = compose.add_prefix_graph(graph, prefix)
        self.assertEqual(prefixed.node[0].name, _prefixed(prefix, "custom"))

        pfx_graphs = list(prefixed.node[0].attribute[0].graphs)
        self.assertEqual(len(pfx_graphs), 2)
        for name in ("id1", "id2"):
            self.assertTrue(
                any(n.name == _prefixed(prefix, name) for g in pfx_graphs for n in g.node)
            )

    def test_add_prefix_all(self) -> None:
        """Tests prefixing all names in the graph."""
        self._test_add_prefix(True, True, True, True, True, True)

    def test_add_prefix_inplace(self) -> None:
        """Tests prefixing inplace."""
        self._test_add_prefix(inplace=True)

    def test_expand_out_dim(self) -> None:
        """Tests expanding output dimensions."""
        m1 = _make_model_m1()

        def _check_model(m1: onnxl.ModelProto, m2: onnxl.ModelProto, dim_idx: int) -> None:
            for out_g2, out_g1 in zip(m2.graph.output, m1.graph.output, strict=True):
                self.assertEqual(out_g2.name, out_g1.name)
                self.assertEqual(
                    out_g2.type.tensor_type.elem_type, out_g1.type.tensor_type.elem_type
                )
                expected_out_shape = _get_shape(out_g1)
                expected_out_shape.insert(dim_idx, 1)
                self.assertEqual(_get_shape(out_g2), expected_out_shape)

        for dim_idx in [0, 2, -1, -3]:
            m2 = compose.expand_out_dim(m1, dim_idx)
            _check_model(m1, m2, dim_idx)

        # Test inplace
        m2 = onnxl.ModelProto()
        m2.CopyFrom(m1)
        dim_idx = 0
        compose.expand_out_dim(m2, dim_idx, inplace=True)
        _check_model(m1, m2, dim_idx)

    def _test_overlapping_names(
        self,
        inputs0: Sequence[str] = ("i0", "i1"),
        inputs1: Sequence[str] = ("i2", "i3"),
        outputs0: Sequence[str] = ("o0", "o1"),
        outputs1: Sequence[str] = ("o2", "o3"),
        value_info0: Sequence[str] = ("v0", "v1"),
        value_info1: Sequence[str] = ("v2", "v3"),
        initializer0: Sequence[str] = ("init0", "init1"),
        initializer1: Sequence[str] = ("init2", "init3"),
        sparse_initializer0: Sequence[str] = ("sparse_init0", "sparse_init1"),
        sparse_initializer1: Sequence[str] = ("sparse_init2", "sparse_init3"),
    ) -> None:
        n0 = [
            oh.make_node("Identity", inputs=[inputs0[i]], outputs=[outputs0[i]])
            for i in range(len(inputs0))
        ]
        i0 = [
            oh.make_tensor_value_info(inputs0[i], onnxl.TensorProto.FLOAT, [])
            for i in range(len(inputs0))
        ]
        o0 = [
            oh.make_tensor_value_info(outputs0[i], onnxl.TensorProto.FLOAT, [])
            for i in range(len(outputs0))
        ]
        vi0 = [
            oh.make_tensor_value_info(value_info0[i], onnxl.TensorProto.FLOAT, [])
            for i in range(len(value_info0))
        ]
        init0 = [
            oh.make_tensor(
                name=initializer0[i], data_type=onnxl.TensorProto.INT64, dims=(), vals=[1]
            )
            for i in range(len(initializer0))
        ]
        sparse_init0 = [
            _make_sparse_tensor(sparse_initializer0[i]) for i in range(len(sparse_initializer0))
        ]

        n1 = [
            oh.make_node("Identity", inputs=[inputs1[i]], outputs=[outputs1[i]])
            for i in range(len(inputs1))
        ]
        i1 = [
            oh.make_tensor_value_info(inputs1[i], onnxl.TensorProto.FLOAT, [])
            for i in range(len(inputs1))
        ]
        o1 = [
            oh.make_tensor_value_info(outputs1[i], onnxl.TensorProto.FLOAT, [])
            for i in range(len(outputs1))
        ]
        vi1 = [
            oh.make_tensor_value_info(value_info1[i], onnxl.TensorProto.FLOAT, [])
            for i in range(len(value_info1))
        ]
        init1 = [
            oh.make_tensor(
                name=initializer1[i], data_type=onnxl.TensorProto.INT64, dims=(), vals=[1]
            )
            for i in range(len(initializer1))
        ]
        sparse_init1 = [
            _make_sparse_tensor(sparse_initializer1[i]) for i in range(len(sparse_initializer1))
        ]

        ops = [oh.make_opsetid("", 10)]
        m0 = oh.make_model(
            oh.make_graph(
                nodes=n0,
                name="g0",
                inputs=i0,
                outputs=o0,
                value_info=vi0,
                initializer=init0,
                sparse_initializer=sparse_init0,
            ),
            opset_imports=ops,
        )
        m1 = oh.make_model(
            oh.make_graph(
                nodes=n1,
                name="g1",
                inputs=i1,
                outputs=o1,
                value_info=vi1,
                initializer=init1,
                sparse_initializer=sparse_init1,
            ),
            opset_imports=ops,
        )

        overlap = compose.check_overlapping_names(m0.graph, m1.graph)
        idx = 0

        overlapping_inputs = list(set(inputs0) & set(inputs1))
        overlapping_outputs = list(set(outputs0) & set(outputs1))
        overlapping_edges = list(set(overlapping_inputs + overlapping_outputs))
        if overlapping_edges:
            self.assertEqual(overlap[idx], ("edge", overlapping_edges))
            idx += 1

        overlapping_vis = list(set(value_info0) & set(value_info1))
        if overlapping_vis:
            self.assertEqual(overlap[idx], ("value_info", overlapping_vis))
            idx += 1

        overlapping_init = list(set(initializer0) & set(initializer1))
        if overlapping_init:
            self.assertEqual(overlap[idx], ("initializer", overlapping_init))
            idx += 1

        overlapping_sparse_init = list(set(sparse_initializer0) & set(sparse_initializer1))
        if overlapping_sparse_init:
            expected_overlap = []
            for overlapping_name in overlapping_sparse_init:
                expected_overlap.append(overlapping_name + "_values")
                expected_overlap.append(overlapping_name + "_idx")
            self.assertEqual(overlap[idx], ("sparse_initializer", expected_overlap))
            idx += 1

        m0_new = compose.add_prefix(m0, prefix="g0/")
        overlap = compose.check_overlapping_names(m0_new.graph, m1.graph)
        self.assertEqual(0, len(overlap))

    def test_overlapping_input_names(self) -> None:
        """Tests error checking when input names overlap."""
        self._test_overlapping_names(inputs0=["i0", "i1"], inputs1=["i1", "i2"])

    def test_overlapping_output_names(self) -> None:
        """Tests error checking when output names overlap."""
        self._test_overlapping_names(outputs0=["o0", "o1"], outputs1=["o1", "o2"])

    def test_overlapping_value_info_names(self) -> None:
        """Tests error checking when value_info names overlap."""
        self._test_overlapping_names(value_info0=["vi0", "vi1"], value_info1=["vi1", "vi2"])

    def test_overlapping_initializer_names(self) -> None:
        """Tests error checking when initializer names overlap."""
        self._test_overlapping_names(
            initializer0=["init0", "init1"], initializer1=["init1", "init2"]
        )

    def test_overlapping_sparse_initializer_names(self) -> None:
        """Tests error checking when sparse_initializer names overlap."""
        self._test_overlapping_names(
            sparse_initializer0=["sparse_init0", "sparse_init1"],
            sparse_initializer1=["sparse_init1", "sparse_init2"],
        )

    def test_overlapping_function_names(self) -> None:
        """Tests error checking when local function names overlap."""
        ops = [oh.make_opsetid("", 10), oh.make_opsetid("local", 10)]

        def _make_function(
            domain: str,
            fname: str,
            inputs: list[str],
            outputs: list[str],
            nodes: list[onnxl.NodeProto],
        ) -> onnxl.FunctionProto:
            f = onnxl.FunctionProto()
            f.domain = domain
            f.name = fname
            f.input.extend(inputs)
            f.output.extend(outputs)
            f.node.extend(nodes)
            f.opset_import.extend(ops)
            return f

        g = onnxl.GraphProto()
        g.input.extend(
            [
                oh.make_tensor_value_info("x0", onnxl.TensorProto.FLOAT, []),
                oh.make_tensor_value_info("x1", onnxl.TensorProto.FLOAT, []),
            ]
        )
        g.output.extend([oh.make_tensor_value_info("y", onnxl.TensorProto.FLOAT, [])])
        g.node.extend([oh.make_node("f1", domain="local", inputs=["x0", "x1"], outputs=["y"])])

        g1 = onnxl.GraphProto()
        g1.CopyFrom(g)
        g1.name = "g1"
        m1 = oh.make_model(g1, opset_imports=ops)
        m1.functions.extend(
            [
                _make_function(
                    "local",
                    "f1",
                    ["x0", "x1"],
                    ["y"],
                    [oh.make_node("Add", inputs=["x0", "x1"], outputs=["y"])],
                )
            ]
        )

        g2 = onnxl.GraphProto()
        g2.CopyFrom(g)
        g2.name = "g2"
        m2 = oh.make_model(g2, opset_imports=ops)
        m2.functions.extend(
            [
                _make_function(
                    "local",
                    "f1",
                    ["x0", "x1"],
                    ["y"],
                    [oh.make_node("Mul", inputs=["x0", "x1"], outputs=["y"])],
                )
            ]
        )

        m = compose.merge_models(
            m1, m2, io_map=[("y", "x0"), ("y", "x1")], prefix1="m1/", prefix2="m2/"
        )
        pychecker.check_model(m)

        nodes = [n.op_type for n in m.graph.node]
        self.assertEqual(["m1/f1", "m2/f1"], nodes)

        functions = [f.name for f in m.functions]
        self.assertEqual(["m1/f1", "m2/f1"], functions)

        g3 = onnxl.GraphProto()
        g3.CopyFrom(g)
        g3.name = "g3"
        g3.node[0].op_type = "f2"
        m3 = oh.make_model(g3, opset_imports=ops)
        m3.functions.extend(
            [
                _make_function(
                    "local",
                    "f1",
                    ["x0", "x1"],
                    ["y"],
                    [
                        oh.make_node("Add", inputs=["x0", "x1"], outputs=["y0"]),
                        oh.make_node("Mul", inputs=["x0", "x1"], outputs=["y1"]),
                        oh.make_node("Add", inputs=["y0", "y1"], outputs=["y"]),
                    ],
                ),
                _make_function(
                    "local",
                    "f2",
                    ["x0", "x1"],
                    ["y"],
                    [
                        oh.make_node("f1", domain="local", inputs=["x0", "x1"], outputs=["y0"]),
                        oh.make_node("Mul", inputs=["x0", "x1"], outputs=["y1"]),
                        oh.make_node("Add", inputs=["y0", "y1"], outputs=["y"]),
                    ],
                ),
            ]
        )

        m = compose.merge_models(
            m1, m3, io_map=[("y", "x0"), ("y", "x1")], prefix1="m1/", prefix2="m3/"
        )
        pychecker.check_model(m)

        nodes = [n.op_type for n in m.graph.node]
        self.assertEqual(["m1/f1", "m3/f2"], nodes)

        functions = [f.name for f in m.functions]
        self.assertEqual(["m1/f1", "m3/f1", "m3/f2"], functions)

        self.assertEqual(["Add"], [n.op_type for n in m.functions[0].node])
        self.assertEqual(["Add", "Mul", "Add"], [n.op_type for n in m.functions[1].node])
        self.assertEqual(["m3/f1", "Mul", "Add"], [n.op_type for n in m.functions[2].node])

    def test_merge_drop_unnecessary_initializers_and_value_info(self) -> None:
        """Tests automatic removal of initializers when merging graphs."""
        ops = [oh.make_opsetid("", 10)]

        x_vi = oh.make_tensor_value_info("x", onnxl.TensorProto.FLOAT, [])
        y_vi = oh.make_tensor_value_info("y", onnxl.TensorProto.FLOAT, [])

        g1 = onnxl.GraphProto()
        g1.input.extend([x_vi])
        g1.output.extend([y_vi])
        g1.node.extend([oh.make_node("Identity", inputs=["x"], outputs=["y"])])
        g1.name = "g1"
        m1 = oh.make_model(g1, opset_imports=ops)

        g2 = onnxl.GraphProto()
        g2.CopyFrom(g1)
        g2.name = "g2"
        g2.initializer.extend(
            [oh.make_tensor(name="x", data_type=onnxl.TensorProto.FLOAT, dims=(), vals=[0])]
        )
        m2 = oh.make_model(g2, opset_imports=ops)

        g3 = onnxl.GraphProto()
        g3.CopyFrom(g1)
        g3.name = "g3"
        g3.sparse_initializer.extend([_make_sparse_tensor("x")])
        m3 = oh.make_model(g3, opset_imports=ops)

        g4 = onnxl.GraphProto()
        g4.CopyFrom(g1)
        g4.name = "g4"
        g4.value_info.extend([oh.make_tensor_value_info("x", onnxl.TensorProto.FLOAT, [])])
        m4 = oh.make_model(g4, opset_imports=ops)

        # Initializer 'x' from m2 is removed because the input no longer exists
        out_m1 = compose.merge_models(m1, m2, prefix1="m1/", io_map=[("y", "x")])
        self.assertEqual(0, len(out_m1.graph.initializer))

        # Sparse initializer 'x' from m3 is removed
        out_m2 = compose.merge_models(m1, m3, prefix1="m1/", io_map=[("y", "x")])
        self.assertEqual(0, len(out_m2.graph.initializer))

        # Value info 'x' from m4 is removed
        out_m3 = compose.merge_models(m1, m4, prefix1="m1/", io_map=[("y", "x")])
        value_info_names = {vi.name for vi in out_m3.graph.value_info}
        self.assertNotIn("x", value_info_names)
        self.assertIn("m1/y", value_info_names)

    def test_merge_preserves_mapped_output_value_info(self) -> None:
        """Mapped outputs should keep their type info as value_info in merged graph."""
        ops = [oh.make_opsetid("", 10)]
        X_vi = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, ["N", 32])
        Y_vi = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, ["N", 32])
        Z_vi = oh.make_tensor_value_info("Z", onnxl.TensorProto.FLOAT, ["N", 32])

        m1 = oh.make_model(
            oh.make_graph([oh.make_node("Identity", ["X"], ["Y"])], "g1", [X_vi], [Y_vi]),
            ir_version=7,
            opset_imports=ops,
        )
        m2 = oh.make_model(
            oh.make_graph([oh.make_node("Relu", ["X"], ["Z"])], "g2", [X_vi], [Z_vi]),
            ir_version=7,
            opset_imports=ops,
        )
        merged = compose.merge_models(m1, m2, io_map=[("Y", "X")])
        output_names = {o.name for o in merged.graph.output}
        value_info_by_name = {vi.name: vi for vi in merged.graph.value_info}

        self.assertNotIn("Y", output_names)
        self.assertIn("Y", value_info_by_name)
        self.assertEqual(
            m1.graph.output[0].SerializeToString(), value_info_by_name["Y"].SerializeToString()
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
