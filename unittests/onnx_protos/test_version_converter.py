# source: https://github.com/onnx/onnx/blob/main/onnx/test/version_converter_test.py
from __future__ import annotations

import struct
import unittest

import numpy as np

import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
import onnx_light.onnx.pychecker as pychecker
import onnx_light.onnx.version_converter as version_converter
from onnx_light.ext_test_case import ExtTestCase


def _register_onnx_schemas() -> None:
    """Registers all onnx operator schemas into onnx_light's schema registry.

    The version converter requires schemas to be populated in the registry to
    determine per-op version ranges and drive adapter selection.  Unlike the
    reference onnx library, onnx_light does not auto-register schemas via
    static initializers, so this helper bridges the two.

    Returns:
        None.
    """
    try:
        import onnx

        from onnx_light.onnx.defs import OpSchema, SchemaError, register_schema
    except ImportError:
        return
    for s in onnx.defs.get_all_schemas_with_history():
        try:
            register_schema(OpSchema(s.name, s.domain, s.since_version))
        except SchemaError:
            pass  # ignore duplicate-registration errors


class TestVersionConverter(ExtTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _register_onnx_schemas()

    def _converted(
        self,
        graph: onnxl.GraphProto,
        initial_version: onnxl.OperatorSetIdProto,
        target_version: int,
    ) -> onnxl.ModelProto:
        """Builds a model, converts it and checks the result."""
        orig_model = oh.make_model(
            graph, producer_name="onnx-test", opset_imports=[initial_version]
        )
        converted_model = version_converter.convert_version(orig_model, target_version)
        pychecker.check_model(converted_model)
        return converted_model

    # Test 1: Backwards Incompatible Conversion: Reshape: 8 -> 2
    def test_backwards_incompatible(self) -> None:
        def test() -> None:
            nodes = [
                oh.make_node("Add", ["W", "Z"], ["shape"]),
                oh.make_node("Reshape", ["X", "shape"], ["A"]),
                oh.make_node("Add", ["A", "W"], ["Y"]),
            ]
            graph = oh.make_graph(
                nodes,
                "test",
                [
                    oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, (5,)),
                    oh.make_tensor_value_info("W", onnxl.TensorProto.FLOAT, (1,)),
                    oh.make_tensor_value_info("Z", onnxl.TensorProto.FLOAT, (1,)),
                ],
                [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, (5,))],
            )
            self._converted(graph, oh.make_operatorsetid("", 8), 2)

        self.assertRaises(RuntimeError, test)

    # Test 2: Backwards Compatible Conversion (No Adaptations): Add: 3 -> 2
    def test_backwards_compatible(self) -> None:
        nodes = [oh.make_node("Add", ["X1", "X2"], ["Y"])]
        graph = oh.make_graph(
            nodes,
            "test",
            [
                oh.make_tensor_value_info("X1", onnxl.TensorProto.FLOAT, (5,)),
                oh.make_tensor_value_info("X2", onnxl.TensorProto.FLOAT, (5,)),
            ],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, (5,))],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 3), 2)
        assert converted_model.graph.node[0].op_type == "Add"
        assert converted_model.opset_import[0].version == 2

    # Test 3: Non-Existent Op Conversion: Cos: 8 -> 6
    def test_non_existent_op(self) -> None:
        def test() -> None:
            nodes = [oh.make_node("Cos", ["X"], ["Y"])]
            graph = oh.make_graph(
                nodes,
                "test",
                [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, (5,))],
                [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, (5,))],
            )
            self._converted(graph, oh.make_operatorsetid("", 8), 6)

        self.assertRaises(RuntimeError, test)

    # Test Add Adapter: 8 -> 5
    def test_add_8_5(self) -> None:
        nodes = [oh.make_node("Add", ["X1", "X2"], ["Y"])]
        graph = oh.make_graph(
            nodes,
            "test",
            [
                oh.make_tensor_value_info("X1", onnxl.TensorProto.FLOAT, (5,)),
                oh.make_tensor_value_info("X2", onnxl.TensorProto.FLOAT, (1,)),
            ],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, (5,))],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 8), 5)
        assert converted_model.graph.node[0].op_type == "Add"
        assert converted_model.opset_import[0].version == 5

    # Test Add Adapter: 5 -> 8
    def test_add_5_8(self) -> None:
        nodes = [oh.make_node("Add", ["X1", "X2"], ["Y"])]
        graph = oh.make_graph(
            nodes,
            "test",
            [
                oh.make_tensor_value_info("X1", onnxl.TensorProto.FLOAT, (5,)),
                oh.make_tensor_value_info("X2", onnxl.TensorProto.FLOAT, (1,)),
            ],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, (5,))],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 5), 8)
        assert converted_model.graph.node[0].op_type == "Add"
        assert converted_model.opset_import[0].version == 8

    # Test Add Adapter: 5 -> 8, requiring insertion of an Unsqueeze node
    def test_add_5_8_with_unsqueeze(self) -> None:
        nodes = [oh.make_node("Add", ["X1", "X2"], ["Y"], axis=0, broadcast=1)]
        graph = oh.make_graph(
            nodes,
            "test",
            [
                oh.make_tensor_value_info("X1", onnxl.TensorProto.FLOAT, (5, 2)),
                oh.make_tensor_value_info("X2", onnxl.TensorProto.FLOAT, (5,)),
            ],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, (5,))],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 5), 8)
        assert converted_model.graph.node[0].op_type == "Unsqueeze"
        assert converted_model.graph.node[1].op_type == "Add"
        assert converted_model.opset_import[0].version == 8

    # Test Mul Adapter: 8 -> 5
    def test_mul_8_5(self) -> None:
        nodes = [oh.make_node("Mul", ["X1", "X2"], ["Y"])]
        graph = oh.make_graph(
            nodes,
            "test",
            [
                oh.make_tensor_value_info("X1", onnxl.TensorProto.FLOAT, (5,)),
                oh.make_tensor_value_info("X2", onnxl.TensorProto.FLOAT, (1,)),
            ],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, (5,))],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 8), 5)
        assert converted_model.graph.node[0].op_type == "Mul"
        assert converted_model.opset_import[0].version == 5

    # Test Mul Adapter: 5 -> 8
    def test_mul_5_8(self) -> None:
        nodes = [oh.make_node("Mul", ["X1", "X2"], ["Y"])]
        graph = oh.make_graph(
            nodes,
            "test",
            [
                oh.make_tensor_value_info("X1", onnxl.TensorProto.FLOAT, (5,)),
                oh.make_tensor_value_info("X2", onnxl.TensorProto.FLOAT, (1,)),
            ],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, (5,))],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 5), 8)
        assert converted_model.graph.node[0].op_type == "Mul"
        assert converted_model.opset_import[0].version == 8

    # Test Gemm Adapter: 1 -> 8
    def test_gemm_up(self) -> None:
        nodes = [oh.make_node("Gemm", ["A", "B", "C"], ["Y"])]
        graph = oh.make_graph(
            nodes,
            "test",
            [
                oh.make_tensor_value_info("A", onnxl.TensorProto.FLOAT, (5, 5)),
                oh.make_tensor_value_info("B", onnxl.TensorProto.FLOAT, (5, 5)),
                oh.make_tensor_value_info("C", onnxl.TensorProto.FLOAT, (5, 5)),
            ],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, (5, 5))],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 1), 8)
        assert converted_model.graph.node[0].op_type == "Gemm"
        assert converted_model.opset_import[0].version == 8

    # Test Gemm Adapter: 8 -> 1
    def test_gemm_down(self) -> None:
        nodes = [oh.make_node("Gemm", ["A", "B", "C"], ["Y"])]
        graph = oh.make_graph(
            nodes,
            "test",
            [
                oh.make_tensor_value_info("A", onnxl.TensorProto.FLOAT, (5, 5)),
                oh.make_tensor_value_info("B", onnxl.TensorProto.FLOAT, (5, 5)),
                oh.make_tensor_value_info("C", onnxl.TensorProto.FLOAT, (5, 5)),
            ],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, (5, 5))],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 8), 1)
        assert converted_model.graph.node[0].op_type == "Gemm"
        assert converted_model.opset_import[0].version == 1

    def test_gemm_7_6_rejects_1d_input(self) -> None:
        # Regression test: heap-buffer-overflow when B has rank < 2
        def test() -> None:
            nodes = [oh.make_node("Gemm", ["A", "B", "C"], ["Y"])]
            graph = oh.make_graph(
                nodes,
                "test_gemm_7_6_rejects_1d_input",
                [
                    oh.make_tensor_value_info("A", onnxl.TensorProto.FLOAT, (4, 3)),
                    oh.make_tensor_value_info("B", onnxl.TensorProto.FLOAT, (28,)),
                    oh.make_tensor_value_info("C", onnxl.TensorProto.FLOAT, (4,)),
                ],
                [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, None)],
            )
            self._converted(graph, oh.make_operatorsetid("", 7), 6)

        self.assertRaises(RuntimeError, test)

    def test_gemm_6_7_rejects_1d_input(self) -> None:
        # Regression test: heap-buffer-overflow when B has rank < 2 (upward)
        def test() -> None:
            nodes = [oh.make_node("Gemm", ["A", "B", "C"], ["Y"])]
            graph = oh.make_graph(
                nodes,
                "test_gemm_6_7_rejects_1d_input",
                [
                    oh.make_tensor_value_info("A", onnxl.TensorProto.FLOAT, (4, 3)),
                    oh.make_tensor_value_info("B", onnxl.TensorProto.FLOAT, (28,)),
                    oh.make_tensor_value_info("C", onnxl.TensorProto.FLOAT, (4,)),
                ],
                [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, None)],
            )
            self._converted(graph, oh.make_operatorsetid("", 6), 7)

        self.assertRaises(RuntimeError, test)

    def test_gemm_7_6_rejects_1d_A(self) -> None:
        def test() -> None:
            nodes = [oh.make_node("Gemm", ["A", "B", "C"], ["Y"])]
            graph = oh.make_graph(
                nodes,
                "test_gemm_7_6_rejects_1d_A",
                [
                    oh.make_tensor_value_info("A", onnxl.TensorProto.FLOAT, (12,)),
                    oh.make_tensor_value_info("B", onnxl.TensorProto.FLOAT, (3, 4)),
                    oh.make_tensor_value_info("C", onnxl.TensorProto.FLOAT, (4,)),
                ],
                [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, None)],
            )
            self._converted(graph, oh.make_operatorsetid("", 7), 6)

        self.assertRaises(RuntimeError, test)

    def test_gemm_6_7_rejects_1d_A(self) -> None:
        def test() -> None:
            nodes = [oh.make_node("Gemm", ["A", "B", "C"], ["Y"])]
            graph = oh.make_graph(
                nodes,
                "test_gemm_6_7_rejects_1d_A",
                [
                    oh.make_tensor_value_info("A", onnxl.TensorProto.FLOAT, (12,)),
                    oh.make_tensor_value_info("B", onnxl.TensorProto.FLOAT, (3, 4)),
                    oh.make_tensor_value_info("C", onnxl.TensorProto.FLOAT, (4,)),
                ],
                [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, None)],
            )
            self._converted(graph, oh.make_operatorsetid("", 6), 7)

        self.assertRaises(RuntimeError, test)

    # Test Relu Adapter: 5 -> 7
    def test_relu_5_7(self) -> None:
        nodes = [oh.make_node("Relu", ["X"], ["Y"])]
        graph = oh.make_graph(
            nodes,
            "test",
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, (5,))],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, (5,))],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 5), 7)
        assert converted_model.graph.node[0].op_type == "Relu"
        assert converted_model.opset_import[0].version == 7

    # Test Relu Adapter: 7 -> 5
    def test_relu_7_5(self) -> None:
        nodes = [oh.make_node("Relu", ["X"], ["Y"])]
        graph = oh.make_graph(
            nodes,
            "test",
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, (5,))],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, (5,))],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 7), 5)
        assert converted_model.graph.node[0].op_type == "Relu"
        assert converted_model.opset_import[0].version == 5

    # Test BatchNormalization Adapter: 8 -> 5
    def test_batch_normalization_8_5(self) -> None:
        nodes = [oh.make_node("BatchNormalization", ["X", "scale", "B", "mean", "var"], ["Y"])]
        graph = oh.make_graph(
            nodes,
            "test",
            [
                oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, (5,)),
                oh.make_tensor_value_info("scale", onnxl.TensorProto.FLOAT, (1,)),
                oh.make_tensor_value_info("B", onnxl.TensorProto.FLOAT, (1,)),
                oh.make_tensor_value_info("mean", onnxl.TensorProto.FLOAT, (1,)),
                oh.make_tensor_value_info("var", onnxl.TensorProto.FLOAT, (1,)),
            ],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, (5,))],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 8), 5)
        assert converted_model.graph.node[0].op_type == "BatchNormalization"
        assert converted_model.opset_import[0].version == 5

    # Test BatchNormalization Adapter: 5 -> 8
    def test_batch_normalization_5_8(self) -> None:
        nodes = [oh.make_node("BatchNormalization", ["X", "scale", "B", "mean", "var"], ["Y"])]
        graph = oh.make_graph(
            nodes,
            "test",
            [
                oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, (5,)),
                oh.make_tensor_value_info("scale", onnxl.TensorProto.FLOAT, (1,)),
                oh.make_tensor_value_info("B", onnxl.TensorProto.FLOAT, (1,)),
                oh.make_tensor_value_info("mean", onnxl.TensorProto.FLOAT, (1,)),
                oh.make_tensor_value_info("var", onnxl.TensorProto.FLOAT, (1,)),
            ],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, (5,))],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 5), 8)
        assert converted_model.graph.node[0].op_type == "BatchNormalization"
        assert converted_model.opset_import[0].version == 8

    # Test Concat Adapter: 3 -> 5
    def test_concat_3_5(self) -> None:
        nodes = [oh.make_node("Concat", ["X1", "X2", "X3", "X4", "X5"], ["Y"])]
        graph = oh.make_graph(
            nodes,
            "test",
            [
                oh.make_tensor_value_info("X1", onnxl.TensorProto.FLOAT, (1,)),
                oh.make_tensor_value_info("X2", onnxl.TensorProto.FLOAT, (1,)),
                oh.make_tensor_value_info("X3", onnxl.TensorProto.FLOAT, (1,)),
                oh.make_tensor_value_info("X4", onnxl.TensorProto.FLOAT, (1,)),
                oh.make_tensor_value_info("X5", onnxl.TensorProto.FLOAT, (1,)),
            ],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, (5,))],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 3), 5)
        assert converted_model.graph.node[0].op_type == "Concat"
        assert converted_model.opset_import[0].version == 5

    # Test Concat Adapter: 5 -> 3
    def test_concat_5_3(self) -> None:
        nodes = [oh.make_node("Concat", ["X1", "X2", "X3", "X4", "X5"], ["Y"], axis=0)]
        graph = oh.make_graph(
            nodes,
            "test",
            [
                oh.make_tensor_value_info("X1", onnxl.TensorProto.FLOAT, (1,)),
                oh.make_tensor_value_info("X2", onnxl.TensorProto.FLOAT, (1,)),
                oh.make_tensor_value_info("X3", onnxl.TensorProto.FLOAT, (1,)),
                oh.make_tensor_value_info("X4", onnxl.TensorProto.FLOAT, (1,)),
                oh.make_tensor_value_info("X5", onnxl.TensorProto.FLOAT, (1,)),
            ],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, (5,))],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 5), 3)
        assert converted_model.graph.node[0].op_type == "Concat"
        assert converted_model.opset_import[0].version == 3

    # Test Reshape Adapter: 6 -> 4
    def test_reshape_6_4(self) -> None:
        nodes = [
            oh.make_node(
                "Constant",
                [],
                ["shape"],
                value=oh.make_tensor("", onnxl.TensorProto.INT64, [1], [5]),
            ),
            oh.make_node("Reshape", ["X", "shape"], ["Y"]),
        ]
        graph = oh.make_graph(
            nodes,
            "test",
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, (5,))],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, (5,))],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 6), 4)
        assert converted_model.graph.node[0].op_type == "Reshape"
        assert converted_model.opset_import[0].version == 4

    # Test Reshape Adapter: 4 -> 6
    def test_reshape_4_6(self) -> None:
        nodes = [oh.make_node("Reshape", ["X"], ["Y"], shape=[5])]
        graph = oh.make_graph(
            nodes,
            "test",
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, (5,))],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, (5,))],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 4), 6)
        assert converted_model.graph.node[0].op_type == "Constant"
        assert converted_model.graph.node[1].op_type == "Reshape"
        assert converted_model.opset_import[0].version == 6

    # Test Sum Adapter: 7 -> 8
    def test_sum_7_8(self) -> None:
        nodes = [oh.make_node("Sum", ["data_0", "data_1", "data_2", "data_3", "data_4"], ["sum"])]
        graph = oh.make_graph(
            nodes,
            "test",
            [
                oh.make_tensor_value_info("data_0", onnxl.TensorProto.FLOAT, (5,)),
                oh.make_tensor_value_info("data_1", onnxl.TensorProto.FLOAT, (5,)),
                oh.make_tensor_value_info("data_2", onnxl.TensorProto.FLOAT, (5,)),
                oh.make_tensor_value_info("data_3", onnxl.TensorProto.FLOAT, (5,)),
                oh.make_tensor_value_info("data_4", onnxl.TensorProto.FLOAT, (5,)),
            ],
            [oh.make_tensor_value_info("sum", onnxl.TensorProto.FLOAT, (5,))],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 7), 8)
        assert converted_model.graph.node[0].op_type == "Sum"
        assert converted_model.opset_import[0].version == 8

    # Test Sum Adapter: 5 -> 7
    def test_sum_5_8(self) -> None:
        nodes = [oh.make_node("Sum", ["data_0", "data_1", "data_2", "data_3", "data_4"], ["sum"])]
        graph = oh.make_graph(
            nodes,
            "test",
            [
                oh.make_tensor_value_info("data_0", onnxl.TensorProto.FLOAT, (5,)),
                oh.make_tensor_value_info("data_1", onnxl.TensorProto.FLOAT, (5,)),
                oh.make_tensor_value_info("data_2", onnxl.TensorProto.FLOAT, (5,)),
                oh.make_tensor_value_info("data_3", onnxl.TensorProto.FLOAT, (5,)),
                oh.make_tensor_value_info("data_4", onnxl.TensorProto.FLOAT, (5,)),
            ],
            [oh.make_tensor_value_info("sum", onnxl.TensorProto.FLOAT, (5,))],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 5), 7)
        assert converted_model.graph.node[0].op_type == "Sum"
        assert converted_model.opset_import[0].version == 7

    # Test Sum Adapter: 8 -> 5
    def test_sum_8_5(self) -> None:
        nodes = [oh.make_node("Sum", ["data_0", "data_1", "data_2", "data_3", "data_4"], ["sum"])]
        graph = oh.make_graph(
            nodes,
            "test",
            [
                oh.make_tensor_value_info("data_0", onnxl.TensorProto.FLOAT, (5,)),
                oh.make_tensor_value_info("data_1", onnxl.TensorProto.FLOAT, (5,)),
                oh.make_tensor_value_info("data_2", onnxl.TensorProto.FLOAT, (5,)),
                oh.make_tensor_value_info("data_3", onnxl.TensorProto.FLOAT, (5,)),
                oh.make_tensor_value_info("data_4", onnxl.TensorProto.FLOAT, (5,)),
            ],
            [oh.make_tensor_value_info("sum", onnxl.TensorProto.FLOAT, (5,))],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 8), 5)
        assert converted_model.graph.node[0].op_type == "Sum"
        assert converted_model.opset_import[0].version == 5

    # Test AveragePool Adapter: 1 -> 8
    def test_averagepool_up(self) -> None:
        nodes = [oh.make_node("AveragePool", ["X"], ["Y"], kernel_shape=[1, 1])]
        graph = oh.make_graph(
            nodes,
            "test",
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, (5, 5, 5, 5))],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, (5, 5, 5, 5))],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 1), 8)
        assert converted_model.graph.node[0].op_type == "AveragePool"
        assert converted_model.opset_import[0].version == 8

    # Test AveragePool Adapter: 8 -> 1
    def test_averagepool_down(self) -> None:
        nodes = [oh.make_node("AveragePool", ["X"], ["Y"], kernel_shape=[1, 1])]
        graph = oh.make_graph(
            nodes,
            "test",
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, (5, 5, 5, 5))],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, (5, 5, 5, 5))],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 8), 1)
        assert converted_model.graph.node[0].op_type == "AveragePool"
        assert converted_model.opset_import[0].version == 1

    # Test Dropout Adapter: 1 -> 8
    def test_dropout_up(self) -> None:
        nodes = [oh.make_node("Dropout", ["data"], ["output"], is_test=1)]
        graph = oh.make_graph(
            nodes,
            "test",
            [oh.make_tensor_value_info("data", onnxl.TensorProto.FLOAT, (5, 5))],
            [oh.make_tensor_value_info("output", onnxl.TensorProto.FLOAT, (5, 5))],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 1), 8)
        assert converted_model.graph.node[0].op_type == "Dropout"
        assert converted_model.opset_import[0].version == 8

    # Test Dropout Adapter: 8 -> 1
    def test_dropout_down(self) -> None:
        nodes = [oh.make_node("Dropout", ["data"], ["output"])]
        graph = oh.make_graph(
            nodes,
            "test",
            [oh.make_tensor_value_info("data", onnxl.TensorProto.FLOAT, (5, 5))],
            [oh.make_tensor_value_info("output", onnxl.TensorProto.FLOAT, (5, 5))],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 8), 1)
        assert converted_model.graph.node[0].op_type == "Dropout"
        assert converted_model.opset_import[0].version == 1

    # Test Max Adapter: 7 -> 8
    def test_max_7_8(self) -> None:
        from_opset = 7
        to_opset = 8
        data_type = onnxl.TensorProto.FLOAT
        data_shape = (2, 3, 4)

        nodes = [oh.make_node("Max", inputs=["X"], outputs=["Y"])]
        graph = oh.make_graph(
            nodes,
            "test_max",
            [oh.make_tensor_value_info("X", data_type, data_shape)],
            [oh.make_tensor_value_info("Y", data_type, data_shape)],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", from_opset), to_opset)
        assert converted_model.graph.node[0].op_type == "Max"
        assert converted_model.graph.output[0].type.tensor_type.elem_type == data_type
        assert converted_model.opset_import[0].version == to_opset

    # Test Min Adapter: 7 -> 8
    def test_min_7_8(self) -> None:
        from_opset = 7
        to_opset = 8
        data_type = onnxl.TensorProto.FLOAT
        data_shape = (2, 3, 4)

        nodes = [oh.make_node("Min", inputs=["X"], outputs=["Y"])]
        graph = oh.make_graph(
            nodes,
            "test_min",
            [oh.make_tensor_value_info("X", data_type, data_shape)],
            [oh.make_tensor_value_info("Y", data_type, data_shape)],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", from_opset), to_opset)
        assert converted_model.graph.node[0].op_type == "Min"
        assert converted_model.graph.output[0].type.tensor_type.elem_type == data_type
        assert converted_model.opset_import[0].version == to_opset

    # Test Mean Adapter: 7 -> 8
    def test_mean_7_8(self) -> None:
        from_opset = 7
        to_opset = 8
        data_type = onnxl.TensorProto.FLOAT
        data_shape = (3,)

        nodes = [oh.make_node("Mean", inputs=["X"], outputs=["Y"])]
        graph = oh.make_graph(
            nodes,
            "test_mean",
            [oh.make_tensor_value_info("X", data_type, data_shape)],
            [oh.make_tensor_value_info("Y", data_type, data_shape)],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", from_opset), to_opset)
        assert converted_model.graph.node[0].op_type == "Mean"
        assert converted_model.graph.output[0].type.tensor_type.elem_type == data_type
        assert converted_model.opset_import[0].version == to_opset

    # Test MaxPool Adapter: 1 -> 8
    def test_maxpool_up(self) -> None:
        nodes = [oh.make_node("MaxPool", ["X"], ["Y"], kernel_shape=[1, 1])]
        graph = oh.make_graph(
            nodes,
            "test",
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, (5, 5, 5, 5))],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, (5, 5, 5, 5))],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 1), 8)
        assert converted_model.graph.node[0].op_type == "MaxPool"
        assert converted_model.opset_import[0].version == 8

    # Test Upsample Adapter: 6 -> 7
    def test_upsample_6_7(self) -> None:
        from_opset = 6
        to_opset = 7
        data_type = onnxl.TensorProto.FLOAT

        nodes = [
            oh.make_node(
                "Upsample",
                inputs=["X"],
                outputs=["Y"],
                mode="nearest",
                width_scale=3.0,
                height_scale=2.0,
            )
        ]
        graph = oh.make_graph(
            nodes,
            "test_upsample_6_7",
            [oh.make_tensor_value_info("X", data_type, [1, 1, 2, 2])],
            [oh.make_tensor_value_info("Y", data_type, [1, 1, 4, 6])],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", from_opset), to_opset)
        assert len(converted_model.graph.node) == 1
        assert converted_model.graph.node[0].op_type == "Upsample"
        attribute_names = [attr.name for attr in converted_model.graph.node[0].attribute]
        assert "scales" in attribute_names
        assert "width_scale" not in attribute_names
        assert "height_scale" not in attribute_names
        assert converted_model.opset_import[0].version == to_opset

    # Test MaxPool Adapter: 8 -> 1
    def test_maxpool_down(self) -> None:
        nodes = [oh.make_node("MaxPool", ["X"], ["Y"], kernel_shape=[1, 1])]
        graph = oh.make_graph(
            nodes,
            "test",
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, (5, 5, 5, 5))],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, (5, 5, 5, 5))],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 8), 1)
        assert converted_model.graph.node[0].op_type == "MaxPool"
        assert converted_model.opset_import[0].version == 1

    # Test BatchNormalization Adapter: 8 -> 9
    def test_batch_normalization_8_9(self) -> None:
        from_opset = 8
        to_opset = 9
        data_type = onnxl.TensorProto.FLOAT
        input_shape = (1, 2, 1, 3)

        nodes = [
            oh.make_node(
                "BatchNormalization", inputs=["x", "s", "bias", "mean", "var"], outputs=["y"]
            )
        ]
        x = oh.make_tensor_value_info("x", data_type, input_shape)
        scale = oh.make_tensor_value_info("s", data_type, [input_shape[1]])
        B = oh.make_tensor_value_info("bias", data_type, [input_shape[1]])
        mean = oh.make_tensor_value_info("mean", data_type, [input_shape[1]])
        var = oh.make_tensor_value_info("var", data_type, [input_shape[1]])
        y = oh.make_tensor_value_info("y", data_type, input_shape)

        graph = oh.make_graph(nodes, "test_batchnormalization_8_9", [x, scale, B, mean, var], [y])
        converted_model = self._converted(graph, oh.make_operatorsetid("", from_opset), to_opset)
        assert converted_model.graph.node[0].op_type == "BatchNormalization"
        assert converted_model.opset_import[0].version == to_opset

    # Test BatchNormalization Adapter: 9 -> 8
    def test_batchnormalization_9_8(self) -> None:
        from_opset = 9
        to_opset = 8
        data_type = onnxl.TensorProto.FLOAT
        input_shape = (2, 3, 4, 5)

        nodes = [
            oh.make_node(
                "BatchNormalization", inputs=["X", "scale", "B", "mean", "var"], outputs=["Y"]
            )
        ]
        x = oh.make_tensor_value_info("X", data_type, input_shape)
        scale = oh.make_tensor_value_info("scale", data_type, [input_shape[1]])
        B = oh.make_tensor_value_info("B", data_type, [input_shape[1]])
        mean = oh.make_tensor_value_info("mean", data_type, [input_shape[1]])
        var = oh.make_tensor_value_info("var", data_type, [input_shape[1]])
        y = oh.make_tensor_value_info("Y", data_type, input_shape)

        graph = oh.make_graph(nodes, "test_batchnormalization", [x, scale, B, mean, var], [y])
        converted_model = self._converted(graph, oh.make_operatorsetid("", from_opset), to_opset)
        assert converted_model.graph.node[0].op_type == "BatchNormalization"
        assert converted_model.opset_import[0].version == to_opset

    # Test Constant Adapter: 8 -> 9
    def test_constant_8_9(self) -> None:
        from_opset = 8
        to_opset = 9
        data_type = onnxl.TensorProto.FLOAT
        output_shape = [2, 3, 4]
        output_value = np.arange(24)

        nodes = [
            oh.make_node(
                "Constant",
                inputs=[],
                outputs=["Y"],
                value=oh.make_tensor("", data_type, output_shape, output_value),
            )
        ]
        graph = oh.make_graph(
            nodes, "test_constant", [], [oh.make_tensor_value_info("Y", data_type, output_shape)]
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", from_opset), to_opset)
        assert converted_model.graph.node[0].op_type == "Constant"
        assert converted_model.graph.output[0].type.tensor_type.elem_type == data_type
        assert converted_model.opset_import[0].version == to_opset

    # Test Constant Adapter: 9 -> 8
    def test_constant_9_8(self) -> None:
        from_opset = 9
        to_opset = 8
        data_type = onnxl.TensorProto.UINT64
        output_shape = [2, 3, 4]
        output_value = np.arange(24)

        nodes = [
            oh.make_node(
                "Constant",
                inputs=[],
                outputs=["Y"],
                value=oh.make_tensor("", data_type, output_shape, output_value),
            )
        ]
        graph = oh.make_graph(
            nodes, "test_constant", [], [oh.make_tensor_value_info("Y", data_type, output_shape)]
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", from_opset), to_opset)
        assert converted_model.graph.node[0].op_type == "Constant"
        assert converted_model.graph.output[0].type.tensor_type.elem_type == data_type
        assert converted_model.opset_import[0].version == to_opset

    # Test Flatten Adapter: 8 -> 9
    def test_flatten_8_9(self) -> None:
        from_opset = 8
        to_opset = 9
        data_type = onnxl.TensorProto.FLOAT

        nodes = [oh.make_node("Flatten", inputs=["X"], outputs=["Y"], axis=1)]
        graph = oh.make_graph(
            nodes,
            "test_flatten",
            [oh.make_tensor_value_info("X", data_type, [2, 3, 4])],
            [oh.make_tensor_value_info("Y", data_type, [2, 12])],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", from_opset), to_opset)
        assert converted_model.graph.node[0].op_type == "Flatten"
        assert converted_model.graph.output[0].type.tensor_type.elem_type == data_type
        assert converted_model.opset_import[0].version == to_opset

    # Test Flatten Adapter: 9 -> 8
    def test_flatten_9_8(self) -> None:
        from_opset = 9
        to_opset = 8
        data_type = onnxl.TensorProto.UINT64

        nodes = [oh.make_node("Flatten", inputs=["X"], outputs=["Y"], axis=1)]
        graph = oh.make_graph(
            nodes,
            "test_flatten",
            [oh.make_tensor_value_info("X", data_type, [2, 3, 4])],
            [oh.make_tensor_value_info("Y", data_type, [2, 12])],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", from_opset), to_opset)
        assert converted_model.graph.node[1].op_type == "Flatten"
        assert converted_model.graph.output[0].type.tensor_type.elem_type == data_type
        assert converted_model.opset_import[0].version == to_opset

    # Test PRelu Adapter: 8 -> 9
    def test_prelu_8_9(self) -> None:
        from_opset = 8
        to_opset = 9
        data_type = onnxl.TensorProto.FLOAT
        input_shape = [2, 3, 4]

        nodes = [oh.make_node("PRelu", inputs=["X", "Slope"], outputs=["Y"])]
        graph = oh.make_graph(
            nodes,
            "test_prelu",
            [
                oh.make_tensor_value_info("X", data_type, input_shape),
                oh.make_tensor_value_info("Slope", data_type, input_shape),
            ],
            [oh.make_tensor_value_info("Y", data_type, input_shape)],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", from_opset), to_opset)
        assert converted_model.graph.node[0].op_type == "PRelu"
        assert converted_model.graph.output[0].type.tensor_type.elem_type == data_type
        assert converted_model.opset_import[0].version == to_opset

    # Test PRelu Adapter: 9 -> 8
    def test_prelu_9_8(self) -> None:
        from_opset = 9
        to_opset = 8
        data_type = onnxl.TensorProto.UINT64
        input_shape = [2, 3, 4]

        nodes = [oh.make_node("PRelu", inputs=["X", "Slope"], outputs=["Y"])]
        graph = oh.make_graph(
            nodes,
            "test_prelu",
            [
                oh.make_tensor_value_info("X", data_type, input_shape),
                oh.make_tensor_value_info("Slope", data_type, input_shape),
            ],
            [oh.make_tensor_value_info("Y", data_type, input_shape)],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", from_opset), to_opset)
        assert converted_model.graph.node[2].op_type == "PRelu"
        assert converted_model.graph.output[0].type.tensor_type.elem_type == data_type
        assert converted_model.opset_import[0].version == to_opset

    # Test Greater Adapter: 8 -> 9
    def test_greater_8_9(self) -> None:
        from_opset = 8
        to_opset = 9
        data_type = onnxl.TensorProto.FLOAT
        input_shape = [2, 3, 4]

        nodes = [oh.make_node("Greater", inputs=["X1", "X2"], outputs=["Y"])]
        graph = oh.make_graph(
            nodes,
            "test_greater",
            [
                oh.make_tensor_value_info("X1", data_type, input_shape),
                oh.make_tensor_value_info("X2", data_type, input_shape),
            ],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.BOOL, input_shape)],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", from_opset), to_opset)
        assert converted_model.graph.node[0].op_type == "Greater"
        assert (
            converted_model.graph.output[0].type.tensor_type.elem_type == onnxl.TensorProto.BOOL
        )
        assert converted_model.opset_import[0].version == to_opset

    # Test Greater Adapter: 9 -> 8
    def test_greater_9_8(self) -> None:
        from_opset = 9
        to_opset = 8
        data_type = onnxl.TensorProto.UINT64
        input_shape = [2, 3, 4]

        nodes = [oh.make_node("Greater", inputs=["X1", "X2"], outputs=["Y"])]
        graph = oh.make_graph(
            nodes,
            "test_greater",
            [
                oh.make_tensor_value_info("X1", data_type, input_shape),
                oh.make_tensor_value_info("X2", data_type, input_shape),
            ],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.BOOL, input_shape)],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", from_opset), to_opset)
        assert converted_model.graph.node[2].op_type == "Greater"
        assert (
            converted_model.graph.output[0].type.tensor_type.elem_type == onnxl.TensorProto.BOOL
        )
        assert converted_model.opset_import[0].version == to_opset

    # Test Less Adapter: 8 -> 9
    def test_less_8_9(self) -> None:
        from_opset = 8
        to_opset = 9
        data_type = onnxl.TensorProto.FLOAT
        input_shape = [2, 3, 4]

        nodes = [oh.make_node("Less", inputs=["X1", "X2"], outputs=["Y"])]
        graph = oh.make_graph(
            nodes,
            "test_less",
            [
                oh.make_tensor_value_info("X1", data_type, input_shape),
                oh.make_tensor_value_info("X2", data_type, input_shape),
            ],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.BOOL, input_shape)],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", from_opset), to_opset)
        assert converted_model.graph.node[0].op_type == "Less"
        assert (
            converted_model.graph.output[0].type.tensor_type.elem_type == onnxl.TensorProto.BOOL
        )
        assert converted_model.opset_import[0].version == to_opset

    # Test Less Adapter: 9 -> 8
    def test_less_9_8(self) -> None:
        from_opset = 9
        to_opset = 8
        data_type = onnxl.TensorProto.UINT64
        input_shape = [2, 3, 4]

        nodes = [oh.make_node("Less", inputs=["X1", "X2"], outputs=["Y"])]
        graph = oh.make_graph(
            nodes,
            "test_less",
            [
                oh.make_tensor_value_info("X1", data_type, input_shape),
                oh.make_tensor_value_info("X2", data_type, input_shape),
            ],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.BOOL, input_shape)],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", from_opset), to_opset)
        assert converted_model.graph.node[2].op_type == "Less"
        assert (
            converted_model.graph.output[0].type.tensor_type.elem_type == onnxl.TensorProto.BOOL
        )
        assert converted_model.opset_import[0].version == to_opset

    # Test MatMul Adapter: 8 -> 9
    def test_matmul_8_9(self) -> None:
        from_opset = 8
        to_opset = 9
        data_type = onnxl.TensorProto.FLOAT

        nodes = [oh.make_node("MatMul", inputs=["X1", "X2"], outputs=["Y"])]
        graph = oh.make_graph(
            nodes,
            "test_matmul",
            [
                oh.make_tensor_value_info("X1", data_type, [3, 4]),
                oh.make_tensor_value_info("X2", data_type, [4, 3]),
            ],
            [oh.make_tensor_value_info("Y", data_type, [3, 3])],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", from_opset), to_opset)
        assert converted_model.graph.node[0].op_type == "MatMul"
        assert converted_model.graph.output[0].type.tensor_type.elem_type == data_type
        assert converted_model.opset_import[0].version == to_opset

    # Test MatMul Adapter: 9 -> 8
    def test_matmul_9_8(self) -> None:
        from_opset = 9
        to_opset = 8
        data_type = onnxl.TensorProto.UINT64

        nodes = [oh.make_node("MatMul", inputs=["X1", "X2"], outputs=["Y"])]
        graph = oh.make_graph(
            nodes,
            "test_matmul",
            [
                oh.make_tensor_value_info("X1", data_type, [3, 4]),
                oh.make_tensor_value_info("X2", data_type, [4, 3]),
            ],
            [oh.make_tensor_value_info("Y", data_type, [3, 3])],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", from_opset), to_opset)
        assert converted_model.graph.node[2].op_type == "MatMul"
        assert converted_model.graph.output[0].type.tensor_type.elem_type == data_type
        assert converted_model.opset_import[0].version == to_opset

    # Test Gemm Adapter: 8 -> 9
    def test_gemm_8_9(self) -> None:
        from_opset = 8
        to_opset = 9
        data_type = onnxl.TensorProto.FLOAT

        nodes = [oh.make_node("Gemm", inputs=["X1", "X2", "X3"], outputs=["Y"])]
        graph = oh.make_graph(
            nodes,
            "test_gemm",
            [
                oh.make_tensor_value_info("X1", data_type, [3, 4]),
                oh.make_tensor_value_info("X2", data_type, [4, 3]),
                oh.make_tensor_value_info("X3", data_type, [3, 3]),
            ],
            [oh.make_tensor_value_info("Y", data_type, [3, 3])],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", from_opset), to_opset)
        assert converted_model.graph.node[0].op_type == "Gemm"
        assert converted_model.graph.output[0].type.tensor_type.elem_type == data_type
        assert converted_model.opset_import[0].version == to_opset

    # Test Gemm Adapter: 9 -> 8
    def test_gemm_9_8(self) -> None:
        from_opset = 9
        to_opset = 8
        data_type = onnxl.TensorProto.UINT64

        nodes = [oh.make_node("Gemm", inputs=["X1", "X2", "X3"], outputs=["Y"])]
        graph = oh.make_graph(
            nodes,
            "test_gemm",
            [
                oh.make_tensor_value_info("X1", data_type, [3, 4]),
                oh.make_tensor_value_info("X2", data_type, [4, 3]),
                oh.make_tensor_value_info("X3", data_type, [3, 3]),
            ],
            [oh.make_tensor_value_info("Y", data_type, [3, 3])],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", from_opset), to_opset)
        assert converted_model.graph.node[3].op_type == "Gemm"
        assert converted_model.graph.output[0].type.tensor_type.elem_type == data_type
        assert converted_model.opset_import[0].version == to_opset

    # Test Upsample Adapter: 8 -> 9
    def test_upsample_8_9(self) -> None:
        from_opset = 8
        to_opset = 9
        data_type = onnxl.TensorProto.FLOAT

        nodes = [
            oh.make_node(
                "Upsample",
                inputs=["X"],
                outputs=["Y"],
                mode="nearest",
                scales=[1.0, 1.0, 2.0, 3.0],
            )
        ]
        graph = oh.make_graph(
            nodes,
            "test_upsample_8_9",
            [oh.make_tensor_value_info("X", data_type, [1, 1, 2, 2])],
            [oh.make_tensor_value_info("Y", data_type, [1, 1, 4, 6])],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", from_opset), to_opset)
        assert len(converted_model.graph.node) == 2
        assert converted_model.graph.node[0].op_type == "Constant"
        assert converted_model.graph.node[1].op_type == "Upsample"
        assert len(converted_model.graph.node[1].attribute) == 1
        assert converted_model.graph.node[1].attribute[0].name == "mode"
        assert converted_model.opset_import[0].version == to_opset

    def helper_upsample_with_initializer(self, raw_scale: bool = False) -> None:
        """Helper for Upsample Adapter: 9 -> 8 with initializer."""
        from_opset = 9
        to_opset = 8
        data_type = onnxl.TensorProto.FLOAT

        nodes = [oh.make_node("Upsample", inputs=["X", "Scales"], outputs=["Y"], mode="nearest")]
        scale_value = [1.0, 1.0, 2.0, 3.0]
        scale_tensor = oh.make_tensor(
            "Scales",
            onnxl.TensorProto.FLOAT,
            [4],
            bytes(struct.pack("4f", *scale_value)) if raw_scale else scale_value,
            raw_scale,
        )
        graph = oh.make_graph(
            nodes,
            "test_upsample",
            [
                oh.make_tensor_value_info("X", data_type, [1, 1, 2, 2]),
                oh.make_tensor_value_info("Scales", data_type, [4]),
            ],
            [oh.make_tensor_value_info("Y", data_type, [1, 1, 4, 6])],
            [scale_tensor],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", from_opset), to_opset)
        assert converted_model.graph.node[0].op_type == "Upsample"
        assert len(converted_model.graph.initializer) == 0
        assert len(converted_model.graph.node[0].attribute) == 2
        assert converted_model.graph.node[0].attribute[1].name == "scales"
        assert converted_model.opset_import[0].version == to_opset

    def helper_upsample_with_constant(self, raw_scale: bool = False) -> None:
        """Helper for Upsample Adapter: 9 -> 8 with constant node."""
        from_opset = 9
        to_opset = 8
        data_type = onnxl.TensorProto.FLOAT

        scale_value = [1.0, 1.0, 2.0, 3.0]
        scale_tensor = oh.make_tensor(
            "const_value",
            onnxl.TensorProto.FLOAT,
            [4],
            bytes(struct.pack("4f", *scale_value)) if raw_scale else scale_value,
            raw_scale,
        )
        nodes = [
            oh.make_node("Constant", inputs=[], outputs=["Constant_Output"], value=scale_tensor),
            oh.make_node(
                "Upsample", inputs=["X", "Constant_Output"], outputs=["Y"], mode="nearest"
            ),
        ]
        graph = oh.make_graph(
            nodes,
            "test_upsample",
            [oh.make_tensor_value_info("X", data_type, [1, 1, 2, 2])],
            [oh.make_tensor_value_info("Y", data_type, [1, 1, 4, 6])],
            value_info=[oh.make_tensor_value_info("Constant_Output", data_type, [4])],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", from_opset), to_opset)
        assert len(converted_model.graph.node) == 1
        assert converted_model.graph.node[0].op_type == "Upsample"
        assert len(converted_model.graph.node[0].attribute) == 2
        assert converted_model.graph.node[0].attribute[1].name == "scales"
        assert converted_model.opset_import[0].version == to_opset

    # Test Upsample Adapter: 9 -> 8 with constant node
    def test_upsample_with_constant_node_9_8(self) -> None:
        self.helper_upsample_with_constant(raw_scale=False)

    # Test Upsample Adapter: 9 -> 8 with initializer
    def test_upsample_with_initializer_9_8(self) -> None:
        self.helper_upsample_with_initializer(raw_scale=False)

    # Test Upsample Adapter: 9 -> 8 with raw constant
    def test_upsample_with_raw_constant_node_9_8(self) -> None:
        self.helper_upsample_with_constant(raw_scale=True)

    # Test Upsample Adapter: 9 -> 8 with raw initializer
    def test_upsample_with_raw_initializer_9_8(self) -> None:
        self.helper_upsample_with_initializer(raw_scale=True)

    # Test Scan Adapter: 8 -> 9
    def test_scan_8_9(self) -> None:
        from_opset = 8
        to_opset = 9
        data_type = onnxl.TensorProto.FLOAT

        node1 = oh.make_node("Add", inputs=["sum_in", "next"], outputs=["sum_out"])
        node2 = oh.make_node("Identity", inputs=["sum_out"], outputs=["scan_out"])
        g = oh.make_graph(
            [node1, node2],
            "scan_body",
            [
                oh.make_tensor_value_info("sum_in", data_type, [2]),
                oh.make_tensor_value_info("next", data_type, [2]),
            ],
            [
                oh.make_tensor_value_info("sum_out", data_type, [2]),
                oh.make_tensor_value_info("scan_out", data_type, [2]),
            ],
        )
        no_sequence_lens = ""
        nodes = [
            oh.make_node(
                "Scan",
                inputs=[no_sequence_lens, "initial", "x"],
                outputs=["y", "z"],
                body=g,
                num_scan_inputs=1,
            )
        ]
        initial = oh.make_tensor_value_info("initial", data_type, [1, 2])
        x = oh.make_tensor_value_info("x", data_type, [1, 3, 2])
        y = oh.make_tensor_value_info("y", data_type, [1, 2])
        z = oh.make_tensor_value_info("z", data_type, [1, 3, 2])

        graph = oh.make_graph(nodes, "test_scan_8_9", [initial, x], [y, z])
        converted_model = self._converted(graph, oh.make_operatorsetid("", from_opset), to_opset)
        assert converted_model.graph.node[0].op_type == "Scan"
        assert converted_model.opset_import[0].version == to_opset

    # Test Cast Adapter: 8 -> 9
    def test_cast_8_9(self) -> None:
        from_opset = 8
        to_opset = 9
        data_type_from = onnxl.TensorProto.FLOAT
        data_type_to = onnxl.TensorProto.UINT32

        nodes = [oh.make_node("Cast", inputs=["X"], outputs=["Y"], to=onnxl.TensorProto.UINT32)]
        graph = oh.make_graph(
            nodes,
            "test_cast",
            [oh.make_tensor_value_info("X", data_type_from, [2, 3])],
            [oh.make_tensor_value_info("Y", data_type_to, [2, 3])],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", from_opset), to_opset)
        assert converted_model.graph.node[0].op_type == "Cast"
        assert converted_model.graph.output[0].type.tensor_type.elem_type == data_type_to
        assert converted_model.opset_import[0].version == to_opset

    # Test Split Adapter: 13 -> 12
    def test_split_13_12(self) -> None:
        nodes = [
            oh.make_node(
                "Constant",
                [],
                ["split"],
                value=oh.make_tensor("", onnxl.TensorProto.INT64, [2], [2, 3]),
            ),
            oh.make_node("Split", ["X", "split"], ["Y1", "Y2"]),
        ]
        graph = oh.make_graph(
            nodes,
            "test",
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, (5,))],
            [
                oh.make_tensor_value_info("Y1", onnxl.TensorProto.FLOAT, (2,)),
                oh.make_tensor_value_info("Y2", onnxl.TensorProto.FLOAT, (3,)),
            ],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 13), 12)
        assert converted_model.graph.node[0].op_type == "Split"
        assert converted_model.opset_import[0].version == 12

    def test_split_with_optional_input(self) -> None:
        nodes = [oh.make_node("Split", ["X"], ["Y1", "Y2"], axis=1)]
        graph = oh.make_graph(
            nodes,
            "test_split_optional_input",
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, (6,))],
            [
                oh.make_tensor_value_info("Y1", onnxl.TensorProto.FLOAT, (3,)),
                oh.make_tensor_value_info("Y2", onnxl.TensorProto.FLOAT, (3,)),
            ],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 12), 18)
        assert converted_model.graph.node[0].op_type == "Split"
        assert converted_model.opset_import[0].version == 18
        assert len(converted_model.graph.node[0].output) == 2

    # Test Split Adapter: 12 -> 13
    def test_split_12_13(self) -> None:
        nodes = [oh.make_node("Split", ["X"], ["Y1", "Y2"], split=[2, 3])]
        graph = oh.make_graph(
            nodes,
            "test",
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, (5,))],
            [
                oh.make_tensor_value_info("Y1", onnxl.TensorProto.FLOAT, (2,)),
                oh.make_tensor_value_info("Y2", onnxl.TensorProto.FLOAT, (3,)),
            ],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 12), 13)
        assert converted_model.graph.node[0].op_type == "Constant"
        assert converted_model.graph.node[1].op_type == "Split"
        assert converted_model.opset_import[0].version == 13

    # Test Split Adapter: 13 -> 12 with optional split input
    def test_split_13_12_optional_input(self) -> None:
        nodes = [oh.make_node("Split", ["X"], ["Y1", "Y2"], axis=0)]
        graph = oh.make_graph(
            nodes,
            "test_split_13_12_optional",
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, (10,))],
            [
                oh.make_tensor_value_info("Y1", onnxl.TensorProto.FLOAT, (5,)),
                oh.make_tensor_value_info("Y2", onnxl.TensorProto.FLOAT, (5,)),
            ],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 13), 12)
        assert converted_model.graph.node[0].op_type == "Split"
        assert converted_model.opset_import[0].version == 12

    # Test AxesInputToAttribute Adapter: 13 -> 12
    def test_axes_input_to_attr_13_12(self) -> None:
        nodes = [
            oh.make_node(
                "Constant",
                [],
                ["axes"],
                value=oh.make_tensor("", onnxl.TensorProto.INT64, [1], [0]),
            ),
            oh.make_node("ReduceSum", ["X", "axes"], ["Y"]),
        ]
        graph = oh.make_graph(
            nodes,
            "test",
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, (5, 5))],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, (1, 5))],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 13), 12)
        assert converted_model.graph.node[0].op_type == "ReduceSum"
        assert converted_model.opset_import[0].version == 12

    # Test AxesAttributeToInput Adapter: 12 -> 13
    def test_axes_attr_to_input_12_13(self) -> None:
        nodes = [oh.make_node("ReduceSum", ["X"], ["Y"], axes=[0])]
        graph = oh.make_graph(
            nodes,
            "test",
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, (5, 5))],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, (1, 5))],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 12), 13)
        assert converted_model.graph.node[0].op_type == "Constant"
        assert converted_model.opset_import[0].version == 13

    # Test AxesInputToAttribute Adapter: 13 -> 11 with optional axes input
    def test_squeeze_13_11_optional_axes(self) -> None:
        nodes = [oh.make_node("Squeeze", ["X"], ["Y"])]
        graph = oh.make_graph(
            nodes,
            "test_squeeze_13_11_optional",
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, (1, 10, 1))],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, (10,))],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 13), 11)
        assert converted_model.graph.node[0].op_type == "Squeeze"
        assert converted_model.opset_import[0].version == 11

    # Test Reshape Adapter: 4 -> 5 with shape attribute becoming input
    def test_reshape_4_5_optional_shape(self) -> None:
        nodes = [oh.make_node("Reshape", ["X"], ["Y"], shape=[2, 5])]
        graph = oh.make_graph(
            nodes,
            "test_reshape_4_5",
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, (10,))],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, (2, 5))],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 4), 5)
        assert converted_model.graph.node[0].op_type == "Constant"
        assert converted_model.graph.node[1].op_type == "Reshape"
        assert converted_model.opset_import[0].version == 5

    # Test Resize Adapter: 10 -> 11
    def test_resize_10_11_bounds_check(self) -> None:
        nodes = [
            oh.make_node(
                "Constant",
                [],
                ["scales"],
                value=oh.make_tensor("", onnxl.TensorProto.FLOAT, [4], [1.0, 1.0, 2.0, 2.0]),
            ),
            oh.make_node("Resize", ["X", "scales"], ["Y"], mode="nearest"),
        ]
        graph = oh.make_graph(
            nodes,
            "test_resize_10_11",
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, (1, 1, 2, 2))],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, (1, 1, 4, 4))],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 10), 11)
        assert converted_model.opset_import[0].version == 11

    # Test Scatter Adapter: 10 -> 11
    def test_scatter_10_11_bounds_check(self) -> None:
        nodes = [oh.make_node("Scatter", ["data", "indices", "updates"], ["Y"], axis=0)]
        graph = oh.make_graph(
            nodes,
            "test_scatter_10_11",
            [
                oh.make_tensor_value_info("data", onnxl.TensorProto.FLOAT, (3,)),
                oh.make_tensor_value_info("indices", onnxl.TensorProto.INT64, (2,)),
                oh.make_tensor_value_info("updates", onnxl.TensorProto.FLOAT, (2,)),
            ],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, (3,))],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 10), 11)
        assert converted_model.graph.node[0].op_type == "ScatterElements"
        assert converted_model.opset_import[0].version == 11

    # Test Slice Adapter: 9 -> 10
    def test_slice_9_10(self) -> None:
        nodes = [oh.make_node("Slice", ["X"], ["Y"], axes=[0, 1], starts=[0, 0], ends=[3, 10])]
        graph = oh.make_graph(
            nodes,
            "test",
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, (20, 10, 5))],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, (3, 10, 5))],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 9), 10)
        assert converted_model.graph.node[0].op_type == "Constant"
        assert converted_model.graph.node[1].op_type == "Constant"
        assert converted_model.graph.node[2].op_type == "Constant"
        assert converted_model.graph.node[3].op_type == "Slice"
        assert converted_model.opset_import[0].version == 10
        assert len(converted_model.graph.node[3].input) == 4
        assert len(converted_model.graph.node[3].attribute) == 0

    # Test RNN Adapter: 13 -> 14
    def test_rnn_13_14(self) -> None:
        from_opset = 13
        to_opset = 14
        data_type = onnxl.TensorProto.FLOAT
        seq_length = 1
        batch_size = 2
        input_size = 3
        num_directions = 1
        hidden_size = 5

        nodes = [
            oh.make_node(
                "RNN", inputs=["X", "W", "R"], outputs=["", "Y_h"], hidden_size=hidden_size
            )
        ]
        graph = oh.make_graph(
            nodes,
            "test_rnn",
            [
                oh.make_tensor_value_info("X", data_type, [seq_length, batch_size, input_size]),
                oh.make_tensor_value_info(
                    "W", data_type, [num_directions, hidden_size, input_size]
                ),
                oh.make_tensor_value_info(
                    "R", data_type, [num_directions, hidden_size, hidden_size]
                ),
                oh.make_tensor_value_info("B", data_type, [num_directions, 2 * hidden_size]),
            ],
            [
                oh.make_tensor_value_info(
                    "Y_h", data_type, [num_directions, batch_size, hidden_size]
                )
            ],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", from_opset), to_opset)
        assert converted_model.graph.node[0].op_type == "RNN"
        assert converted_model.opset_import[0].version == to_opset
        assert len(converted_model.graph.node[0].attribute) == 2
        assert converted_model.graph.node[0].attribute[1].name == "layout"

    # Test GRU Adapter: 13 -> 14
    def test_gru_13_14(self) -> None:
        from_opset = 13
        to_opset = 14
        data_type = onnxl.TensorProto.FLOAT
        seq_length = 1
        batch_size = 2
        input_size = 3
        num_directions = 1
        hidden_size = 5

        nodes = [
            oh.make_node(
                "GRU", inputs=["X", "W", "R"], outputs=["", "Y_h"], hidden_size=hidden_size
            )
        ]
        graph = oh.make_graph(
            nodes,
            "test_gru",
            [
                oh.make_tensor_value_info("X", data_type, [seq_length, batch_size, input_size]),
                oh.make_tensor_value_info(
                    "W", data_type, [num_directions, 3 * hidden_size, input_size]
                ),
                oh.make_tensor_value_info(
                    "R", data_type, [num_directions, 3 * hidden_size, hidden_size]
                ),
                oh.make_tensor_value_info("B", data_type, [num_directions, 6 * hidden_size]),
            ],
            [
                oh.make_tensor_value_info(
                    "Y_h", data_type, [num_directions, batch_size, hidden_size]
                )
            ],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", from_opset), to_opset)
        assert converted_model.graph.node[0].op_type == "GRU"
        assert converted_model.opset_import[0].version == to_opset
        assert len(converted_model.graph.node[0].attribute) == 2
        assert converted_model.graph.node[0].attribute[1].name == "layout"

    # Test LSTM Adapter: 13 -> 14
    def test_lstm_13_14(self) -> None:
        from_opset = 13
        to_opset = 14
        data_type = onnxl.TensorProto.FLOAT
        seq_length = 1
        batch_size = 2
        input_size = 3
        num_directions = 1
        hidden_size = 5

        nodes = [
            oh.make_node(
                "LSTM", inputs=["X", "W", "R"], outputs=["", "Y_h"], hidden_size=hidden_size
            )
        ]
        graph = oh.make_graph(
            nodes,
            "test_lstm",
            [
                oh.make_tensor_value_info("X", data_type, [seq_length, batch_size, input_size]),
                oh.make_tensor_value_info(
                    "W", data_type, [num_directions, 4 * hidden_size, input_size]
                ),
                oh.make_tensor_value_info(
                    "R", data_type, [num_directions, 4 * hidden_size, hidden_size]
                ),
                oh.make_tensor_value_info("B", data_type, [num_directions, 8 * hidden_size]),
            ],
            [
                oh.make_tensor_value_info(
                    "Y_h", data_type, [num_directions, batch_size, hidden_size]
                )
            ],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", from_opset), to_opset)
        assert converted_model.graph.node[0].op_type == "LSTM"
        assert converted_model.opset_import[0].version == to_opset
        assert len(converted_model.graph.node[0].attribute) == 2
        assert converted_model.graph.node[0].attribute[1].name == "layout"

    # Test RNN Adapter: 14 -> 13
    def test_rnn_14_13(self) -> None:
        from_opset = 14
        to_opset = 13
        data_type = onnxl.TensorProto.FLOAT
        seq_length = 1
        batch_size = 2
        input_size = 3
        num_directions = 1
        hidden_size = 5

        nodes = [
            oh.make_node(
                "RNN",
                inputs=["X", "W", "R"],
                outputs=["", "Y_h"],
                hidden_size=hidden_size,
                layout=0,
            )
        ]
        graph = oh.make_graph(
            nodes,
            "test_rnn",
            [
                oh.make_tensor_value_info("X", data_type, [seq_length, batch_size, input_size]),
                oh.make_tensor_value_info(
                    "W", data_type, [num_directions, hidden_size, input_size]
                ),
                oh.make_tensor_value_info(
                    "R", data_type, [num_directions, hidden_size, hidden_size]
                ),
                oh.make_tensor_value_info("B", data_type, [num_directions, 2 * hidden_size]),
            ],
            [
                oh.make_tensor_value_info(
                    "Y_h", data_type, [num_directions, batch_size, hidden_size]
                )
            ],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", from_opset), to_opset)
        assert converted_model.graph.node[0].op_type == "RNN"
        assert converted_model.opset_import[0].version == to_opset
        assert len(converted_model.graph.node[0].attribute) == 1

    # Test GRU Adapter: 14 -> 13
    def test_gru_14_13(self) -> None:
        from_opset = 14
        to_opset = 13
        data_type = onnxl.TensorProto.FLOAT
        seq_length = 1
        batch_size = 2
        input_size = 3
        num_directions = 1
        hidden_size = 5

        nodes = [
            oh.make_node(
                "GRU",
                inputs=["X", "W", "R"],
                outputs=["", "Y_h"],
                hidden_size=hidden_size,
                layout=0,
            )
        ]
        graph = oh.make_graph(
            nodes,
            "test_gru",
            [
                oh.make_tensor_value_info("X", data_type, [seq_length, batch_size, input_size]),
                oh.make_tensor_value_info(
                    "W", data_type, [num_directions, 3 * hidden_size, input_size]
                ),
                oh.make_tensor_value_info(
                    "R", data_type, [num_directions, 3 * hidden_size, hidden_size]
                ),
                oh.make_tensor_value_info("B", data_type, [num_directions, 6 * hidden_size]),
            ],
            [
                oh.make_tensor_value_info(
                    "Y_h", data_type, [num_directions, batch_size, hidden_size]
                )
            ],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", from_opset), to_opset)
        assert converted_model.graph.node[0].op_type == "GRU"
        assert converted_model.opset_import[0].version == to_opset
        assert len(converted_model.graph.node[0].attribute) == 1

    # Test LSTM Adapter: 14 -> 13
    def test_lstm_14_13(self) -> None:
        from_opset = 14
        to_opset = 13
        data_type = onnxl.TensorProto.FLOAT
        seq_length = 1
        batch_size = 2
        input_size = 3
        num_directions = 1
        hidden_size = 5

        nodes = [
            oh.make_node(
                "LSTM",
                inputs=["X", "W", "R"],
                outputs=["", "Y_h"],
                hidden_size=hidden_size,
                layout=0,
            )
        ]
        graph = oh.make_graph(
            nodes,
            "test_lstm",
            [
                oh.make_tensor_value_info("X", data_type, [seq_length, batch_size, input_size]),
                oh.make_tensor_value_info(
                    "W", data_type, [num_directions, 4 * hidden_size, input_size]
                ),
                oh.make_tensor_value_info(
                    "R", data_type, [num_directions, 4 * hidden_size, hidden_size]
                ),
                oh.make_tensor_value_info("B", data_type, [num_directions, 8 * hidden_size]),
            ],
            [
                oh.make_tensor_value_info(
                    "Y_h", data_type, [num_directions, batch_size, hidden_size]
                )
            ],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", from_opset), to_opset)
        assert converted_model.graph.node[0].op_type == "LSTM"
        assert converted_model.opset_import[0].version == to_opset
        assert len(converted_model.graph.node[0].attribute) == 1

    # Test Pad Adapter: 10 -> 11
    def test_pad_10_11(self) -> None:
        pads = (0, 1, 2, 0, 2, 1)
        nodes = [oh.make_node("Pad", ["X"], ["Y"], pads=pads)]
        graph = oh.make_graph(
            nodes,
            "test",
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, (1, 2, 2))],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, (1, 5, 5))],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 10), 11)
        assert converted_model.graph.node[1].op_type == "Pad"
        assert converted_model.opset_import[0].version == 11

    def test_pad_with_value_10_11(self) -> None:
        pads = (0, 1, 2, 0, 2, 1)
        nodes = [oh.make_node("Pad", ["X"], ["Y"], pads=pads, value=1.0)]
        graph = oh.make_graph(
            nodes,
            "test",
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, (1, 2, 2))],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, (1, 5, 5))],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 10), 11)
        assert converted_model.graph.node[1].op_type == "Pad"
        assert converted_model.opset_import[0].version == 11

    # Test that subgraphs are converted
    def test_if_subgraph_10_11(self) -> None:
        from_opset = 10
        to_opset = 11
        data_type = onnxl.TensorProto.FLOAT
        data_shape = [2]

        subg1_node = [
            oh.make_node("Clip", inputs=["sub_in"], outputs=["sub_out"], min=2.0, max=3.0)
        ]
        subg1 = oh.make_graph(
            subg1_node,
            "then_g",
            [oh.make_tensor_value_info("sub_in", data_type, data_shape)],
            [oh.make_tensor_value_info("sub_out", data_type, data_shape)],
        )

        subg2_node = [
            oh.make_node("Clip", inputs=["sub_in"], outputs=["sub_out"], min=2.0, max=3.0)
        ]
        subg2 = oh.make_graph(
            subg2_node,
            "then_g",
            [oh.make_tensor_value_info("sub_in", data_type, data_shape)],
            [oh.make_tensor_value_info("sub_out", data_type, data_shape)],
        )

        node = [
            oh.make_node(
                "If", inputs=["cond"], outputs=["out"], then_branch=subg1, else_branch=subg2
            )
        ]
        init = [oh.make_tensor("sub_in", data_type, data_shape, [4.0, 5.0])]
        graph = oh.make_graph(
            node,
            "test_subgraphs",
            [oh.make_tensor_value_info("cond", onnxl.TensorProto.BOOL, [])],
            [oh.make_tensor_value_info("out", data_type, data_shape)],
            init,
        )
        converted = self._converted(graph, oh.make_operatorsetid("", from_opset), to_opset)
        assert converted.graph.node[0].op_type == "If"
        assert converted.opset_import[0].version == to_opset
        assert converted.graph.node[0].attribute[0].g.node[2].op_type == "Clip"
        assert len(converted.graph.node[0].attribute[0].g.node[2].attribute) == 0
        assert converted.graph.node[0].attribute[1].g.node[2].op_type == "Clip"
        assert len(converted.graph.node[0].attribute[1].g.node[2].attribute) == 0

    # Test initializer not in graph input (above IR4)
    def test_initializer_not_in_input_above_ir4(self) -> None:
        nodes = [oh.make_node("BatchNormalization", ["X", "scale", "B", "mean", "var"], ["Y"])]
        scale_tensor = oh.make_tensor("scale", onnxl.TensorProto.FLOAT, [2], [0.55, 0.72])
        b_tensor = oh.make_tensor("B", onnxl.TensorProto.FLOAT, [2], [0.60, 0.54])
        mean_tensor = oh.make_tensor("mean", onnxl.TensorProto.FLOAT, [2], [0.42, 0.65])
        var_tensor = oh.make_tensor("var", onnxl.TensorProto.FLOAT, [2], [0.44, 0.89])

        graph = oh.make_graph(
            nodes,
            "test",
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, (1, 2, 2, 3))],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, (1, 2, 2, 3))],
            [scale_tensor, b_tensor, mean_tensor, var_tensor],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 11), 12)
        assert converted_model.graph.node[0].op_type == "BatchNormalization"
        assert converted_model.opset_import[0].version == 12

    def test_softmax_12_13(self) -> None:
        axis = 0
        nodes = [oh.make_node("Softmax", ["X"], ["Y"], axis=axis)]
        graph = oh.make_graph(
            nodes,
            "test",
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, (1, 2, 3))],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, (1, 2, 3))],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 11), 13)
        assert converted_model.graph.node[0].op_type == "Shape"
        assert converted_model.graph.node[1].op_type == "Flatten"
        assert converted_model.graph.node[1].attribute[0].name == "axis"
        assert converted_model.graph.node[1].attribute[0].i == axis
        assert converted_model.graph.node[2].op_type == "Softmax"
        assert converted_model.graph.node[2].attribute[0].name == "axis"
        assert converted_model.graph.node[2].attribute[0].i == -1
        assert converted_model.graph.node[3].op_type == "Reshape"
        assert converted_model.opset_import[0].version == 13

    def test_softmax_13_12(self) -> None:
        axis = -1
        nodes = [oh.make_node("Softmax", ["X"], ["Y"], axis=axis)]
        graph = oh.make_graph(
            nodes,
            "test",
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, (1, 2, 3))],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, (1, 2, 3))],
        )
        converted_model = self._converted(graph, oh.make_operatorsetid("", 13), 12)
        assert converted_model.graph.node[0].op_type == "Softmax"
        assert converted_model.graph.node[0].attribute[0].name == "axis"
        assert converted_model.graph.node[0].attribute[0].i == 2
        assert converted_model.opset_import[0].version == 12

    # Test rejects_missing_required_inputs (expanded from parameterized)
    def test_rejects_missing_required_inputs_cast_9_8(self) -> None:
        def test() -> None:
            nodes = [oh.make_node("Cast", [], ["Y"], to=onnxl.TensorProto.FLOAT)]
            graph = oh.make_graph(
                nodes,
                "test_cast_9_8_rejects_missing_input",
                [],
                [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, (1,))],
            )
            self._converted(graph, oh.make_operatorsetid("", 9), 8)

        self.assertRaises(RuntimeError, test)

    def test_rejects_missing_required_inputs_softmax_12_13(self) -> None:
        def test() -> None:
            nodes = [oh.make_node("Softmax", [], ["Y"], axis=1)]
            graph = oh.make_graph(
                nodes,
                "test_softmax_12_13_rejects_missing_input",
                [],
                [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, (1,))],
            )
            self._converted(graph, oh.make_operatorsetid("", 12), 13)

        self.assertRaises(RuntimeError, test)

    def test_rejects_missing_required_inputs_upsample_9_10(self) -> None:
        def test() -> None:
            nodes = [oh.make_node("Upsample", [], ["Y"], mode="nearest")]
            graph = oh.make_graph(
                nodes,
                "test_upsample_9_10_rejects_missing_input",
                [],
                [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, (1, 1, 2, 2))],
            )
            self._converted(graph, oh.make_operatorsetid("", 9), 10)

        self.assertRaises(RuntimeError, test)

    # Where 16 -> 15
    def test_where_16_15_success(self) -> None:
        nodes = [oh.make_node("Where", ["cond", "x", "y"], ["out"])]
        graph = oh.make_graph(
            nodes,
            "where",
            [
                oh.make_tensor_value_info("cond", onnxl.TensorProto.BOOL, (2, 3)),
                oh.make_tensor_value_info("x", onnxl.TensorProto.FLOAT, (2, 3)),
                oh.make_tensor_value_info("y", onnxl.TensorProto.FLOAT, (2, 3)),
            ],
            [oh.make_tensor_value_info("out", onnxl.TensorProto.FLOAT, (2, 3))],
        )
        converted = self._converted(graph, oh.make_operatorsetid("", 16), 15)
        assert converted.opset_import[0].version == 15

    def test_where_bfloat16_16_15_fails(self) -> None:
        def test() -> None:
            nodes = [oh.make_node("Where", ["cond", "x", "y"], ["out"])]
            graph = oh.make_graph(
                nodes,
                "where_bf16",
                [
                    oh.make_tensor_value_info("cond", onnxl.TensorProto.BOOL, (2, 3)),
                    oh.make_tensor_value_info("x", onnxl.TensorProto.BFLOAT16, (2, 3)),
                    oh.make_tensor_value_info("y", onnxl.TensorProto.BFLOAT16, (2, 3)),
                ],
                [oh.make_tensor_value_info("out", onnxl.TensorProto.BFLOAT16, (2, 3))],
            )
            self._converted(graph, oh.make_operatorsetid("", 16), 15)

        self.assertRaises(RuntimeError, test)

    def _make_scatter_graph(
        self, op_name: str, dtype: int = onnxl.TensorProto.FLOAT, reduction: str | None = None
    ) -> onnxl.GraphProto:
        """Builds a graph for ScatterElements or ScatterND with standard test shapes."""
        scatter_graph_config = {
            "ScatterElements": ((2, 3), (2, 3), (2, 3), {"axis": 0}),
            "ScatterND": ((4, 5), (2, 1), (2, 5), {}),
        }
        data_s, indices_s, updates_s, attrs = scatter_graph_config[op_name]
        attrs = dict(attrs)
        if reduction is not None:
            attrs["reduction"] = reduction
        nodes = [oh.make_node(op_name, ["data", "indices", "updates"], ["out"], **attrs)]
        return oh.make_graph(
            nodes,
            op_name.lower(),
            [
                oh.make_tensor_value_info("data", dtype, data_s),
                oh.make_tensor_value_info("indices", onnxl.TensorProto.INT64, indices_s),
                oh.make_tensor_value_info("updates", dtype, updates_s),
            ],
            [oh.make_tensor_value_info("out", dtype, data_s)],
        )

    # Scatter 16 -> 15: FLOAT with "none" reduction succeeds
    def test_scatter_elements_16_15_success(self) -> None:
        graph = self._make_scatter_graph("ScatterElements", onnxl.TensorProto.FLOAT, "none")
        converted = self._converted(graph, oh.make_operatorsetid("", 16), 15)
        assert converted.opset_import[0].version == 15

    def test_scatter_nd_16_15_success(self) -> None:
        graph = self._make_scatter_graph("ScatterND", onnxl.TensorProto.FLOAT, "none")
        converted = self._converted(graph, oh.make_operatorsetid("", 16), 15)
        assert converted.opset_import[0].version == 15

    # Scatter 16 -> 15: bfloat16 fails
    def test_scatter_elements_16_15_bfloat16_fails(self) -> None:
        def test() -> None:
            graph = self._make_scatter_graph("ScatterElements", onnxl.TensorProto.BFLOAT16)
            self._converted(graph, oh.make_operatorsetid("", 16), 15)

        self.assertRaises(RuntimeError, test)

    def test_scatter_nd_16_15_bfloat16_fails(self) -> None:
        def test() -> None:
            graph = self._make_scatter_graph("ScatterND", onnxl.TensorProto.BFLOAT16)
            self._converted(graph, oh.make_operatorsetid("", 16), 15)

        self.assertRaises(RuntimeError, test)

    # Scatter 16 -> 15: reduction "add" / "mul" fails
    def test_scatter_elements_16_15_reduction_add_fails(self) -> None:
        def test() -> None:
            graph = self._make_scatter_graph("ScatterElements", onnxl.TensorProto.FLOAT, "add")
            self._converted(graph, oh.make_operatorsetid("", 16), 15)

        self.assertRaises(RuntimeError, test)

    def test_scatter_elements_16_15_reduction_mul_fails(self) -> None:
        def test() -> None:
            graph = self._make_scatter_graph("ScatterElements", onnxl.TensorProto.FLOAT, "mul")
            self._converted(graph, oh.make_operatorsetid("", 16), 15)

        self.assertRaises(RuntimeError, test)

    def test_scatter_nd_16_15_reduction_add_fails(self) -> None:
        def test() -> None:
            graph = self._make_scatter_graph("ScatterND", onnxl.TensorProto.FLOAT, "add")
            self._converted(graph, oh.make_operatorsetid("", 16), 15)

        self.assertRaises(RuntimeError, test)

    def test_scatter_nd_16_15_reduction_mul_fails(self) -> None:
        def test() -> None:
            graph = self._make_scatter_graph("ScatterND", onnxl.TensorProto.FLOAT, "mul")
            self._converted(graph, oh.make_operatorsetid("", 16), 15)

        self.assertRaises(RuntimeError, test)

    # Scatter 18 -> 17: reduction "max" / "min" fails
    def test_scatter_elements_18_17_reduction_max_fails(self) -> None:
        def test() -> None:
            graph = self._make_scatter_graph("ScatterElements", onnxl.TensorProto.FLOAT, "max")
            self._converted(graph, oh.make_operatorsetid("", 18), 17)

        self.assertRaises(RuntimeError, test)

    def test_scatter_elements_18_17_reduction_min_fails(self) -> None:
        def test() -> None:
            graph = self._make_scatter_graph("ScatterElements", onnxl.TensorProto.FLOAT, "min")
            self._converted(graph, oh.make_operatorsetid("", 18), 17)

        self.assertRaises(RuntimeError, test)

    def test_scatter_nd_18_17_reduction_max_fails(self) -> None:
        def test() -> None:
            graph = self._make_scatter_graph("ScatterND", onnxl.TensorProto.FLOAT, "max")
            self._converted(graph, oh.make_operatorsetid("", 18), 17)

        self.assertRaises(RuntimeError, test)

    def test_scatter_nd_18_17_reduction_min_fails(self) -> None:
        def test() -> None:
            graph = self._make_scatter_graph("ScatterND", onnxl.TensorProto.FLOAT, "min")
            self._converted(graph, oh.make_operatorsetid("", 18), 17)

        self.assertRaises(RuntimeError, test)

    # Scatter 18 -> 17: allowed reductions succeed
    def test_scatter_elements_18_17_no_reduction_success(self) -> None:
        graph = self._make_scatter_graph("ScatterElements", onnxl.TensorProto.FLOAT)
        converted = self._converted(graph, oh.make_operatorsetid("", 18), 17)
        pychecker.check_model(converted)

    def test_scatter_elements_18_17_none_reduction_success(self) -> None:
        graph = self._make_scatter_graph("ScatterElements", onnxl.TensorProto.FLOAT, "none")
        converted = self._converted(graph, oh.make_operatorsetid("", 18), 17)
        pychecker.check_model(converted)

    def test_scatter_elements_18_17_add_reduction_success(self) -> None:
        graph = self._make_scatter_graph("ScatterElements", onnxl.TensorProto.FLOAT, "add")
        converted = self._converted(graph, oh.make_operatorsetid("", 18), 17)
        pychecker.check_model(converted)

    def test_scatter_elements_18_17_mul_reduction_success(self) -> None:
        graph = self._make_scatter_graph("ScatterElements", onnxl.TensorProto.FLOAT, "mul")
        converted = self._converted(graph, oh.make_operatorsetid("", 18), 17)
        pychecker.check_model(converted)

    def test_scatter_nd_18_17_no_reduction_success(self) -> None:
        graph = self._make_scatter_graph("ScatterND", onnxl.TensorProto.FLOAT)
        converted = self._converted(graph, oh.make_operatorsetid("", 18), 17)
        pychecker.check_model(converted)

    def test_scatter_nd_18_17_none_reduction_success(self) -> None:
        graph = self._make_scatter_graph("ScatterND", onnxl.TensorProto.FLOAT, "none")
        converted = self._converted(graph, oh.make_operatorsetid("", 18), 17)
        pychecker.check_model(converted)

    def test_scatter_nd_18_17_add_reduction_success(self) -> None:
        graph = self._make_scatter_graph("ScatterND", onnxl.TensorProto.FLOAT, "add")
        converted = self._converted(graph, oh.make_operatorsetid("", 18), 17)
        pychecker.check_model(converted)

    def test_scatter_nd_18_17_mul_reduction_success(self) -> None:
        graph = self._make_scatter_graph("ScatterND", onnxl.TensorProto.FLOAT, "mul")
        converted = self._converted(graph, oh.make_operatorsetid("", 18), 17)
        pychecker.check_model(converted)

    # raw_data INT64 initializers + dims/raw byte-length mismatch guard
    def test_split_13_12_raw_data_initializer(self) -> None:
        split_init = oh.make_tensor(
            "split", onnxl.TensorProto.INT64, [2], np.array([1, 3], dtype=np.int64), raw=True
        )
        graph = oh.make_graph(
            [oh.make_node("Split", ["X", "split"], ["Y0", "Y1"], axis=0)],
            "g",
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, (4, 2))],
            [
                oh.make_tensor_value_info("Y0", onnxl.TensorProto.FLOAT, (1, 2)),
                oh.make_tensor_value_info("Y1", onnxl.TensorProto.FLOAT, (3, 2)),
            ],
            initializer=[split_init],
        )
        converted = self._converted(graph, oh.make_operatorsetid("", 13), 12)
        attr = next(a for a in converted.graph.node[0].attribute if a.name == "split")
        assert list(attr.ints) == [1, 3]

    def test_reshape_5_4_raw_data_initializer(self) -> None:
        shape_init = oh.make_tensor(
            "shape", onnxl.TensorProto.INT64, [2], np.array([2, 5], dtype=np.int64), raw=True
        )
        graph = oh.make_graph(
            [oh.make_node("Reshape", ["X", "shape"], ["Y"])],
            "g",
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, (10,))],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, (2, 5))],
            initializer=[shape_init],
        )
        converted = self._converted(graph, oh.make_operatorsetid("", 5), 4)
        reshape = next(n for n in converted.graph.node if n.op_type == "Reshape")
        attr = next(a for a in reshape.attribute if a.name == "shape")
        assert list(attr.ints) == [2, 5]

    def test_axes_input_to_attr_13_12_raw_data(self) -> None:
        axes_init = oh.make_tensor(
            "axes", onnxl.TensorProto.INT64, [1], np.array([0], dtype=np.int64), raw=True
        )
        graph = oh.make_graph(
            [oh.make_node("ReduceSum", ["X", "axes"], ["Y"])],
            "g",
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, (5, 5))],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, (1, 5))],
            initializer=[axes_init],
        )
        converted = self._converted(graph, oh.make_operatorsetid("", 13), 12)
        attr = next(a for a in converted.graph.node[0].attribute if a.name == "axes")
        assert list(attr.ints) == [0]

    def test_split_13_12_raw_data_dims_mismatch_rejected(self) -> None:
        # dims=[2] vs 8-byte raw_data fed via a Constant node; old code OOB-read.
        split_value = onnxl.TensorProto()
        split_value.data_type = onnxl.TensorProto.INT64
        split_value.dims.extend([2])
        split_value.raw_data = struct.pack("<q", 1)
        graph = oh.make_graph(
            [
                oh.make_node("Constant", [], ["split"], value=split_value),
                oh.make_node("Split", ["X", "split"], ["Y0", "Y1"], axis=0),
            ],
            "g",
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, (4, 2))],
            [
                oh.make_tensor_value_info("Y0", onnxl.TensorProto.FLOAT, (1, 2)),
                oh.make_tensor_value_info("Y1", onnxl.TensorProto.FLOAT, (3, 2)),
            ],
        )
        # Bypass checker to exercise the converter guard directly.
        model = oh.make_model(graph, opset_imports=[oh.make_operatorsetid("", 13)])
        with self.assertRaises(RuntimeError):
            version_converter.convert_version(model, 12)

    def test_split_13_12_raw_data_constant_node(self) -> None:
        split_value = oh.make_tensor(
            "", onnxl.TensorProto.INT64, [2], np.array([1, 3], dtype=np.int64), raw=True
        )
        graph = oh.make_graph(
            [
                oh.make_node("Constant", [], ["split"], value=split_value),
                oh.make_node("Split", ["X", "split"], ["Y0", "Y1"], axis=0),
            ],
            "g",
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, (4, 2))],
            [
                oh.make_tensor_value_info("Y0", onnxl.TensorProto.FLOAT, (1, 2)),
                oh.make_tensor_value_info("Y1", onnxl.TensorProto.FLOAT, (3, 2)),
            ],
        )
        converted = self._converted(graph, oh.make_operatorsetid("", 13), 12)
        split = next(n for n in converted.graph.node if n.op_type == "Split")
        attr = next(a for a in split.attribute if a.name == "split")
        assert list(attr.ints) == [1, 3]


if __name__ == "__main__":
    unittest.main(verbosity=2)
