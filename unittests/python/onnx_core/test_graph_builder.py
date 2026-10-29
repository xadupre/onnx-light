# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Python :class:`onnx_light.onnx_core.graph_builder.GraphBuilder`.

These exercise the incremental builder API directly (naming, node creation,
opset resolution, incremental shape inference, initializers, nested builders and
finalisation). They mirror the C++ counterpart in
``unittests/cc/onnx_core/builder/test_graph_builder.cc``.
"""

from __future__ import annotations

import gc
import unittest

import numpy

import onnx_light.onnx as onnxl
import onnx_light.onnx.checker as checker
import onnx_light.onnx.helper as oh
from onnx_light.onnx.defs import onnx_opset_version
from onnx_light.ext_test_case import ExtTestCase, import_or_skip

# The builder extension ships in the core build; skip the module when it (or the
# schema library it relies on) is unavailable.
GraphBuilder = import_or_skip("onnx_light.onnx_core.graph_builder", "GraphBuilder")
ProtoKind = import_or_skip("onnx_light.onnx_core.graph_builder", "ProtoKind")
OperatorRegistry = import_or_skip("onnx_light.onnx_core.graph_builder", "OperatorRegistry")

FLOAT = onnxl.TensorProto.FLOAT


class TestGraphBuilder(ExtTestCase):
    def test_unavailable_global_schema_rejects_before_mutation(self):
        builder = GraphBuilder("g", opsets={"ai.onnx.ml": 4})
        with self.assertRaisesRegex(ValueError, "unavailable at opset 4"):
            builder.op.TreeEnsemble(
                numpy.ones((2, 3), dtype=numpy.float32), domain="ai.onnx.ml", outputs="y"
            )
        self.assertEqual(len(builder.build_graph().node), 0)
        self.assertEqual(len(builder.build_graph().initializer), 0)
        self.assertFalse(builder.has_name("y"))
        builder.inp("x", FLOAT, [2, 3])
        with self.assertRaisesRegex(ValueError, "unavailable at opset 4"):
            builder.make_node("TreeEnsemble", ["x"], ["y"], domain="ai.onnx.ml")
        self.assertFalse(builder.has_name("y"))
        builder.set_opset_version("ai.onnx.ml", 5)
        self.assertEqual(builder.find_schema("TreeEnsemble", "ai.onnx.ml").since_version, 5)
        other = GraphBuilder("other", opsets={"custom": 4})
        self.assertIsNone(other.find_schema("TreeEnsemble", "custom"))

    def test_import_slice_opset_histories(self):
        for opset in (10, 11, 12, 13):
            for with_steps in (False, True):
                with self.subTest(opset=opset, with_steps=with_steps):
                    parameters = [("starts", 1), ("ends", 5)]
                    if with_steps:
                        parameters.extend([("axes", 0), ("steps", 2)])
                    model = oh.make_model(
                        oh.make_graph(
                            [
                                oh.make_node(
                                    "Slice", ["x"] + [name for name, _ in parameters], ["y"]
                                )
                            ],
                            "slice",
                            [oh.make_tensor_value_info("x", FLOAT, [6])],
                            [oh.make_tensor_value_info("y", FLOAT, [2 if with_steps else 4])],
                            initializer=[
                                oh.make_tensor(name, onnxl.TensorProto.INT64, [1], [value])
                                for name, value in parameters
                            ],
                        ),
                        opset_imports=[oh.make_opsetid("", opset)],
                    )
                    checker.check_model(model)
                    builder = GraphBuilder(model)
                    self.assertEqual(
                        builder.find_schema("Slice").since_version,
                        10 if opset == 10 else 11 if opset < 13 else 13,
                    )
                    self.assertEqual(list(builder.get_shape("y").shape), [2 if with_steps else 4])
                    checker.check_model(builder.to_onnx())

    def test_local_shape_registry_precedes_global_and_is_snapshotted(self):
        calls = []

        def infer(ctx, node):
            calls.append(node.name)
            ctx.set(str(node.output[0]), ctx.get(str(node.input[0])))

        registry = OperatorRegistry()
        registry.register_shape_inference("", "Neg", infer)
        builder = GraphBuilder("g", opsets=18, local_registry=registry)
        x = builder.inp("x", FLOAT, [2, 3])
        builder.op.Neg(x, outputs="y", name="local")
        builder.op.Abs("y", outputs="z")
        builder.out("z")
        self.assertIn("local", calls)
        self.assertEqual(list(builder.get_shape("z").shape), [2, 3])
        before = len(calls)
        isolated = GraphBuilder("isolated", opsets=18)
        isolated.inp("x", FLOAT, [2, 3])
        isolated.op.Neg("x", outputs="y")
        self.assertEqual(len(calls), before)

        def failing(ctx, node):
            raise ValueError("local shape failure")

        registry.register_shape_inference("ai.onnx", "Neg", failing)
        builder.op.Neg("z", outputs="snapshot")
        failing_builder = GraphBuilder("failing", opsets=18, local_registry=registry)
        failing_builder.inp("x", FLOAT, [2, 3])
        with self.assertRaisesRegex(ValueError, "local shape failure"):
            failing_builder.op.Neg("x", outputs="y")
        del registry, infer

        imported = GraphBuilder(builder.to_onnx(), local_registry=builder.local_registry())
        self.assertEqual(list(imported.get_shape("z").shape), [2, 3])
        optimization = import_or_skip("onnx_light.onnx_core.graph_builder", "OptimizationMode")
        before = len(calls)
        imported.to_onnx(optimize=optimization.kOptimize)
        self.assertGreater(len(calls), before)

    def test_local_schema_registry_is_used_by_compact_authoring(self):
        from onnx_light.onnx_op import FormalParameter, LightOpSchema

        parameter = FormalParameter()
        parameter.name = "X"
        parameter.type = "tensor(float)"
        registry = OperatorRegistry()
        registry.register_schema(
            LightOpSchema("LocalOp", "custom", 1, "", [parameter], [parameter, parameter], [])
        )

        def infer(ctx, node):
            for output in node.output:
                ctx.set(str(output), ctx.get(str(node.input[0])))

        registry.register_shape_inference("custom", "LocalOp", infer)
        builder = GraphBuilder("g", opsets={"": 18, "custom": 1}, local_registry=registry)
        x = builder.inp("x", FLOAT, [2, 3])
        with self.assertRaisesRegex(ValueError, "Unknown attribute"):
            builder.op.LocalOp(x, domain="custom", bogus=1)
        with self.assertRaisesRegex(ValueError, "outputs"):
            builder.make_node("LocalOp", [x], ["bad"], domain="custom")
        self.assertFalse(builder.has_name("bad"))
        outputs = builder.op.LocalOp(x, domain="custom")
        self.assertEqual(len(outputs), 2)
        for output in outputs:
            self.assertEqual(list(builder.get_shape(output).shape), [2, 3])

        registry.register_schema(
            LightOpSchema("LocalOp", "custom", 1, "", [parameter], [parameter], [])
        )
        self.assertEqual(builder.find_schema("LocalOp", "custom").max_output, 2)
        future = GraphBuilder("future", opsets={"custom": 1}, local_registry=registry)
        self.assertEqual(future.find_schema("LocalOp", "custom").max_output, 1)

        function = oh.make_function(
            "custom",
            "LocalOp",
            ["a"],
            ["b"],
            [oh.make_node("Identity", ["a"], ["b"])],
            [oh.make_opsetid("", 18)],
            attributes=["alpha"],
        )
        value = oh.make_tensor_value_info("x", FLOAT, [2, 3])
        model = oh.make_model(
            oh.make_graph([], "functions", [value], [value]),
            functions=[function],
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("custom", 1)],
        )
        imported = GraphBuilder(model, local_registry=builder.local_registry())
        self.assertTrue(imported.has_local_function("LocalOp", "custom"))
        self.assertFalse(imported.has_local_function("LocalOp", "other"))
        with self.assertRaisesRegex(ValueError, "Unknown attribute"):
            imported.op.LocalOp("x", domain="custom", alpha=1)
        with self.assertRaisesRegex(ValueError, "outputs"):
            imported.op.LocalOp("x", domain="custom", outputs="result")
        outputs = imported.op.LocalOp("x", domain="custom")
        self.assertEqual(len(outputs), 2)
        for output in outputs:
            self.assertEqual(list(imported.get_shape(output).shape), [2, 3])

    def test_future_schemas_preserve_local_function_fallback(self):
        from onnx_light.onnx_op import (
            FormalParameter,
            GetAllOnnxOpSchemasWithHistory,
            LightOpSchema,
        )

        parameter = FormalParameter()
        parameter.name = "X"
        parameter.type = "tensor(float)"
        schema = LightOpSchema(
            "FutureF", "custom", 2, "", [parameter], [parameter, parameter], []
        )
        for local_schema in (False, True):
            with self.subTest(local_schema=local_schema):
                registry = OperatorRegistry()
                if local_schema:
                    registry.register_schema(schema)
                    lookup = GetAllOnnxOpSchemasWithHistory
                else:

                    def lookup(name):
                        return (
                            [schema]
                            if name == "FutureF"
                            else GetAllOnnxOpSchemasWithHistory(name, False)
                        )

                unavailable = GraphBuilder(
                    "unavailable", lookup, opsets={"": 18, "custom": 1}, local_registry=registry
                )
                unavailable.inp("x", FLOAT, [2])
                with self.assertRaisesRegex(ValueError, "unavailable at opset 1"):
                    unavailable.op.FutureF("x", domain="custom")
                function = oh.make_function(
                    "custom",
                    "FutureF",
                    ["A"],
                    ["B"],
                    [oh.make_node("Identity", ["A"], ["B"])],
                    [oh.make_opsetid("", 18)],
                    attributes=["alpha"],
                )
                value = oh.make_tensor_value_info("x", FLOAT, [2])
                model = oh.make_model(
                    oh.make_graph([], "future", [value], [value]),
                    functions=[function],
                    opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("custom", 1)],
                )
                builder = GraphBuilder(model, lookup, local_registry=registry)
                self.assertIsNone(builder.find_schema("FutureF", "custom"))
                output = builder.op.FutureF("x", domain="custom", alpha=1)
                self.assertIsInstance(output, str)
                self.assertEqual(list(builder.get_shape(output).shape), [2])
                builder.make_node(
                    "FutureF", ["x"], ["direct"], domain="custom", attributes={"alpha": 1}
                )
                self.assertEqual(list(builder.get_shape("direct").shape), [2])
                checker.check_model(builder.to_onnx())

    def test_missing_opset_rejects_node_before_mutation(self):
        for with_schema in (False, True):
            builder = (
                GraphBuilder("g", opsets={})
                if with_schema
                else GraphBuilder("g", None, opsets={})
            )
            x = builder.inp("x", FLOAT, [2])
            with self.assertRaisesRegex(ValueError, "opset"):
                builder.make_node("Relu", [x], ["y"])
            with self.assertRaisesRegex(ValueError, "opset"):
                builder.op.Add(x, numpy.array([1, 2], dtype=numpy.float32), outputs="y")
            self.assertEqual(len(builder.build_graph().node), 0)
            self.assertEqual(len(builder.build_graph().initializer), 0)
            self.assertFalse(builder.has_name("y"))
            builder.set_opset_version("", 18)
            builder.make_node("Relu", [x], ["y"])
            with self.assertRaisesRegex(ValueError, "opset"):
                builder.make_node("Custom", ["y"], ["z"], domain="custom")
            self.assertFalse(builder.has_name("z"))
            builder.set_opset_version("custom", 1)
            builder.make_node("Custom", ["y"], ["z"], domain="custom")
            self.assertEqual(len(builder.build_graph().node), 2)

    def test_rejects_nonpositive_opsets(self):
        builder = GraphBuilder(opsets=18)
        for version in (0, -1):
            with self.assertRaises(ValueError):
                builder.set_opset_version("", version)
            self.assertEqual(builder.opset_version(""), 18)

    def test_constructor_opsets(self):
        for opsets in (10, {"": 10, "com.example": 7}, {"ai.onnx": 10}):
            with self.subTest(opsets=opsets):
                builder = GraphBuilder("g", opsets=opsets)
                self.assertEqual(builder.opset_version(""), 10)
                self.assertEqual(builder.opset_version("ai.onnx"), 10)
                self.assertEqual(builder.shapes.opset_version(""), 10)
                x = builder.inp("x", FLOAT, [2])
                y = builder.op.Clip(x, outputs="y", min=0.0, max=1.0)
                builder.out(y)
                model = builder.to_onnx(ProtoKind.kModel)
                checker.check_model(model)
                imports = {
                    opset.domain or "ai.onnx": opset.version for opset in model.opset_import
                }
                self.assertEqual(imports["ai.onnx"], 10)
                if isinstance(opsets, dict) and "com.example" in opsets:
                    self.assertEqual(imports["com.example"], 7)
                builder.set_opset_version("", 11)
                self.assertEqual(builder.opset_version(""), 11)
                self.assertRaise(lambda: builder.op.Clip(x, min=0.0), ValueError)

    def test_constructor_default_opsets(self):
        latest = onnx_opset_version()
        for with_schema in (False, True):
            for kwargs in ({}, {"opsets": None}):
                with self.subTest(with_schema=with_schema, kwargs=kwargs):
                    builder = (
                        GraphBuilder("g", **kwargs)
                        if with_schema
                        else GraphBuilder("g", None, **kwargs)
                    )
                    self.assertEqual(builder.opset_version(""), latest)
                    self.assertEqual(builder.opset_version("ai.onnx"), latest)
                    self.assertEqual(builder.shapes.opset_version(""), latest)
                    x = builder.inp("x", FLOAT, [2])
                    builder.make_node("Relu", [x], ["y"])
                    builder.out(builder.op.Identity("y", outputs="z"))
                    model = builder.to_onnx()
                    checker.check_model(model)
                    self.assertEqual(
                        [
                            (opset.domain or "ai.onnx", opset.version)
                            for opset in model.opset_import
                        ],
                        [("ai.onnx", latest)],
                    )
                    with self.assertRaisesRegex(ValueError, "opset"):
                        builder.make_node("Custom", ["z"], ["custom"], domain="custom")
                    self.assertFalse(builder.has_name("custom"))

    def test_constructor_opsets_without_schema(self):
        builder = GraphBuilder("g", None, opsets={"": 18, "com.example": 7})
        x = builder.inp("x", FLOAT, [2])
        y = builder.op.CustomOp(x, domain="com.example")
        builder.out(y)
        self.assertEqual(builder.opset_version("com.example"), 7)
        self.assertEqual(builder.to_onnx(ProtoKind.kModel).graph.node[0].domain, "com.example")

    def test_constructor_rejects_invalid_opsets(self):
        for opsets in ("18", {"": "18"}, {"": 1 << 64}):
            with self.subTest(opsets=opsets):
                self.assertRaise(lambda: GraphBuilder(opsets=opsets), TypeError)

    def test_compact_standard_roundtrip(self):
        builder = GraphBuilder("g", opsets=28)
        self.assertIs(builder.op, builder.op)
        x = builder.inp("x", FLOAT, [None, 3])
        bias = builder.init(numpy.array([1, 2, 3], dtype=numpy.float32), name="bias")
        added = builder.op.Add(x, bias)
        output = builder.op.Relu(added, outputs="y", name="relu")
        self.assertEqual(builder.out(output, FLOAT, [None, 3]), "y")

        model = builder.to_onnx()
        checker.check_model(model)
        roundtrip = onnxl.ModelProto()
        roundtrip.ParseFromString(model.SerializeToString())
        self.assertEqual(roundtrip.SerializeToString(), model.SerializeToString())
        self.assertEqual([node.op_type for node in model.graph.node], ["Add", "Relu"])
        self.assertEqual(model.graph.node[1].name, "relu")

    def test_if_capture_model_passes_checker(self):
        branch = oh.make_graph(
            [oh.make_node("Identity", ["x"], ["local_result"])],
            "branch",
            [],
            [oh.make_tensor_value_info("local_result", FLOAT, [2])],
        )
        builder = GraphBuilder("valid_if_capture")
        builder.set_opset_version("", 22)
        condition = builder.inp("condition", onnxl.TensorProto.BOOL, [])
        builder.inp("x", FLOAT, [2])
        result = builder.op.If(
            condition, outputs="result", then_branch=branch, else_branch=branch
        )
        builder.out(result, FLOAT, [2])
        model = builder.to_onnx()
        checker.check_model(model)

    def test_compact_optional_variadic_and_numpy_inputs(self):
        builder = GraphBuilder("g", opsets=28)
        x = builder.inp("x", FLOAT, [3])
        clipped = builder.op.Clip(x, None, numpy.array([1], dtype=numpy.float32))
        summed = builder.op.Sum(clipped, x, x)
        builder.out(summed)
        graph = builder.to_onnx(ProtoKind.kGraph)
        self.assertEqual(list(graph.node[0].input)[:2], ["x", ""])
        self.assertEqual(len(graph.initializer), 1)
        self.assertEqual(list(graph.node[1].input), [clipped, "x", "x"])

    def test_compact_multi_output(self):
        builder = GraphBuilder("g", opsets=28)
        x = builder.inp("x", FLOAT, [4])
        k = builder.init(numpy.array([2], dtype=numpy.int64))
        values, indices = builder.op.TopK(x, k, outputs=["values", "indices"], axis=0)
        self.assertEqual((values, indices), ("values", "indices"))
        generated = builder.op.TopK(x, k, outputs=2)
        self.assertEqual(len(generated), 2)

    def test_layer_normalization_optional_outputs(self):
        for output_count in (1, 2, 3):
            with self.subTest(output_count=output_count):
                outputs = ["Y", "Mean", "InvStdDev"][:output_count]
                model = oh.make_model(
                    oh.make_graph(
                        [oh.make_node("LayerNormalization", ["X", "Scale"], outputs)],
                        "layer_normalization",
                        [
                            oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT16, [1, 3]),
                            oh.make_tensor_value_info("Scale", onnxl.TensorProto.FLOAT16, [3]),
                        ],
                        [
                            oh.make_tensor_value_info(
                                name,
                                onnxl.TensorProto.FLOAT16 if name == "Y" else FLOAT,
                                [1, 3] if name == "Y" else [1, 1],
                            )
                            for name in outputs
                        ],
                    ),
                    opset_imports=[oh.make_opsetid("", 17)],
                )
                rebuilt = GraphBuilder(model).to_onnx()
                checker.check_model(rebuilt)
                self.assertEqual(list(rebuilt.graph.node[0].output), outputs)
                self.assertEqual([value.name for value in rebuilt.graph.output], outputs)

    def test_compact_custom_operator_without_schema(self):
        builder = GraphBuilder("g", opsets=28)
        builder.set_opset_version("com.example", 7)
        x = builder.inp("x", FLOAT, [2])
        outputs = builder.op.CustomOp(
            x, domain="com.example", outputs=["left", "right"], name="custom", alpha=1.5
        )
        self.assertEqual(outputs, ("left", "right"))
        model = builder.to_onnx()
        node = model.graph.node[0]
        self.assertEqual(
            (node.op_type, node.domain, node.name), ("CustomOp", "com.example", "custom")
        )
        self.assertEqual(list(node.input), ["x"])
        self.assertEqual(list(node.output), ["left", "right"])
        self.assertEqual(node.attribute[0].name, "alpha")
        self.assertEqual(
            {opset.domain: opset.version for opset in model.opset_import}["com.example"], 7
        )

    def test_compact_documented_optimization_workflow(self):
        optimization = import_or_skip("onnx_light.onnx_core.optimization")
        source = GraphBuilder("optimization")
        source.set_opset_version("", 18)
        x = source.inp("X", FLOAT, [4])
        y = source.op.Cast(x, outputs="Y", to=FLOAT)
        source.out(y, FLOAT, [4])
        original = source.to_onnx()

        builder = GraphBuilder(original)
        graph = optimization.GraphGraph(builder, optimization.standard_patterns(["Cast"]))
        rewrites, report = graph.optimize(report=True)
        optimized_graph = builder.build_graph()
        replayed_graph = optimization.apply_rewritings(original, rewrites)

        self.assertEqual(len(rewrites.rewrites), 1)
        self.assertEqual(report.rewrites, 1)
        self.assertEqual(rewrites.rewrites[0].pattern_name, "Cast")
        self.assertEqual(replayed_graph.SerializeToString(), optimized_graph.SerializeToString())
        checker.check_model(builder.to_onnx())

    def test_compact_imported_opset_attribute_validation(self):
        graph = oh.make_graph([], "g", [oh.make_tensor_value_info("x", FLOAT, [2])], [])
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 10)])
        builder = GraphBuilder(model)
        self.assertEqual(builder.opset_version(""), 10)
        output = builder.op.Clip("x", outputs="y", min=0.0, max=1.0)
        self.assertEqual(output, "y")

    def test_compact_rejects_invalid_calls(self):
        builder = GraphBuilder("g", opsets=28)
        x = builder.inp("x", FLOAT, [2])
        self.assertRaise(lambda: builder.op.UnknownOperator(x), ValueError)
        self.assertRaise(lambda: builder.op.Relu(x, unknown_attribute=1), ValueError)
        self.assertRaise(
            lambda: builder.op.CustomOp(x, domain="com.missing", outputs="y"), ValueError
        )
        self.assertRaise(lambda: builder.op.Relu(1), TypeError)
        self.assertRaise(lambda: builder.op.Relu(x, outputs=0), ValueError)
        self.assertRaise(lambda: builder.init([1, 2]), TypeError)
        self.assertRaise(lambda: builder.inp("x", FLOAT, [2]), ValueError)
        self.assertRaise(lambda: builder.init(numpy.array([1]), name="x"), ValueError)
        self.assertRaise(lambda: builder.out(x, shape=[2]), ValueError)

    def test_starts_empty(self):
        builder = GraphBuilder("g")
        self.assertFalse(builder.has_name("x"))

    def test_names_are_never_reused(self):
        builder = GraphBuilder("g")
        builder.make_input("x", FLOAT, [2, 3])
        self.assertTrue(builder.has_name("x"))
        self.assertRaise(lambda: builder.make_input("x", FLOAT, [2, 3]), ValueError)

    def test_unique_name_never_collides(self):
        builder = GraphBuilder("g")
        a = builder.unique_name("t")
        b = builder.unique_name("t")
        self.assertNotEqual(a, b)
        self.assertTrue(builder.has_name(a))
        self.assertTrue(builder.has_name(b))

    def test_make_node_uses_explicit_opset_and_infers_shape(self):
        builder = GraphBuilder("g", opsets=28)
        builder.make_input("x", FLOAT, [2, 3])
        builder.make_input("y", FLOAT, [2, 3])
        outputs = builder.make_node("Add", ["x", "y"])
        self.assertEqual(len(outputs), 1)
        self.assertTrue(builder.has_name(outputs[0]))
        self.assertEqual(builder.opset_version(""), 28)
        # The output shape was inferred incrementally.
        self.assertTrue(builder.has_shape(outputs[0]))
        self.assertEqual(builder.get_shape(outputs[0]).shape.rank(), 2)

    def test_make_node_uses_provided_output_name(self):
        builder = GraphBuilder("g", opsets=28)
        builder.make_input("x", FLOAT, [2, 3])
        builder.make_input("y", FLOAT, [2, 3])
        outputs = builder.make_node("Add", ["x", "y"], ["z"])
        self.assertEqual(outputs, ["z"])

    def test_make_node_rejects_unknown_input(self):
        builder = GraphBuilder("g", opsets=28)
        self.assertRaise(lambda: builder.make_node("Add", ["missing", "y"]), ValueError)

    def test_make_node_with_attributes(self):
        builder = GraphBuilder("g", opsets=28)
        builder.make_input("x", FLOAT, [2, 3])
        outputs = builder.make_node("Softmax", ["x"], attributes={"axis": 1})
        self.assertEqual(len(outputs), 1)
        self.assertTrue(builder.has_shape(outputs[0]))

    def test_external_initializer_is_recorded(self):
        builder = GraphBuilder("g")
        name = builder.make_external_initializer("w", FLOAT, [4, 4], "weights.bin", 0, 64)
        self.assertEqual(name, "w")
        self.assertTrue(builder.has_name("w"))

    def test_make_initializer_move(self):
        for raw in (False, True):
            with self.subTest(raw=raw):
                values = numpy.arange(1024, dtype=numpy.float32)
                tensor = oh.make_tensor(
                    "weight", FLOAT, [1024], values.tobytes() if raw else values, raw=raw
                )
                serialized = tensor.SerializeToString()
                copying = GraphBuilder("copy")
                self.assertEqual(copying.make_initializer(tensor), "weight")
                self.assertEqual(tensor.SerializeToString(), serialized)

                builder = GraphBuilder("move")
                same_source = tensor
                self.assertEqual(builder.make_initializer_move(tensor), "weight")
                self.assertEqual(
                    same_source.SerializeToString(), onnxl.TensorProto().SerializeToString()
                )
                self.assertEqual(
                    builder.to_graph().initializer[0].SerializeToString(),
                    copying.to_graph().initializer[0].SerializeToString(),
                )
                self.assertEqual(builder.get_shape("weight").shape.rank(), 1)
                tensor.CopyFrom(oh.make_tensor("reused", FLOAT, [1], [1.0]))
                self.assertEqual(builder.make_initializer_move(tensor), "reused")
                self.assertEqual(
                    tensor.SerializeToString(), onnxl.TensorProto().SerializeToString()
                )

    def test_make_initializer_move_external_data(self):
        tensor = onnxl.TensorProto()
        tensor.name = "external"
        tensor.data_type = FLOAT
        tensor.dims.extend([4])
        tensor.data_location = onnxl.TensorProto.EXTERNAL
        for key, value in (
            ("location", "missing-weights.bin"),
            ("offset", "16"),
            ("length", "16"),
        ):
            entry = tensor.external_data.add()
            entry.key, entry.value = key, value
        serialized = tensor.SerializeToString()
        builder = GraphBuilder("g")
        self.assertEqual(builder.make_initializer_move(tensor), "external")
        exported = builder.to_graph().initializer[0]
        exported.ClearField("metadata_props")
        self.assertEqual(exported.SerializeToString(), serialized)
        self.assertEqual(tensor.SerializeToString(), onnxl.TensorProto().SerializeToString())

    def test_make_initializer_move_overridable_default(self):
        builder = GraphBuilder("g")
        builder.make_input("x", FLOAT, ["batch", 2])
        tensor = oh.make_tensor("x", FLOAT, [3, 2], [1.0] * 6)
        self.assertEqual(builder.make_initializer_move(tensor), "x")
        self.assertEqual(tensor.SerializeToString(), onnxl.TensorProto().SerializeToString())
        graph = builder.to_graph()
        self.assertEqual(graph.input[0].type.tensor_type.shape.dim[0].dim_param, "batch")
        self.assertEqual(list(graph.initializer[0].dims), [3, 2])

    def test_make_initializer_move_borrowed_owner(self):
        serialized = oh.make_tensor(
            "weight", FLOAT, [32], numpy.zeros(32, dtype=numpy.float32).tobytes(), raw=True
        ).SerializeToString()
        released = []

        def callback(tensor, graph, owner=serialized):
            def release():
                released.append(len(owner))

            return release

        options = onnxl.ParseOptions()
        options.no_copy = True
        options.raw_data_callback = callback
        tensor = onnxl.TensorProto()
        tensor.ParseFromString(serialized, options)
        builder = GraphBuilder("g")
        self.assertEqual(builder.make_initializer_move(tensor), "weight")
        self.assertEqual(tensor.SerializeToString(), onnxl.TensorProto().SerializeToString())
        del tensor, serialized, options, callback
        gc.collect()
        self.assertEqual(released, [])
        self.assertEqual(builder.to_graph().initializer[0].raw_data, bytes(128))
        del builder
        gc.collect()
        self.assertEqual(len(released), 1)

    def test_make_initializer_move_validation_preserves_source(self):
        builder = GraphBuilder("g")
        builder.make_input("x", FLOAT, [2])
        tensors = [
            oh.make_tensor("x", onnxl.TensorProto.INT64, [2], [1, 2]),
            oh.make_tensor("x", FLOAT, [1, 2], [1.0, 2.0]),
            oh.make_tensor("x", FLOAT, [3], [1.0, 2.0, 3.0]),
            oh.make_tensor("", FLOAT, [2], [1.0, 2.0]),
        ]
        for tensor in tensors:
            with self.subTest(tensor=str(tensor)):
                serialized = tensor.SerializeToString()
                with self.assertRaises(ValueError):
                    builder.make_initializer_move(tensor)
                self.assertEqual(tensor.SerializeToString(), serialized)
                self.assertEqual(len(builder.to_graph().initializer), 0)
        for name in ("x", "weight"):
            tensor = oh.make_tensor(name, FLOAT, [2], [1.0, 2.0])
            builder.make_initializer(tensor)
            serialized = tensor.SerializeToString()
            with self.assertRaises(ValueError):
                builder.make_initializer_move(tensor)
            self.assertEqual(tensor.SerializeToString(), serialized)

    def test_to_model_produces_graph_with_opset(self):
        builder = GraphBuilder("g", opsets=28)
        builder.make_input("x", FLOAT, [2, 3])
        builder.make_input("y", FLOAT, [2, 3])
        (z,) = builder.make_node("Add", ["x", "y"])
        builder.make_output(z)
        model = builder.to_onnx()
        self.assertIsInstance(model, onnxl.ModelProto)
        self.assertGreater(model.ir_version, 0)
        self.assertEqual(len(model.graph.node), 1)
        self.assertGreater(len(model.opset_import), 0)
        self.assertEqual(
            model.SerializeToString(), builder.to_onnx(kind=ProtoKind.kModel).SerializeToString()
        )
        self.assertEqual(builder.to_onnx(ir_version=8).ir_version, 8)
        self.assertEqual(builder.to_onnx(ProtoKind.kModel, ir_version=8).ir_version, 8)

    def test_to_graph_and_to_function(self):
        builder = GraphBuilder("g", opsets=28)
        builder.make_input("x", FLOAT, [2, 3])
        builder.make_input("y", FLOAT, [2, 3])
        (z,) = builder.make_node("Add", ["x", "y"])
        builder.make_output(z)
        graph = builder.to_onnx(ProtoKind.kGraph)
        self.assertIsInstance(graph, onnxl.GraphProto)
        self.assertEqual(len(graph.node), 1)
        self.assertEqual(graph.SerializeToString(), builder.to_graph().SerializeToString())
        function = builder.to_onnx(kind=ProtoKind.kFunction, domain="custom")
        self.assertIsInstance(function, onnxl.FunctionProto)
        self.assertEqual(len(function.node), 1)
        self.assertEqual(function.domain, "custom")
        self.assertEqual(
            function.SerializeToString(), builder.to_function(domain="custom").SerializeToString()
        )

    def test_subgraph_to_onnx_uses_proto_kind(self):
        builder = GraphBuilder("g", opsets=18)
        body = builder.make_subgraph("body")
        body.make_input("x", FLOAT, [2])
        (output,) = body.make_node("Identity", ["x"])
        body.make_output(output)
        self.assertIsInstance(body.to_onnx(), onnxl.ModelProto)
        self.assertIsInstance(body.to_onnx(ProtoKind.kGraph), onnxl.GraphProto)
        self.assertIsInstance(body.to_onnx(ProtoKind.kFunction), onnxl.FunctionProto)

    def test_to_function_materializes_external_initializer(self):
        builder = GraphBuilder("g")
        builder.make_external_initializer("w", FLOAT, [2, 2], "w.bin", 0, 16)
        builder.make_output("w")
        before = builder.to_graph().SerializeToString()
        function = builder.to_onnx(ProtoKind.kFunction, domain="custom")
        self.assertEqual(len(function.node), 1)
        self.assertEqual(function.node[0].op_type, "Constant")
        self.assertEqual(list(function.node[0].output), ["w"])
        tensor = function.node[0].attribute[0].t
        self.assertEqual(list(tensor.dims), [2, 2])
        self.assertEqual(
            {entry.key: entry.value for entry in tensor.external_data},
            {"location": "w.bin", "offset": "0", "length": "16"},
        )
        self.assertEqual(builder.to_graph().SerializeToString(), before)

    def test_to_function_rejects_overridable_initializer(self):
        builder = GraphBuilder("g")
        builder.make_input("w", FLOAT, [2])
        builder.make_initializer(oh.make_tensor("w", FLOAT, [2], [1, 2]))
        builder.make_output("w")
        with self.assertRaisesRegex(ValueError, "cannot preserve overridable initializer 'w'"):
            builder.to_onnx(ProtoKind.kFunction, domain="custom")

    def test_to_onnx_rejects_non_enum_kind(self):
        builder = GraphBuilder("g")
        for kind in ("model", "graph", "function", "unknown", "", 0, 1, -1, True, None):
            with self.subTest(kind=kind), self.assertRaises(TypeError):
                builder.to_onnx(kind)

    def test_explicit_opset_is_preserved(self):
        builder = GraphBuilder("g")
        builder.set_opset_version("", 17)
        builder.make_input("x", FLOAT, [2, 3])
        builder.make_input("y", FLOAT, [2, 3])
        builder.make_node("Add", ["x", "y"])
        self.assertEqual(builder.opset_version(""), 17)

    def test_input_output_from_proto(self):
        builder = GraphBuilder("g")
        vi = oh.make_tensor_value_info("x", FLOAT, [2, 3])
        builder.make_input(vi)
        self.assertTrue(builder.has_name("x"))
        self.assertTrue(builder.has_shape("x"))
        builder.make_output("x")

    def test_to_string_describes_content(self):
        builder = GraphBuilder("g", opsets=28)
        builder.make_input("x", FLOAT, [2, 3])
        builder.make_input("y", FLOAT, [2, 3])
        (z,) = builder.make_node("Add", ["x", "y"])
        builder.make_output(z)
        text = builder.to_string()
        self.assertIn("GraphBuilder(name=g)", text)
        self.assertIn("inputs (2)", text)
        self.assertIn("nodes (1)", text)
        self.assertIn("Add(", text)
        self.assertEqual(text, str(builder))

    def test_local_function_is_a_nested_builder(self):
        builder = GraphBuilder("g", opsets=28)
        builder.set_opset_version("custom", 1)
        fct = builder.make_local_function("MyFct", "custom")
        fct.make_input("a", FLOAT, [2, 3])
        fct.make_input("b", FLOAT, [2, 3])
        (out,) = fct.make_node("Add", ["a", "b"])
        fct.make_output(out)
        self.assertRaise(lambda: builder.make_local_function("MyFct"), ValueError)

        builder.make_input("x", FLOAT, [2, 3])
        builder.make_input("z", FLOAT, [2, 3])
        (top,) = builder.make_node("MyFct", ["x", "z"], domain="custom")
        builder.make_output(top)
        model = builder.to_onnx()
        self.assertEqual(len(model.functions), 1)
        self.assertEqual(model.functions[0].name, "MyFct")

    def test_subgraph_is_a_nested_builder(self):
        builder = GraphBuilder("g")
        body = builder.make_subgraph("body")
        body.make_input("a", FLOAT, [2, 3])
        self.assertRaise(lambda: builder.make_subgraph("body"), ValueError)

    def test_remove_unused_nodes(self):
        builder = GraphBuilder("g", opsets=28)
        builder.make_input("x", FLOAT, [2, 3])
        builder.make_input("y", FLOAT, [2, 3])
        (used,) = builder.make_node("Add", ["x", "y"])
        # Two chained dead-end nodes that no output depends on.
        (dead,) = builder.make_node("Mul", ["x", "y"])
        builder.make_node("Neg", [dead])
        builder.make_output(used)

        self.assertEqual(len(builder.build_graph().node), 3)
        self.assertEqual(builder.remove_unused_nodes(), 2)
        graph = builder.build_graph()
        self.assertEqual(len(graph.node), 1)
        self.assertEqual(graph.node[0].op_type, "Add")
        # A second pass has nothing left to remove.
        self.assertEqual(builder.remove_unused_nodes(), 0)

    def test_remove_duplicate_initializers(self):
        builder = GraphBuilder("g", opsets=28)
        builder.make_input("x", FLOAT, [2, 2])
        w1 = oh.make_tensor("w1", FLOAT, [2, 2], [1.0, 2.0, 3.0, 4.0])
        w2 = oh.make_tensor("w2", FLOAT, [2, 2], [1.0, 2.0, 3.0, 4.0])
        builder.make_initializer(w1)
        builder.make_initializer(w2)
        (a,) = builder.make_node("Add", ["x", "w1"])
        (b,) = builder.make_node("Add", ["x", "w2"])
        builder.make_output(a)
        builder.make_output(b)

        self.assertEqual(builder.remove_duplicate_initializers(), 1)
        graph = builder.build_graph()
        self.assertEqual(len(graph.initializer), 1)
        self.assertEqual(graph.initializer[0].name, "w1")
        # The reference to the dropped duplicate is rewritten to the survivor.
        self.assertEqual(graph.node[1].input[1], "w1")
        # A second pass has nothing left to collapse.
        self.assertEqual(builder.remove_duplicate_initializers(), 0)

    def test_remove_duplicate_initializers_across_scopes(self):
        builder = GraphBuilder("g", opsets=28)
        builder.make_input("x", FLOAT, [2, 2])
        w = oh.make_tensor("w", FLOAT, [2, 2], [1.0, 1.0, 1.0, 1.0])
        builder.make_initializer(w)
        (a,) = builder.make_node("Add", ["x", "w"])
        builder.make_output(a)

        # The subgraph declares its own initializer with the same content; it is
        # collapsed onto the enclosing "w" since a subgraph body sees the outer
        # scope.
        body = builder.make_subgraph("body")
        body.make_input("b", FLOAT, [2, 2])
        c = oh.make_tensor("c", FLOAT, [2, 2], [1.0, 1.0, 1.0, 1.0])
        body.make_initializer(c)
        (s,) = body.make_node("Add", ["b", "c"])
        body.make_output(s)

        self.assertEqual(builder.remove_duplicate_initializers(), 1)
        graph = builder.build_graph()
        self.assertEqual(len(graph.initializer), 1)
        self.assertEqual(graph.initializer[0].name, "w")
        body_graph = body.build_graph()
        self.assertEqual(len(body_graph.initializer), 0)
        self.assertEqual(body_graph.node[0].input[1], "w")

    def test_remove_identity_nodes(self):
        builder = GraphBuilder("g", opsets=28)
        builder.make_input("x", FLOAT, [2, 3])
        # A chain of two identities feeding a live consumer.
        (i1,) = builder.make_node("Identity", ["x"])
        (i2,) = builder.make_node("Identity", [i1])
        (used,) = builder.make_node("Neg", [i2])
        builder.make_output(used)

        self.assertEqual(len(builder.build_graph().node), 3)
        self.assertEqual(builder.remove_identity_nodes(), 2)
        graph = builder.build_graph()
        self.assertEqual(len(graph.node), 1)
        self.assertEqual(graph.node[0].op_type, "Neg")
        # The chain collapses onto the original producer.
        self.assertEqual(graph.node[0].input[0], "x")
        # A second pass has nothing left to remove.
        self.assertEqual(builder.remove_identity_nodes(), 0)

    def test_remove_identity_nodes_keeps_graph_output(self):
        builder = GraphBuilder("g", opsets=28)
        builder.make_input("x", FLOAT, [2, 3])
        # The identity output is a declared graph output, so it is kept.
        (out,) = builder.make_node("Identity", ["x"])
        builder.make_output(out)

        self.assertEqual(builder.remove_identity_nodes(), 0)
        graph = builder.build_graph()
        self.assertEqual(len(graph.node), 1)
        self.assertEqual(graph.node[0].op_type, "Identity")

    def test_remove_identity_nodes_recurses_into_subgraphs(self):
        builder = GraphBuilder("g", opsets=28)
        builder.make_input("x", FLOAT, [2, 3])
        (top,) = builder.make_node("Neg", ["x"])
        builder.make_output(top)

        body = builder.make_subgraph("body")
        body.make_input("a", FLOAT, [2, 3])
        (inner,) = body.make_node("Identity", ["a"])
        (body_used,) = body.make_node("Neg", [inner])
        body.make_output(body_used)

        self.assertEqual(builder.remove_identity_nodes(), 1)
        body_graph = body.build_graph()
        self.assertEqual(len(body_graph.node), 1)
        self.assertEqual(body_graph.node[0].op_type, "Neg")
        self.assertEqual(body_graph.node[0].input[0], "a")

    def test_remove_duplicate_nodes(self):
        builder = GraphBuilder("g", opsets=28)
        builder.make_input("x", FLOAT, [2, 3])
        # Two Neg nodes computing the same value feed a shared consumer.
        (a,) = builder.make_node("Neg", ["x"])
        (b,) = builder.make_node("Neg", ["x"])
        (s,) = builder.make_node("Add", [a, b])
        builder.make_output(s)

        self.assertEqual(builder.remove_duplicate_nodes(), 1)
        graph = builder.build_graph()
        self.assertEqual(len(graph.node), 2)
        self.assertEqual(graph.node[0].op_type, "Neg")
        self.assertEqual(graph.node[1].op_type, "Add")
        # Both Add inputs now read from the surviving Neg.
        self.assertEqual(graph.node[1].input[0], a)
        self.assertEqual(graph.node[1].input[1], a)
        # A second pass has nothing left to collapse.
        self.assertEqual(builder.remove_duplicate_nodes(), 0)

    def test_remove_duplicate_nodes_collapses_branches(self):
        builder = GraphBuilder("g", opsets=28)
        builder.make_input("x", FLOAT, [2, 3])
        # Two identical branches collapse level by level in a single pass.
        (a1,) = builder.make_node("Mul", ["x", "x"])
        (b1,) = builder.make_node("Mul", ["x", "x"])
        (a2,) = builder.make_node("Relu", [a1])
        (b2,) = builder.make_node("Relu", [b1])
        (s,) = builder.make_node("Add", [a2, b2])
        builder.make_output(s)

        self.assertEqual(builder.remove_duplicate_nodes(), 2)
        graph = builder.build_graph()
        self.assertEqual([n.op_type for n in graph.node], ["Mul", "Relu", "Add"])
        self.assertEqual(graph.node[2].input[0], a2)
        self.assertEqual(graph.node[2].input[1], a2)

    def test_remove_duplicate_nodes_keeps_distinct_attributes(self):
        builder = GraphBuilder("g", opsets=28)
        builder.make_input("x", FLOAT, [2, 3])
        (a,) = builder.make_node("LeakyRelu", ["x"], attributes={"alpha": 0.1})
        (b,) = builder.make_node("LeakyRelu", ["x"], attributes={"alpha": 0.2})
        (s,) = builder.make_node("Add", [a, b])
        builder.make_output(s)

        self.assertEqual(builder.remove_duplicate_nodes(), 0)
        self.assertEqual(len(builder.build_graph().node), 3)

    def test_remove_duplicate_nodes_keeps_graph_outputs(self):
        builder = GraphBuilder("g", opsets=28)
        builder.make_input("x", FLOAT, [2, 3])
        # Both duplicates are declared graph outputs, so neither can be dropped.
        (a,) = builder.make_node("Neg", ["x"])
        (b,) = builder.make_node("Neg", ["x"])
        builder.make_output(a)
        builder.make_output(b)

        self.assertEqual(builder.remove_duplicate_nodes(), 0)
        self.assertEqual(len(builder.build_graph().node), 2)

    def test_remove_duplicate_nodes_recurses_into_subgraphs(self):
        builder = GraphBuilder("g", opsets=28)
        builder.make_input("x", FLOAT, [2, 3])
        (top,) = builder.make_node("Neg", ["x"])
        builder.make_output(top)

        body = builder.make_subgraph("body")
        body.make_input("a", FLOAT, [2, 3])
        (n1,) = body.make_node("Neg", ["a"])
        (n2,) = body.make_node("Neg", ["a"])
        (s,) = body.make_node("Add", [n1, n2])
        body.make_output(s)

        self.assertEqual(builder.remove_duplicate_nodes(), 1)
        body_graph = body.build_graph()
        self.assertEqual(len(body_graph.node), 2)
        self.assertEqual(body_graph.node[1].op_type, "Add")
        self.assertEqual(body_graph.node[1].input[0], n1)
        self.assertEqual(body_graph.node[1].input[1], n1)

    def test_move_shape_and_size_nodes(self):
        builder = GraphBuilder("g", opsets=28)
        builder.make_input("x", FLOAT, [2, 3])
        (p,) = builder.make_node("Neg", ["x"])
        (a,) = builder.make_node("Add", [p, p])
        # Shape reads p but is inserted after an unrelated node; it must move
        # right after its producer, and the Size chained on it follows.
        (sh,) = builder.make_node("Shape", [p])
        (sz,) = builder.make_node("Size", [sh])
        builder.make_output(a)
        builder.make_output(sz)

        self.assertEqual(
            [n.op_type for n in builder.build_graph().node], ["Neg", "Add", "Shape", "Size"]
        )
        self.assertEqual(builder.move_shape_and_size_nodes(), 2)
        self.assertEqual(
            [n.op_type for n in builder.build_graph().node], ["Neg", "Shape", "Size", "Add"]
        )
        # A second pass leaves the already-tightened graph untouched.
        self.assertEqual(builder.move_shape_and_size_nodes(), 0)

    def test_move_shape_and_size_nodes_keeps_graph_input_reader(self):
        builder = GraphBuilder("g", opsets=28)
        builder.make_input("x", FLOAT, [2, 3])
        # Shape reads a graph input (no producing node), so it stays in place.
        (n,) = builder.make_node("Neg", ["x"])
        (sh,) = builder.make_node("Shape", ["x"])
        builder.make_output(n)
        builder.make_output(sh)

        self.assertEqual(builder.move_shape_and_size_nodes(), 0)
        self.assertEqual([n.op_type for n in builder.build_graph().node], ["Neg", "Shape"])

    def test_move_shape_and_size_nodes_runs_on_export(self):
        builder = GraphBuilder("g", opsets=28)
        builder.make_input("x", FLOAT, [2, 3])
        (p,) = builder.make_node("Neg", ["x"])
        (a,) = builder.make_node("Add", [p, p])
        (sh,) = builder.make_node("Shape", [p])
        builder.make_output(a)
        builder.make_output(sh)

        # to_onnx hoists Shape next to its producer before exporting.
        model = builder.to_onnx()
        self.assertEqual([n.op_type for n in model.graph.node], ["Neg", "Shape", "Add"])

    def test_move_shape_and_size_nodes_recurses_into_subgraphs(self):
        builder = GraphBuilder("g", opsets=28)
        builder.make_input("x", FLOAT, [2, 3])
        (top,) = builder.make_node("Neg", ["x"])
        builder.make_output(top)

        body = builder.make_subgraph("body")
        body.make_input("a", FLOAT, [2, 3])
        (p,) = body.make_node("Neg", ["a"])
        (u,) = body.make_node("Add", [p, p])
        (sh,) = body.make_node("Shape", [p])
        body.make_output(u)
        body.make_output(sh)

        self.assertEqual(builder.move_shape_and_size_nodes(), 1)
        self.assertEqual([n.op_type for n in body.build_graph().node], ["Neg", "Shape", "Add"])

    def test_inline_local_functions(self):
        builder = GraphBuilder("g", opsets=28)
        # A local function computing Neg(Add(a, b)).
        builder.set_opset_version("custom", 1)
        fct = builder.make_local_function("MyFct", domain="custom")
        fct.make_input("a", FLOAT, [2, 3])
        fct.make_input("b", FLOAT, [2, 3])
        (s,) = fct.make_node("Add", ["a", "b"])
        (n,) = fct.make_node("Neg", [s])
        fct.make_output(n)

        builder.make_input("x", FLOAT, [2, 3])
        builder.make_input("z", FLOAT, [2, 3])
        (out,) = builder.make_node("MyFct", ["x", "z"], domain="custom")
        builder.make_output(out)

        self.assertEqual(builder.inline_local_functions(), 1)
        graph = builder.build_graph()
        # The call node is replaced by the two body nodes.
        self.assertEqual(len(graph.node), 2)
        self.assertEqual(graph.node[0].op_type, "Add")
        self.assertEqual(list(graph.node[0].input), ["x", "z"])
        self.assertEqual(graph.node[1].op_type, "Neg")
        self.assertEqual(graph.node[1].output[0], out)
        # The fully inlined function definition is dropped from the model.
        self.assertFalse(builder.has_local_function("MyFct"))
        model = builder.to_onnx()
        self.assertEqual(len(model.functions), 0)
        # A second pass has nothing left to inline.
        self.assertEqual(builder.inline_local_functions(), 0)

    def test_inline_local_functions_keeps_uncalled_function(self):
        builder = GraphBuilder("g", opsets=28)
        builder.set_opset_version("custom", 1)
        fct = builder.make_local_function("MyFct", domain="custom")
        fct.make_input("a", FLOAT, [2, 3])
        (n,) = fct.make_node("Neg", ["a"])
        fct.make_output(n)

        builder.make_input("x", FLOAT, [2, 3])
        (top,) = builder.make_node("Neg", ["x"])
        builder.make_output(top)

        # No call site, so nothing is inlined and the definition is left in place.
        self.assertEqual(builder.inline_local_functions(), 0)
        self.assertTrue(builder.has_local_function("MyFct"))

    def test_inline_local_functions_include_exclude(self):
        def build():
            builder = GraphBuilder("g", opsets=28)
            builder.set_opset_version("custom", 1)
            keep = builder.make_local_function("Keep", domain="custom")
            keep.make_input("a", FLOAT, [2, 3])
            (kn,) = keep.make_node("Neg", ["a"])
            keep.make_output(kn)

            drop = builder.make_local_function("Drop", domain="custom")
            drop.make_input("a", FLOAT, [2, 3])
            (dn,) = drop.make_node("Neg", ["a"])
            drop.make_output(dn)

            builder.make_input("x", FLOAT, [2, 3])
            (kc,) = builder.make_node("Keep", ["x"], domain="custom")
            (dc,) = builder.make_node("Drop", [kc], domain="custom")
            builder.make_output(dc)
            return builder

        # include: only "Drop" is inlined; a (domain, name) tuple selects it.
        builder = build()
        self.assertEqual(builder.inline_local_functions(include=[("custom", "Drop")]), 1)
        self.assertTrue(builder.has_local_function("Keep"))
        self.assertFalse(builder.has_local_function("Drop"))

        # exclude: everything except "Keep" is inlined.
        builder = build()
        self.assertEqual(builder.inline_local_functions(exclude=[("custom", "Keep")]), 1)
        self.assertTrue(builder.has_local_function("Keep"))
        self.assertFalse(builder.has_local_function("Drop"))

        # An empty domain matches every function sharing the name.
        builder = build()
        self.assertEqual(builder.inline_local_functions(include=[("", "Drop")]), 1)
        self.assertTrue(builder.has_local_function("Keep"))
        self.assertFalse(builder.has_local_function("Drop"))

        # An empty name matches every function in the domain (here both).
        builder = build()
        self.assertEqual(builder.inline_local_functions(include=[("custom", "")]), 2)
        self.assertFalse(builder.has_local_function("Keep"))
        self.assertFalse(builder.has_local_function("Drop"))

        # Passing both include and exclude is rejected.
        builder = build()
        with self.assertRaises(ValueError):
            builder.inline_local_functions(
                include=[("custom", "Keep")], exclude=[("custom", "Drop")]
            )

    def test_constant_folding_options_are_exposed(self):
        from onnx_light.onnx_core.graph_builder import ConstantFoldingOptions

        options = ConstantFoldingOptions()
        # Defaults mirror the C++ ConstantFoldingOptions struct.
        self.assertTrue(options.enabled)
        self.assertEqual(options.max_element_count, -1)
        self.assertTrue(options.fold_weights)
        self.assertFalse(options.raise_on_missing_weight_kernel)
        self.assertEqual(options.excluded_ops, set())

        # Every field is writable from Python.
        options.enabled = False
        options.max_element_count = 1024
        options.fold_weights = False
        options.raise_on_missing_weight_kernel = True
        options.excluded_ops = {("", "RandomNormal")}
        self.assertFalse(options.enabled)
        self.assertEqual(options.max_element_count, 1024)
        self.assertFalse(options.fold_weights)
        self.assertTrue(options.raise_on_missing_weight_kernel)
        self.assertEqual(options.excluded_ops, {("", "RandomNormal")})

    def test_constant_fold_is_callable(self):
        from onnx_light.onnx_core.graph_builder import ConstantFoldingOptions

        builder = GraphBuilder("g", opsets=28)
        builder.make_input("x", FLOAT, [2, 3])
        builder.make_input("y", FLOAT, [2, 3])
        (z,) = builder.make_node("Add", ["x", "y"])
        builder.make_output(z)

        # Nothing is constant here, so folding is a no-op regardless of the
        # kernels linked into this build variant.
        self.assertEqual(builder.constant_fold(), 0)
        # A disabled pass is always a no-op and accepts an explicit options object.
        options = ConstantFoldingOptions()
        options.enabled = False
        self.assertEqual(builder.constant_fold(options), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
