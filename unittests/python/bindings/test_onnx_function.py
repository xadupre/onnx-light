# source: https://github.com/onnx/onnx/blob/main/onnx/test/function_test.py
from onnx_light.ext_test_case import ExtTestCase
import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
from onnx_light.onnx import checker
from onnx_light.onnx_lib import utils


class TestOnnxLightFunction(ExtTestCase):
    def _verify_function_set(self, extracted_model, function_set, func_domain):
        checker.check_model(extracted_model)
        self.assertEqual(len(extracted_model.functions), len(function_set))
        present = {
            function.name
            for function in extracted_model.functions
            if function.domain == func_domain
        }
        self.assertEqual(present, set(function_set))

    def _make_model_with_local_function(self):
        func_domain = "local"
        func_opset_imports = [oh.make_opsetid("", 14)]
        func_nested_opset_imports = [oh.make_opsetid("", 14), oh.make_opsetid(func_domain, 1)]

        func_add_name = "func_add"
        func_identity_name = "func_identity"
        func_nested_identity_add_name = "func_nested_identity_add"

        func_add = oh.make_function(
            func_domain,
            func_add_name,
            ["a", "b"],
            ["c"],
            [oh.make_node("Add", ["a", "b"], ["c"])],
            func_opset_imports,
        )
        func_identity = oh.make_function(
            func_domain,
            func_identity_name,
            ["a"],
            ["b"],
            [oh.make_node("Identity", ["a"], ["b"])],
            func_opset_imports,
        )
        func_nested_identity_add = oh.make_function(
            func_domain,
            func_nested_identity_add_name,
            ["a", "b"],
            ["c"],
            [
                oh.make_node("func_identity", ["a"], ["a1"], domain=func_domain),
                oh.make_node("func_identity", ["b"], ["b1"], domain=func_domain),
                oh.make_node("func_add", ["a1", "b1"], ["c"], domain=func_domain),
            ],
            func_nested_opset_imports,
        )

        tensor_type_proto = oh.make_tensor_type_proto(elem_type=2, shape=[5])
        graph = oh.make_graph(
            [
                oh.make_node(func_add_name, ["i0", "i1"], ["t0"], domain=func_domain),
                oh.make_node("Add", ["i1", "i2"], ["t2"]),
                oh.make_node("Add", ["t0", "t2"], ["o_func_add"]),
                oh.make_node(func_identity_name, ["i1"], ["t1"], domain=func_domain),
                oh.make_node("Identity", ["i1"], ["t3"]),
                oh.make_node(
                    func_nested_identity_add_name,
                    ["t0", "t1"],
                    ["o_all_func0"],
                    domain=func_domain,
                ),
                oh.make_node(
                    func_nested_identity_add_name,
                    ["t3", "t2"],
                    ["o_all_func1"],
                    domain=func_domain,
                ),
                oh.make_node("Add", ["t3", "t2"], ["o_no_func"]),
            ],
            "graph_with_embedded_functions",
            [
                oh.make_value_info(name="i0", type_proto=tensor_type_proto),
                oh.make_value_info(name="i1", type_proto=tensor_type_proto),
                oh.make_value_info(name="i2", type_proto=tensor_type_proto),
            ],
            [
                oh.make_value_info(name="o_no_func", type_proto=tensor_type_proto),
                oh.make_value_info(name="o_func_add", type_proto=tensor_type_proto),
                oh.make_value_info(name="o_all_func0", type_proto=tensor_type_proto),
                oh.make_value_info(name="o_all_func1", type_proto=tensor_type_proto),
            ],
        )

        model = oh.make_model(
            graph,
            ir_version=8,
            opset_imports=[oh.make_opsetid("", 14), oh.make_opsetid(func_domain, 1)],
            producer_name="test_extract_model_with_local_function",
            functions=[func_identity, func_add, func_nested_identity_add],
        )
        model_onnx = onnxl.ModelProto()
        model_onnx.ParseFromString(model.SerializeToString())
        return model_onnx

    def test_extract_model_with_local_function(self):
        model = self._make_model_with_local_function()

        self._verify_function_set(
            utils.Extractor(model).extract_model(["i0", "i1", "i2"], ["o_no_func"]),
            set(),
            "local",
        )
        self._verify_function_set(
            utils.Extractor(model).extract_model(["i0", "i1", "i2"], ["o_func_add"]),
            {"func_add"},
            "local",
        )
        self._verify_function_set(
            utils.Extractor(model).extract_model(["i0", "i1", "i2"], ["o_all_func0"]),
            {"func_add", "func_identity", "func_nested_identity_add"},
            "local",
        )
        self._verify_function_set(
            utils.Extractor(model).extract_model(["i0", "i1", "i2"], ["o_all_func1"]),
            {"func_add", "func_identity", "func_nested_identity_add"},
            "local",
        )
        self._verify_function_set(
            utils.Extractor(model).extract_model(
                ["i0", "i1", "i2"], ["o_no_func", "o_func_add", "o_all_func0", "o_all_func1"]
            ),
            {"func_add", "func_identity", "func_nested_identity_add"},
            "local",
        )

    def test_extract_model_with_local_function_in_subgraph(self):
        opsets = [oh.make_opsetid("", 18), oh.make_opsetid("local", 1)]
        double = oh.make_function(
            "local", "Double", ["x"], ["y"], [oh.make_node("Add", ["x", "x"], ["y"])], opsets
        )
        unused = oh.make_function(
            "local", "Unused", ["x"], ["y"], [oh.make_node("Identity", ["x"], ["y"])], opsets
        )
        output = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [2])
        then_branch = oh.make_graph(
            [oh.make_node("Double", ["X"], ["then_y"], domain="local")],
            "then_branch",
            [],
            [oh.make_tensor_value_info("then_y", onnxl.TensorProto.FLOAT, [2])],
        )
        else_branch = oh.make_graph(
            [oh.make_node("Double", ["X"], ["else_y"], domain="local")],
            "else_branch",
            [],
            [oh.make_tensor_value_info("else_y", onnxl.TensorProto.FLOAT, [2])],
        )
        if_node = oh.make_node(
            "If", ["cond"], ["Y"], then_branch=then_branch, else_branch=else_branch
        )

        for in_function in (False, True):
            with self.subTest(in_function=in_function):
                functions = [double, unused]
                node = if_node
                expected = {"Double"}
                if in_function:
                    functions.append(
                        oh.make_function(
                            "local", "Wrapper", ["X", "cond"], ["Y"], [if_node], opsets
                        )
                    )
                    node = oh.make_node("Wrapper", ["X", "cond"], ["Y"], domain="local")
                    expected.add("Wrapper")
                graph = oh.make_graph(
                    [node],
                    "subgraph_local_function",
                    [
                        oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [2]),
                        oh.make_tensor_value_info("cond", onnxl.TensorProto.BOOL, []),
                    ],
                    [output],
                )
                model = oh.make_model(
                    graph, functions=functions, opset_imports=opsets, ir_version=9
                )
                checker.check_model(model)
                extracted = utils.Extractor(model).extract_model(["X", "cond"], ["Y"])
                self._verify_function_set(extracted, expected, "local")

    def test_collect_local_functions_in_graphs_attribute(self):
        opsets = [oh.make_opsetid("", 18), oh.make_opsetid("local", 1)]
        function = oh.make_function(
            "local", "Double", ["x"], ["y"], [oh.make_node("Add", ["x", "x"], ["y"])], opsets
        )
        branch = oh.make_graph(
            [oh.make_node("Double", ["X"], ["Y"], domain="local")],
            "branch",
            [],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [2])],
        )
        node = oh.make_node("GraphContainer", ["X"], ["Y"], branches=[branch, branch])
        model = oh.make_model(
            oh.make_graph(
                [node],
                "graph",
                [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [2])],
                [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [2])],
            ),
            functions=[function],
            opset_imports=opsets,
        )
        self.assertEqual(
            [f.name for f in utils.Extractor(model)._collect_referred_local_functions([node])],
            ["Double"],
        )
