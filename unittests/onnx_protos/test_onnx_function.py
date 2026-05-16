# source: https://github.com/onnx/onnx/blob/main/onnx/test/function_test.py
from onnx_light.ext_test_case import ExtTestCase
import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
from onnx_light.onnx import checker, utils


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
