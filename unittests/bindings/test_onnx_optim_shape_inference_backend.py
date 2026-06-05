import unittest
from onnx_light.ext_test_case import ExtTestCase
import onnx_light.onnx as onnxl
from onnx_light.backend_test import collect_test_cases


class TestOnnxOptimShapeInferenceModelBackend(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_collect_test_cases_by_name(self):
        shape_tests = []
        for test in collect_test_cases():
            if "test_cc_shape_inference_add_concat_reshape" == test.name:
                shape_tests.append(test)
        self.assertEqual(len(shape_tests), 1)
        tests = [
            test
            for test in collect_test_cases("Add")
            if "test_cc_shape_inference_add_concat_reshape" == test.name
        ]
        self.assertEqual(len(tests), 1)
        test = tests[0]
        self.assertEqual(tests[0].name, shape_tests[0].name)

    def test_inference_shape(self):
        from onnx_light.onnx_optim.shape_inference import infer_shapes_model

        tests = [
            test
            for test in collect_test_cases("Add")
            if "test_cc_shape_inference_add_concat_reshape" == test.name
        ]
        self.assertEqual(len(tests), 1)
        test = tests[0]
        model = onnxl.ModelProto()
        model.CopyFrom(test.model)
        model.graph.value_info.clear()
        infer_shapes_model(model)
        self.assertEqual(len(test.model.graph.value_info), len(model.graph.value_info))
        for expected, inferred in zip(test.model.graph.value_info, model.graph.value_info):
            self.assertEqual(expected, inferred)

    def test_inference_by_node(self):
        from onnx_light.onnx_py._onnxpy.shape_inference import (
            compute_shape_node,
            ShapesContext,
            OptimTensor,
        )

        tests = [
            test
            for test in collect_test_cases("Add")
            if "test_cc_shape_inference_add_concat_reshape" == test.name
        ]
        self.assertEqual(len(tests), 1)
        test = tests[0]
        expected = {info.name: info for info in test.model.graph.value_info}

        ctx = ShapesContext()
        for opset in test.model.opset_import:
            ctx.set_opset_version(opset.domain, opset.version)
        for inp in test.model.graph.input:
            tt = inp.type.tensor_type
            dims = [d.dim_value if d.dim_value else d.dim_param for d in tt.shape.dim]
            ctx.set(inp.name, OptimTensor(tt.elem_type, dims))
        for init in test.model.graph.initializer:
            ctx.set(init.name, OptimTensor(init.data_type, list(init.dims)))

        for node in test.model.graph.node:
            compute_shape_node(ctx, node)
            for out_name in node.output:
                if not out_name:
                    continue
                t = ctx.get(str(out_name))
                self.assertIn(out_name, expected)
                self.assertEqual(t, expected[out_name])


if __name__ == "__main__":
    unittest.main()
