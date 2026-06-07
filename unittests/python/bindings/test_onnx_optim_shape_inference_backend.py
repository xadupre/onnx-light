import unittest
from onnx_light.ext_test_case import ExtTestCase
import onnx_light.onnx as onnxl
import onnx_light.onnx.numpy_helper as onh
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

    def test_dataset_repr(self):
        abs_tests = [test for test in collect_test_cases("Abs") if test.name == "test_cc_abs"]
        self.assertEqual(len(abs_tests), 1)
        self.assertEqual(repr(abs_tests[0].data_sets[0]), "DataSet(inputs=1, outputs=1)")

    def test_tensor_repr(self):
        abs_tests = [test for test in collect_test_cases("Abs") if test.name == "test_cc_abs"]
        self.assertEqual(len(abs_tests), 1)
        ds = abs_tests[0].data_sets[0]
        self.assertEqual(repr(ds.inputs[0]), "Tensor(name='x', data_type=FLOAT, shape=[2, 3])")
        self.assertEqual(repr(ds.outputs[0]), "Tensor(name='y', data_type=FLOAT, shape=[2, 3])")

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
        from onnx_light.onnx_py._onnxpy import shape_inference as si

        tests = [
            test
            for test in collect_test_cases("Add")
            if "test_cc_shape_inference_add_concat_reshape" == test.name
        ]
        self.assertEqual(len(tests), 1)
        test = tests[0]
        expected = {info.name: info for info in test.model.graph.value_info}

        ctx = si.ShapesContext()
        for opset in test.model.opset_import:
            ctx.set_opset_version(opset.domain, opset.version)
        for inp in test.model.graph.input:
            tt = inp.type.tensor_type
            dims = [d.dim_value if d.dim_value else d.dim_param for d in tt.shape.dim]
            t = si.OptimTensor(tt.elem_type, dims)
            ctx.set(inp.name, t)
        for init in test.model.graph.initializer:
            t = si.OptimTensor(init.data_type, list(init.dims))
            a = onh.to_array(init)
            t.set_value_as_shape([int(i) for i in a])
            ctx.set(init.name, t)

        for node in test.model.graph.node:
            si.compute_shape_node(ctx, node)
            for out_name in node.output:
                if not out_name or out_name in {"Z"}:
                    continue
                t = ctx.get(str(out_name))
                self.assertIn(out_name, expected)
                v = expected[out_name]
                self.assertEqual(len(v.type.tensor_type.shape.dim), len(t.shape))
                for a, b in zip(v.type.tensor_type.shape.dim, t.shape):
                    self.assertEqual(a.dim_param, b)

        # outputs
        self.assertEqual(["batch", "seq", "2*d_model"], list(ctx.get("Z").shape))


if __name__ == "__main__":
    unittest.main()
