import unittest
from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx_optim.shape_inference import infer_shapes_model
from onnx_light.backend_test import collect_test_cases


class TestOnnxOptimShapeInferenceModelBackend(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_rejects_model_without_graph(self):
        shape_tests = []
        for test in collect_test_cases():
            if "test_cc_shape_inference_add_concat_reshape" == test.name:
                shape_tests.append(test)
        self.assertEqual(len(shape_tests), 1)
        tests = collect_test_cases("test_cc_shape_inference_add_concat_reshape")
        self.assertEqual(len(tests), 1)
        test = tests[0]
        model = test.model
        infer_shapes_model(model)


if __name__ == "__main__":
    unittest.main()
