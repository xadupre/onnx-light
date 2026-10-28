import subprocess
import sys
import unittest

from shape_inference_test_utils import TestShapeInferenceHelper

from onnx_light.ext_test_case import ExtTestCase


class TestShapeInferenceHarness(ExtTestCase):
    def test_helper_has_no_test_cases(self):
        self.assertEqual(
            unittest.defaultTestLoader.loadTestsFromTestCase(
                TestShapeInferenceHelper
            ).countTestCases(),
            0,
        )

    def test_consumers_import_independently(self):
        consumers = {
            "test_shape_inference": "TestShapeInference",
            "test_data_propagation": "TestDataPropagation",
            "test_function_inference": "TestFunctionInference",
        }
        for module_name, class_name in consumers.items():
            with self.subTest(module=module_name):
                code = f"""
import importlib
import sys
sys.path = {sys.path!r}
module = importlib.import_module({module_name!r})
from shape_inference_test_utils import TestShapeInferenceHelper
assert issubclass(getattr(module, {class_name!r}), TestShapeInferenceHelper)
assert not (set({list(consumers)!r}) - {{{module_name!r}}}) & sys.modules.keys()
"""
                result = subprocess.run(
                    [sys.executable, "-c", code],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
