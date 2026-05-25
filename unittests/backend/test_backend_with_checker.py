import unittest

import onnx_light.onnx.checker as checker
from onnx_light.backend.test.case import make_test_class


TestCheckerBackend = make_test_class(checker.check_model)


if __name__ == "__main__":
    unittest.main(verbosity=2)
