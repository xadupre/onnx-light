import unittest
from onnx_light.backend.test.case import make_test_class

TestOrtBackend = make_test_class()


if __name__ == "__main__":
    unittest.main(verbosity=2)
