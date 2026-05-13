import unittest

from onnx_light.ext_test_case import ExtTestCase

import onnx_light


class TestImport(ExtTestCase):
    def test_import(self):
        self.assertIsNotNone(onnx_light.__version__)


if __name__ == "__main__":
    unittest.main(verbosity=2)
