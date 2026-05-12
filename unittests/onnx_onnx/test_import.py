import unittest

import onnx_light


class TestImport(unittest.TestCase):
    def test_import(self):
        self.assertIsNotNone(onnx_light.__version__)


if __name__ == "__main__":
    unittest.main(verbosity=2)
