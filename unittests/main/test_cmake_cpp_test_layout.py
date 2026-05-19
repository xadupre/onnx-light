import unittest
from pathlib import Path


class TestCMakeCppTestLayout(unittest.TestCase):
    def test_helper_test_uses_cc_common_directory(self):
        root = Path(__file__).resolve().parents[2]
        content = (root / "CMakeLists.txt").read_text(encoding="utf-8")

        self.assertIn("unittests/cc_common/test_onnx_light_helpers.cc", content)
        self.assertNotIn("unittests/cpp-exe/test_onnx_light_helpers.cc", content)
        self.assertTrue(
            (root / "unittests" / "cc_common" / "test_onnx_light_helpers.cc").exists()
        )
        self.assertFalse((root / "unittests" / "cpp-exe").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
