import unittest

from onnx_light.ext_test_case import ExtTestCase
from pathlib import Path


class TestCMakeCppTestLayout(ExtTestCase):
    def test_helper_test_belongs_to_standalone_proto_project(self):
        root = Path(__file__).resolve().parents[2]
        content = (root / "CMakeLists.txt").read_text(encoding="utf-8")
        proto_tests = (root / "onnx_proto/tests/CMakeLists.txt").read_text(encoding="utf-8")

        self.assertIn("set(ONNX_PROTO_BUILD_TESTS ${ONNX_LIGHT_BUILD_TESTS})", content)
        self.assertIn(
            "target_link_libraries(test_onnx_light_helpers PRIVATE lib_onnx_proto)", proto_tests
        )
        self.assertNotIn("unittests/cc/onnx_proto", content)
        self.assertNotIn("unittests/cpp-exe/test_onnx_light_helpers.cc", content)
        self.assertTrue((root / "onnx_proto/tests/test_onnx_light_helpers.cc").exists())
        self.assertFalse((root / "unittests" / "cpp-exe").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
