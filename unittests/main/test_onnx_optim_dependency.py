import unittest
from pathlib import Path
import re


class TestOnnxOptimDependency(unittest.TestCase):
    def test_optim_tensor_tensor_type_matches_onnx_op(self):
        root = Path(__file__).resolve().parents[2]
        optim = (root / "onnx_light" / "onnx_optim" / "optim_tensor.h").read_text(
            encoding="utf-8"
        )
        op = (root / "onnx_light" / "onnx_op" / "light_op_schema.h").read_text(encoding="utf-8")

        pattern = r"enum class TensorType : uint8_t \{(?P<body>.*?)\n\};"
        optim_match = re.search(pattern, optim, re.DOTALL)
        op_match = re.search(pattern, op, re.DOTALL)
        self.assertIsNotNone(optim_match)
        self.assertIsNotNone(op_match)
        self.assertEqual(optim_match.group("body"), op_match.group("body"))

    def test_optim_tensor_header_does_not_include_onnx_op(self):
        root = Path(__file__).resolve().parents[2]
        content = (root / "onnx_light" / "onnx_optim" / "optim_tensor.h").read_text(
            encoding="utf-8"
        )

        self.assertNotIn('#include "onnx_op/light_op_schema.h"', content)
        self.assertIn("enum class TensorType : uint8_t {", content)

    def test_cmake_does_not_link_onnx_optim_against_onnx_op(self):
        root = Path(__file__).resolve().parents[2]
        content = (root / "CMakeLists.txt").read_text(encoding="utf-8")

        self.assertIn("target_link_libraries(lib_onnx_optim PUBLIC lib_onnx_proto)", content)
        self.assertNotIn("target_link_libraries(lib_onnx_optim PUBLIC lib_onnx_op)", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
