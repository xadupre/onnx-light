import unittest
from onnx_light.ext_test_case import ExtTestCase
import onnx
import onnx.defs as onnx_defs
import onnx_light.onnx


class TestSchemaSyncWithOnnxDefs(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        onnx_light.onnx.defs.register_onnx_operator_set_schema()

    def test_onnx_light_ir_and_opset_versions_match_onnx(self):
        self.assertEqual(onnx_light.onnx.defs.onnx_ir_version(), onnx.IR_VERSION)
        self.assertGreaterEqual(
            onnx_light.onnx.defs.onnx_opset_version(), onnx_defs.onnx_opset_version()
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
