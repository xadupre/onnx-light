import unittest
from onnx_light.ext_test_case import ExtTestCase
import onnx
import onnx.defs as onnx_defs
import onnx_light.onnx


class TestSchemaSyncWithOnnx(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        onnx_light.onnx.defs.register_onnx_operator_set_schema()

    def test_onnx_light_ir_and_opset_versions_match_onnx(self):
        self.assertEqual(onnx_light.onnx.defs.onnx_ir_version(), onnx.IR_VERSION)
        self.assertGreaterEqual(
            onnx_light.onnx.defs.onnx_opset_version(), onnx_defs.onnx_opset_version()
        )

    def test_flex_attention_inputs_outputs_match_onnx(self):
        flex_attention_key = ("ai.onnx.preview", "FlexAttention", 1)
        onnx_schema_keys = {
            (schema.domain, schema.name, schema.since_version)
            for schema in onnx_defs.get_all_schemas_with_history()
        }
        if flex_attention_key not in onnx_schema_keys:
            self.skipTest("Installed onnx does not include FlexAttention schema.")

        light_schema = onnx_light.onnx.defs.get_schema("FlexAttention", 1, "ai.onnx.preview")
        onnx_schema = onnx_defs.get_schema("FlexAttention", 1, "ai.onnx.preview")

        self.assertEqual(
            self._io_signature(light_schema.inputs), self._io_signature(onnx_schema.inputs)
        )
        self.assertEqual(
            self._io_signature(light_schema.outputs), self._io_signature(onnx_schema.outputs)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
