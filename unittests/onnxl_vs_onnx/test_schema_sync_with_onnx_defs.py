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

    def test_registered_onnx_ops_match_onnx(self):
        flex_attention_key = ("ai.onnx.preview", "FlexAttention", 1)
        onnx_light_schema_keys = {
            (schema.domain, schema.name, schema.since_version)
            for schema in onnx_light.onnx.defs.get_all_schemas_with_history()
        }
        onnx_schema_keys = {
            (schema.domain, schema.name, schema.since_version)
            for schema in onnx_defs.get_all_schemas_with_history()
        }

        if flex_attention_key not in onnx_schema_keys:
            onnx_light_schema_keys.discard(flex_attention_key)

        self.assertEqual(onnx_light_schema_keys, onnx_schema_keys)

    def test_registered_onnx_ops_match_onnx_match_input_output(self):
        light_hist = onnx_light.onnx.defs.get_all_schemas_with_history()
        hist = onnx.defs.get_all_schemas_with_history()
        self.assertEqual(len(light_hist), len(hist))
        light_dict = {(s.domain, s.name, s.since_version): s for s in light_hist}
        onnx_dict = {(s.domain, s.name, s.since_version): s for s in hist}
        self.assertEqual(set(light_dict), set(onnx_dict))
        for key, schema in onnx_dict.items():
            lights = light_dict[key]
            self.assertEqual(len(schema.inputs), len(lights.inputs))
            self.assertEqual(len(schema.outputs), len(lights.outputs))
            self.assertEqual(len(schema.type_constraints), len(lights.type_constraints))
            self.assertEqual(schema.has_function, lights.has_function)
            self.assertEqual(schema.deprecated, lights.deprecated)
            self.assertEqual(schema.is_infinite(0), lights.is_infinite(0))
            self.assertEqual(schema.support_level, lights.support_level)
            self.assertEqual(schema.max_input, lights.max_input)
            self.assertEqual(schema.max_output, lights.max_output)
            # file and line are build-environment-specific and intentionally not compared
            self.assertEqual(schema.non_deterministic, lights.non_deterministic)
            self.assertEqual(schema.doc, lights.doc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
