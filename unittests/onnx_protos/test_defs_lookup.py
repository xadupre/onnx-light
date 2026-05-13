import unittest

import onnx_light.onnx as onnxl
import onnx_light.onnx.defs as defs


class TestDefsLookup(unittest.TestCase):
    def test_defs_module_exposed_from_package(self):
        self.assertIs(onnxl.defs, defs)

    def test_schema_lookup(self):
        op_type = "CopilotUnitLookupOp"
        version = 42
        self.assertFalse(defs.has(op_type))
        schema = defs.OpSchema(op_type, defs.ONNX_DOMAIN, version, doc="test schema")
        defs.register_schema(schema)
        self.addCleanup(defs.deregister_schema, op_type, version, defs.ONNX_DOMAIN)
        self.assertTrue(defs.has(op_type))
        self.assertTrue(defs.has_schema(op_type, version))
        retrieved = defs.get_schema(op_type)
        self.assertEqual(retrieved.name, op_type)
        self.assertEqual(retrieved.domain, defs.ONNX_DOMAIN)
        self.assertEqual(retrieved.since_version, version)
        self.assertEqual(defs.get_schema(op_type, version).since_version, version)

    def test_schema_lookup_error(self):
        with self.assertRaises(defs.SchemaError):
            defs.get_schema("DefinitelyUnknownOperator123")

    def test_onnx_opset_version(self):
        self.assertEqual(
            defs.onnx_opset_version(), defs.schema_version_map()[defs.ONNX_DOMAIN][1]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
