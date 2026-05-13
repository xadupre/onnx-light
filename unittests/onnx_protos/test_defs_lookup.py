import unittest

from onnx_light.ext_test_case import ExtTestCase
from unittest import mock

import onnx
import onnx_light.onnx as onnxl
import onnx_light.onnx.defs as defs


class TestDefsLookup(ExtTestCase):
    def test_defs_module_exposed_from_package(self):
        self.assertIs(onnxl.defs, defs)

    def test_schema_lookup(self):
        op_type = "CopilotUnitLookupOp"
        version = 42
        self.assertFalse(defs.has(op_type))
        schema = defs.OpSchema(op_type, defs.ONNX_DOMAIN, version, doc="test schema")
        defs.register_schema(schema)
        self.assertTrue(defs.has(op_type))
        self.addCleanup(defs.deregister_schema, op_type, version, defs.ONNX_DOMAIN)
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

    def test_onnx_ir_version(self):
        self.assertEqual(defs.onnx_ir_version(), onnx.IR_VERSION)

    def test_onnx_ir_version_opset_mapping(self):
        cases = (
            (8, 3),
            (9, 4),
            (10, 5),
            (11, 6),
            (12, 7),
            (15, 8),
            (19, 9),
            (21, 10),
            (23, 11),
            (24, 12),
            (25, 13),
            (27, 13),
        )
        for opset, expected_ir in cases:
            with mock.patch.object(defs, "onnx_opset_version", return_value=opset):
                self.assertEqual(defs.onnx_ir_version(), expected_ir)


if __name__ == "__main__":
    unittest.main(verbosity=2)
