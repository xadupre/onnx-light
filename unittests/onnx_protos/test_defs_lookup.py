import unittest

from onnx_light.ext_test_case import ExtTestCase
from unittest import mock
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

    def test_get_schema_returns_highest_opset_less_or_equal(self):
        op_type = "CopilotHighestOpsetOp"
        versions = [7, 13, 14]
        for v in versions:
            schema = defs.OpSchema(op_type, defs.ONNX_DOMAIN, v, doc=f"schema v{v}")
            defs.register_schema(schema)
            self.addCleanup(defs.deregister_schema, op_type, v, defs.ONNX_DOMAIN)

        # Exact version matches return the schema introduced at that version.
        self.assertEqual(defs.get_schema(op_type, 7).since_version, 7)
        self.assertEqual(defs.get_schema(op_type, 13).since_version, 13)
        self.assertEqual(defs.get_schema(op_type, 14).since_version, 14)

        # A version between two registered versions returns the highest registered
        # version that is still less than or equal to the requested version.
        self.assertEqual(defs.get_schema(op_type, 8).since_version, 7)
        self.assertEqual(defs.get_schema(op_type, 12).since_version, 7)
        self.assertEqual(defs.get_schema(op_type, 100).since_version, 14)

        # A version lower than the earliest registered version raises SchemaError.
        with self.assertRaises(defs.SchemaError):
            defs.get_schema(op_type, 6)

    def test_get_schema_abs_all_opsets(self):
        # Abs is defined at opset versions 1, 6, and 13 in the ONNX standard.
        # In a full onnx_light build these schemas are pre-registered from C++
        # static initializers, so only register them when they are absent.
        abs_versions = [1, 6, 13]
        registered = {
            (s.name, s.since_version, s.domain) for s in defs.get_all_schemas_with_history()
        }
        for version in abs_versions:
            if ("Abs", version, defs.ONNX_DOMAIN) not in registered:
                schema = defs.OpSchema(
                    "Abs", defs.ONNX_DOMAIN, version, doc=f"Abs schema v{version}"
                )
                defs.register_schema(schema)
                self.addCleanup(defs.deregister_schema, "Abs", version, defs.ONNX_DOMAIN)

        # For every opset from 1 to some maximum, get_schema should return the
        # schema introduced at the highest registered version <= the requested version.
        expected_since = {range(1, 6): 1, range(6, 13): 6, range(13, 21): 13}
        for version_range, expected in expected_since.items():
            for opset in version_range:
                with self.subTest(opset=opset):
                    schema = defs.get_schema("Abs", opset)
                    self.assertEqual(schema.since_version, expected)
                    self.assertEqual(schema.name, "Abs")
                    self.assertEqual(schema.domain, defs.ONNX_DOMAIN)

    def test_schema_lookup_error(self):
        with self.assertRaises(defs.SchemaError):
            defs.get_schema("DefinitelyUnknownOperator123")

    def test_onnx_opset_version(self):
        self.assertEqual(
            defs.onnx_opset_version(), defs.schema_version_map()[defs.ONNX_DOMAIN][1]
        )

    def test_onnx_ir_version(self):
        self.assertEqual(defs.onnx_ir_version(), onnxl.IR_VERSION)

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
