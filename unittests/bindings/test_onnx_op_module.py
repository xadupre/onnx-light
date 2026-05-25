"""Tests for the :mod:`onnx_light.onnx_op` Python re-export module."""

import unittest

from onnx_light.ext_test_case import ExtTestCase
import onnx_light.onnx_op as op


class TestOnnxOpModule(ExtTestCase):
    def test_light_op_schema_exposed(self):
        self.assertTrue(hasattr(op, "LightOpSchema"))
        self.assertEqual(op.LightOpSchema.__name__, "LightOpSchema")

    def test_module_exposes_expected_symbols(self):
        for name in (
            "FormalParameter",
            "GetAllOnnxOpSchemasWithHistory",
            "LightOpSchema",
            "TensorType",
            "ToTypeString",
            "TypeConstraintParam",
            "get_all_schemas",
            "get_all_schemas_with_history",
            "kOnnxDomain",
        ):
            self.assertIn(name, op.__all__, name)
            self.assertTrue(hasattr(op, name), name)

    def test_onnx_domain_constant(self):
        self.assertEqual(op.kOnnxDomain, "ai.onnx")

    def test_get_all_schemas_with_history_nonempty(self):
        history = op.get_all_schemas_with_history()
        self.assertGreater(len(history), 50)
        sample = history[0]
        self.assertIsInstance(sample, op.LightOpSchema)
        self.assertTrue(sample.name)
        self.assertTrue(sample.domain)
        self.assertGreater(sample.since_version, 0)

    def test_get_all_schemas_returns_one_per_operator(self):
        latest = op.get_all_schemas()
        self.assertGreater(len(latest), 0)
        keys = [(s.domain, s.name) for s in latest]
        self.assertEqual(len(keys), len(set(keys)))

    def test_to_type_string_returns_str(self):
        s = op.ToTypeString(op.TensorType.kFloat)
        self.assertIsInstance(s, str)
        self.assertEqual(s, "tensor(float)")


if __name__ == "__main__":
    unittest.main()
