# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for the C++ ``onnx_op`` submodule exposed via nanobind."""

from __future__ import annotations

import unittest

from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx_py import _onnxpy  # type: ignore[attr-defined]


class TestOnnxPyOp(ExtTestCase):
    def setUp(self) -> None:
        self.mod = _onnxpy.onnx_op

    def test_submodule_exposed(self) -> None:
        self.assertTrue(hasattr(_onnxpy, "onnx_op"))
        self.assertEqual(self.mod.kOnnxDomain, "ai.onnx")

    def test_tensor_type_and_to_type_string(self) -> None:
        self.assertEqual(self.mod.ToTypeString(self.mod.TensorType.kFloat), "tensor(float)")
        self.assertEqual(self.mod.ToTypeString(self.mod.TensorType.kInt64), "tensor(int64)")
        self.assertEqual(
            self.mod.ToTypeString(self.mod.TensorType.kSeqInt64), "seq(tensor(int64))"
        )

    def test_construct_formal_parameter(self) -> None:
        fp = self.mod.FormalParameter()
        fp.name = "X"
        fp.description = "input"
        fp.type = "T"
        self.assertEqual(fp.name, "X")
        self.assertEqual(fp.description, "input")
        self.assertEqual(fp.type, "T")

    def test_construct_type_constraint_param(self) -> None:
        tc = self.mod.TypeConstraintParam()
        tc.type_param_str = "T"
        tc.allowed_type_strs = [self.mod.TensorType.kFloat, self.mod.TensorType.kDouble]
        tc.description = "any float"
        self.assertEqual(tc.type_param_str, "T")
        self.assertEqual(len(tc.allowed_type_strs), 2)
        self.assertEqual(tc.allowed_type_strs[0], self.mod.TensorType.kFloat)

    def test_construct_light_op_schema(self) -> None:
        fp = self.mod.FormalParameter()
        fp.name, fp.description, fp.type = "X", "in", "T"
        tc = self.mod.TypeConstraintParam()
        tc.type_param_str = "T"
        tc.allowed_type_strs = [self.mod.TensorType.kFloat]
        tc.description = "any float"
        schema = self.mod.LightOpSchema("MyOp", "ai.onnx", 1, "doc", [fp], [fp], [tc])
        self.assertEqual(schema.name, "MyOp")
        self.assertEqual(schema.domain, "ai.onnx")
        self.assertEqual(schema.since_version, 1)
        self.assertEqual(schema.doc, "doc")
        self.assertEqual(len(schema.inputs), 1)
        self.assertEqual(len(schema.outputs), 1)
        self.assertEqual(len(schema.type_constraints), 1)
        self.assertFalse(schema.has_function_implementation)

    def test_get_all_onnx_op_schemas_with_history(self) -> None:
        schemas = self.mod.GetAllOnnxOpSchemasWithHistory()
        self.assertIsInstance(schemas, list)
        self.assertGreater(len(schemas), 0)
        for schema in schemas:
            self.assertIsInstance(schema, self.mod.LightOpSchema)
            self.assertTrue(schema.name)
            self.assertTrue(schema.domain)
            self.assertGreaterEqual(schema.since_version, 1)
        domains = {s.domain for s in schemas}
        self.assertIn("ai.onnx", domains)

    def test_get_all_onnx_op_schemas_with_history_op_type(self) -> None:
        all_schemas = self.mod.GetAllOnnxOpSchemasWithHistory()
        abs_schemas = self.mod.GetAllOnnxOpSchemasWithHistory(True, "Abs")
        self.assertIsInstance(abs_schemas, list)
        self.assertGreater(len(abs_schemas), 0)
        for schema in abs_schemas:
            self.assertEqual(schema.name, "Abs")
        expected = sum(1 for s in all_schemas if s.name == "Abs")
        self.assertEqual(len(abs_schemas), expected)
        self.assertEqual(
            self.mod.GetAllOnnxOpSchemasWithHistory(False, "ThisOpDoesNotExist"), []
        )


if __name__ == "__main__":
    unittest.main()
