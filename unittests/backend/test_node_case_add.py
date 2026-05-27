# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Python-defined node test cases mirroring upstream onnx."""
import unittest

import numpy as np

import onnx_light.onnx as onnxl
from onnx_light.backend.test.case.base import collect_test_case, get_test_cases_for_op
from onnx_light.ext_test_case import ExtTestCase


# All eight Add test case names exported by ``onnx.backend.test.case.node.add``.
_EXPECTED_ADD_CASES = {
    "test_add": np.float32,
    "test_add_int8": np.int8,
    "test_add_int16": np.int16,
    "test_add_uint8": np.uint8,
    "test_add_uint16": np.uint16,
    "test_add_uint32": np.uint32,
    "test_add_uint64": np.uint64,
    "test_add_bcast": np.float32,
}


class TestNodeCaseAdd(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        onnxl.defs.register_onnx_operator_set_schema()

    def test_all_add_cases_registered(self):
        cases = collect_test_case()
        for name in _EXPECTED_ADD_CASES:
            self.assertIn(name, cases)

    def test_add_cases_have_expected_dtype_and_shape(self):
        cases = collect_test_case()
        for name, dtype in _EXPECTED_ADD_CASES.items():
            tc = cases[name]
            self.assertEqual(len(tc.data_sets), 1)
            inputs, outputs = tc.data_sets[0]
            self.assertEqual(len(inputs), 2)
            self.assertEqual(len(outputs), 1)
            for arr in inputs + outputs:
                self.assertEqual(arr.dtype, np.dtype(dtype))
            # Output is element-wise sum of the two inputs (with broadcasting).
            np.testing.assert_array_equal(outputs[0], inputs[0] + inputs[1])
            self.assertEqual(inputs[0].shape, (3, 4, 5))
            self.assertEqual(outputs[0].shape, (3, 4, 5))
        # Broadcast case has a (5,) second input.
        bcast = cases["test_add_bcast"]
        self.assertEqual(bcast.data_sets[0][0][1].shape, (5,))

    def test_add_cases_use_add_op_at_opset_14(self):
        cases = get_test_cases_for_op("Add", opset_version=14)
        for name in _EXPECTED_ADD_CASES:
            self.assertIn(name, cases)


if __name__ == "__main__":
    unittest.main(verbosity=2)
