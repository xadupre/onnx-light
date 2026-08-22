# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`onnx_light.tools.kernel_inventory` (Step D of the kernel
parallelization roadmap)."""

from __future__ import annotations

import unittest

from onnx_light.ext_test_case import ExtTestCase
from onnx_light.tools import kernel_inventory


class TestKernelInventory(ExtTestCase):
    def test_dispatch_table_entries_are_parsed(self):
        entries = kernel_inventory.dispatch_table_entries()
        self.assertGreater(len(entries), 100)
        identifiers = {(domain, op_type) for domain, op_type, _ in entries}
        self.assertIn(("ai.onnx", "Abs"), identifiers)
        self.assertIn(("ai.onnx", "Gemm"), identifiers)
        # Every dispatch table identifier is unique.
        self.assertEqual(len(identifiers), len(entries))

    def _fake_tuning_report(self, kernels):
        return {"kernels": kernels}

    def test_kernel_with_tuning_schema_is_tunable_or_calibratable(self):
        fake_report = self._fake_tuning_report(
            [
                {
                    "library": "onnx_light",
                    "kernel": "Abs",
                    "implementation": "portable",
                    "element_type": 1,
                    "tuning_abi": 2,
                    "calibratable": True,
                    "parameter_names": ["parallel.minimum_elements"],
                },
                {
                    "library": "onnx_light",
                    "kernel": "Abs",
                    "implementation": "portable",
                    "element_type": 11,
                    "tuning_abi": 2,
                    "calibratable": False,
                    "parameter_names": ["parallel.minimum_elements"],
                },
            ]
        )
        rows = kernel_inventory.build_kernel_inventory(tuning_report=fake_report)
        kernel_inventory.validate_inventory(rows)
        abs_rows = [row for row in rows if row["op_type"] == "Abs"]
        self.assertEqual(len(abs_rows), 2)
        states = {row["element_type"]: row["coverage_state"] for row in abs_rows}
        self.assertEqual(states[1], "calibratable")
        self.assertEqual(states[11], "tunable")
        for row in abs_rows:
            self.assertTrue(row["uses_parallel_for"])
            self.assertIsNone(row["serial_reason"])

    def test_kernel_without_tuning_schema_falls_back_to_source_scan(self):
        rows = kernel_inventory.build_kernel_inventory(tuning_report=self._fake_tuning_report([]))
        kernel_inventory.validate_inventory(rows)
        gemm_rows = [row for row in rows if row["op_type"] == "Gemm"]
        self.assertEqual(len(gemm_rows), 1)
        gemm_row = gemm_rows[0]
        self.assertTrue(gemm_row["uses_parallel_for"])
        self.assertEqual(gemm_row["coverage_state"], "parallel_fixed_policy")
        self.assertIsNone(gemm_row["element_type"])
        self.assertIsNone(gemm_row["serial_reason"])

    def test_every_path_appears_exactly_once(self):
        rows = kernel_inventory.build_kernel_inventory(tuning_report=self._fake_tuning_report([]))
        kernel_inventory.validate_inventory(rows)
        identities = [
            (row["domain"], row["op_type"], row["device"], row["element_type"], row["implementation"])
            for row in rows
        ]
        self.assertEqual(len(identities), len(set(identities)))
        for row in rows:
            self.assertIn(
                row["coverage_state"],
                ("serial", "parallel_fixed_policy", "tunable", "calibratable"),
            )

    def test_validate_inventory_rejects_duplicates(self):
        rows = kernel_inventory.build_kernel_inventory(tuning_report=self._fake_tuning_report([]))
        rows.append(dict(rows[0]))
        with self.assertRaises(ValueError):
            kernel_inventory.validate_inventory(rows)

    def test_serial_kernel_reports_a_reason(self):
        rows = kernel_inventory.build_kernel_inventory(tuning_report=self._fake_tuning_report([]))
        serial_rows = [row for row in rows if row["coverage_state"] == "serial"]
        self.assertGreater(len(serial_rows), 0)
        for row in serial_rows:
            self.assertIsInstance(row["serial_reason"], str)
            self.assertGreater(len(row["serial_reason"]), 0)


if __name__ == "__main__":
    unittest.main()
