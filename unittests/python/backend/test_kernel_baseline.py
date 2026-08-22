# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`onnx_light.tools.kernel_baseline` (Step E of the kernel
parallelization roadmap)."""

from __future__ import annotations

import unittest

from onnx_light.ext_test_case import ExtTestCase, import_or_skip
from onnx_light.tools import kernel_baseline

import_or_skip("onnx_light.onnx_py._onnxpykernels", "runtime")


class TestKernelBaseline(ExtTestCase):
    def test_get_cpu_descriptor_reports_architecture(self):
        descriptor = kernel_baseline.get_cpu_descriptor()
        self.assertIn("architecture", descriptor)
        self.assertIsInstance(descriptor["architecture"], str)
        self.assertGreater(len(descriptor["architecture"]), 0)
        self.assertIn("logical_cores", descriptor)

    def test_run_benchmark_corpus_small_shapes_only(self):
        cases = (
            {
                "op_type": "Abs",
                "arity": "unary",
                "element_type": "FLOAT",
                "shapes": (("small", 64),),
            },
        )
        results = kernel_baseline.run_benchmark_corpus(
            cases=cases,
            cpu_policies=(("serial", 1),),
            repeat=2,
            warmup=1,
            collect_diagnostics=False,
        )
        self.assertEqual(len(results), 1)
        row = results[0]
        self.assertEqual(row["op_type"], "Abs")
        self.assertEqual(row["cpu_policy"], "serial")
        self.assertEqual(row["effective_threads"], 1)
        self.assertGreaterEqual(row["startup_seconds"], 0.0)
        self.assertGreaterEqual(row["median_kernel_execution_seconds"], 0.0)
        self.assertGreaterEqual(row["wall_seconds"], 0.0)
        self.assertIsNone(row["diagnostics"])

    def test_run_benchmark_corpus_with_diagnostics(self):
        cases = (
            {
                "op_type": "Abs",
                "arity": "unary",
                "element_type": "FLOAT",
                "shapes": (("small", 64),),
            },
        )
        results = kernel_baseline.run_benchmark_corpus(
            cases=cases,
            cpu_policies=(("serial", 1),),
            repeat=1,
            warmup=0,
            collect_diagnostics=True,
        )
        row = results[0]
        self.assertIsNotNone(row["diagnostics"])
        self.assertIn("dropped_events", row["diagnostics"])

    def test_run_kernel_baseline_report_does_not_modify_tuning_cache(self):
        from onnx_light.kernel_tuning import kernel_tuning_parameters

        before = kernel_tuning_parameters(kernel="Abs", element_type=1)
        report = kernel_baseline.run_kernel_baseline_report(
            cases=(
                {
                    "op_type": "Not",
                    "arity": "unary",
                    "element_type": "BOOL",
                    "shapes": (("small", 64),),
                },
            ),
            cpu_policies=(("serial", 1),),
            repeat=1,
            warmup=0,
            collect_diagnostics=False,
        )
        after = kernel_tuning_parameters(kernel="Abs", element_type=1)
        self.assertEqual(before["kernels"], after["kernels"])
        self.assertIn("cpu_descriptor", report)
        self.assertIn("inventory", report)
        self.assertIn("benchmarks", report)
        self.assertGreater(len(report["inventory"]), 100)
        self.assertEqual(len(report["benchmarks"]), 1)


if __name__ == "__main__":
    unittest.main()
