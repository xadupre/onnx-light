# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``python -m onnx_light kernel-baseline`` (``onnx_light.__main__``)."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

from onnx_light.__main__ import _build_parser, main
from onnx_light.ext_test_case import import_or_skip

import_or_skip("onnx_light.onnx_py._onnxpykernels", "runtime")


class TestMainKernelBaselineParser(unittest.TestCase):
    def test_kernel_baseline_defaults(self):
        parser = _build_parser()
        args = parser.parse_args(["kernel-baseline"])
        self.assertIsNone(args.output)
        self.assertEqual(args.repeat, 5)
        self.assertEqual(args.warmup, 2)
        self.assertFalse(args.no_diagnostics)


class TestMainKernelBaseline(unittest.TestCase):
    _SMALL_CASES = (
        {"op_type": "Not", "arity": "unary", "element_type": "BOOL", "shapes": (("small", 64),)},
    )

    def test_writes_json_report_to_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "report.json")
            with (
                mock.patch(
                    "onnx_light.tools.kernel_baseline.BENCHMARK_CORPUS", self._SMALL_CASES
                ),
                mock.patch("onnx_light.tools.kernel_baseline.CPU_POLICIES", (("serial", 1),)),
            ):
                main(
                    [
                        "kernel-baseline",
                        "--output",
                        output_path,
                        "--repeat",
                        "1",
                        "--warmup",
                        "0",
                        "--no-diagnostics",
                    ]
                )
            with open(output_path, encoding="utf-8") as report_file:
                report = json.load(report_file)
        self.assertIn("cpu_descriptor", report)
        self.assertIn("inventory", report)
        self.assertIn("benchmarks", report)
        self.assertEqual(len(report["benchmarks"]), 1)

    def test_prints_json_report_to_stdout(self):
        with (
            mock.patch("onnx_light.tools.kernel_baseline.BENCHMARK_CORPUS", self._SMALL_CASES),
            mock.patch("onnx_light.tools.kernel_baseline.CPU_POLICIES", (("serial", 1),)),
        ):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                main(["kernel-baseline", "--repeat", "1", "--warmup", "0", "--no-diagnostics"])
        report = json.loads(buffer.getvalue())
        self.assertIn("cpu_descriptor", report)
        self.assertEqual(len(report["benchmarks"]), 1)


if __name__ == "__main__":
    unittest.main()
