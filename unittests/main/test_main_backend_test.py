# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``python -m onnx_light backend``."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

from onnx_light.__main__ import _build_parser, _run_backend_test_timing, main
from onnx_light.ext_test_case import import_or_skip

import_or_skip("onnx_light.onnx_py._onnxpybackend", "backend_test")
import_or_skip("onnx_light.onnx_py._onnxpykernels", "runtime")


class TestMainBackendTest(unittest.TestCase):
    def test_parser_defaults(self):
        args = _build_parser().parse_args(["backend"])
        self.assertEqual(args.regex, "")
        self.assertEqual(args.mode, "test")
        self.assertEqual(args.repeat, 1)
        self.assertEqual(args.warmup, 0)
        self.assertFalse(args.include_big)
        self.assertFalse(args.json)
        self.assertIsNone(args.output)

    def test_parser_rejects_invalid_counts(self):
        parser = _build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["backend", "--repeat", "0"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["backend", "--warmup", "-1"])

    def test_times_one_correctness_case(self):
        report = _run_backend_test_timing(name_regex=r"^test_cc_abs$", repeat=1)
        self.assertEqual(report["mode"], "test")
        self.assertEqual(report["selected"], 1)
        self.assertEqual(report["cases"][0]["name"], "test_cc_abs")
        self.assertEqual(report["cases"][0]["data_sets"], 1)
        self.assertEqual(len(report["cases"][0]["iteration_seconds"]), 1)
        self.assertGreaterEqual(report["cases"][0]["run_seconds"], 0)

    def test_times_one_benchmark_case(self):
        report = _run_backend_test_timing(
            name_regex=r"^test_cc_abs_benchmark$", mode="benchmark", repeat=1
        )
        self.assertEqual(report["mode"], "benchmark")
        self.assertEqual(report["selected"], 1)
        self.assertEqual(report["cases"][0]["name"], "test_cc_abs_benchmark")
        self.assertGreater(report["cases"][0]["run_seconds"], 0)

    def test_json_output(self):
        expected = {
            "name_regex": "abs",
            "mode": "benchmark",
            "include_big": False,
            "repeat": 2,
            "warmup": 1,
            "selected": 0,
            "collection_seconds": 0.1,
            "cases": [],
            "total_seconds": 0.2,
        }
        buffer = io.StringIO()
        with (
            mock.patch("onnx_light.__main__._run_backend_test_timing", return_value=expected),
            redirect_stdout(buffer),
        ):
            main(
                [
                    "backend",
                    "--regex",
                    "abs",
                    "--mode",
                    "benchmark",
                    "--repeat",
                    "2",
                    "--warmup",
                    "1",
                    "--json",
                ]
            )
        self.assertEqual(json.loads(buffer.getvalue()), expected)

    def test_rejects_invalid_counts(self):
        with self.assertRaisesRegex(ValueError, "repeat must be positive"):
            _run_backend_test_timing(repeat=0)
        with self.assertRaisesRegex(ValueError, "warmup must be non-negative"):
            _run_backend_test_timing(warmup=-1)

    def test_rejects_invalid_regex(self):
        with self.assertRaises(ValueError):
            _run_backend_test_timing(name_regex="(")

    def test_python_module_command(self):
        process = subprocess.run(
            [sys.executable, "-m", "onnx_light", "backend", "--regex", "^test_cc_abs$", "--json"],
            check=True,
            capture_output=True,
            text=True,
        )
        report = json.loads(process.stdout)
        self.assertEqual(report["selected"], 1)
        self.assertEqual(report["cases"][0]["name"], "test_cc_abs")

    def test_writes_csv_or_xlsx_based_on_extension(self):
        expected = {
            "name_regex": "not",
            "mode": "test",
            "include_big": False,
            "repeat": 2,
            "warmup": 1,
            "selected": 1,
            "collection_seconds": 0.1,
            "cases": [
                {
                    "name": "test_cc_not",
                    "kind": "node",
                    "tag": "Not",
                    "data_sets": 1,
                    "materialization_seconds": 0.2,
                    "setup_seconds": 0.3,
                    "warmup_seconds": 0.4,
                    "iteration_seconds": [0.5, 0.6],
                    "run_seconds": 1.1,
                    "mean_seconds": 0.55,
                    "min_seconds": 0.5,
                    "max_seconds": 0.6,
                }
            ],
            "total_seconds": 2.1,
        }
        with tempfile.TemporaryDirectory() as temporary:
            for extension in (".csv", ".xlsx"):
                with self.subTest(extension=extension):
                    output = os.path.join(temporary, f"not{extension}")
                    with (
                        mock.patch(
                            "onnx_light.__main__._run_backend_test_timing", return_value=expected
                        ),
                        redirect_stdout(io.StringIO()),
                    ):
                        main(["backend", "--regex", "not", "--output", output])
                    self.assertTrue(os.path.exists(output))

            with open(os.path.join(temporary, "not.csv"), encoding="utf-8") as csv_file:
                csv_text = csv_file.read()
            self.assertIn("name,kind,tag,data_sets", csv_text)
            self.assertIn("test_cc_not,node,Not,1", csv_text)

            from openpyxl import load_workbook

            workbook = load_workbook(os.path.join(temporary, "not.xlsx"), read_only=True)
            worksheet = workbook["backend"]
            self.assertEqual(worksheet["A1"].value, "name")
            self.assertEqual(worksheet["A2"].value, "test_cc_not")

    def test_rejects_unknown_output_extension(self):
        with (
            mock.patch(
                "onnx_light.__main__._run_backend_test_timing",
                return_value={
                    "mode": "test",
                    "repeat": 1,
                    "warmup": 0,
                    "collection_seconds": 0.0,
                    "cases": [],
                    "total_seconds": 0.0,
                },
            ),
            self.assertRaisesRegex(SystemExit, "expected .csv or .xlsx"),
        ):
            main(["backend", "--output", "report.json"])


if __name__ == "__main__":
    unittest.main()
