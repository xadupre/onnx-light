# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``onnx-light backend-test``."""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from unittest import mock

from onnx_light.__main__ import _build_parser, main
from onnx_light.ext_test_case import import_or_skip
from onnx_light.tools.backend_test_timing import run_backend_test_timing

import_or_skip("onnx_light.onnx_py._onnxpybackend", "backend_test")
import_or_skip("onnx_light.onnx_py._onnxpykernels", "runtime")


class TestMainBackendTest(unittest.TestCase):
    def test_parser_defaults(self):
        args = _build_parser().parse_args(["backend-test"])
        self.assertEqual(args.regex, "")
        self.assertEqual(args.mode, "test")
        self.assertEqual(args.repeat, 1)
        self.assertEqual(args.warmup, 0)
        self.assertFalse(args.include_big)
        self.assertFalse(args.json)

    def test_parser_rejects_invalid_counts(self):
        parser = _build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["backend-test", "--repeat", "0"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["backend-test", "--warmup", "-1"])

    def test_times_one_correctness_case(self):
        report = run_backend_test_timing(name_regex=r"^test_cc_abs$", repeat=1)
        self.assertEqual(report["mode"], "test")
        self.assertEqual(report["selected"], 1)
        self.assertEqual(report["cases"][0]["name"], "test_cc_abs")
        self.assertEqual(report["cases"][0]["data_sets"], 1)
        self.assertEqual(len(report["cases"][0]["iteration_seconds"]), 1)
        self.assertGreaterEqual(report["cases"][0]["run_seconds"], 0)

    def test_times_one_benchmark_case(self):
        report = run_backend_test_timing(
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
            mock.patch(
                "onnx_light.tools.backend_test_timing.run_backend_test_timing",
                return_value=expected,
            ),
            redirect_stdout(buffer),
        ):
            main(
                [
                    "backend-test",
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
            run_backend_test_timing(repeat=0)
        with self.assertRaisesRegex(ValueError, "warmup must be non-negative"):
            run_backend_test_timing(warmup=-1)

    def test_rejects_invalid_regex(self):
        with self.assertRaises(ValueError):
            run_backend_test_timing(name_regex="(")


if __name__ == "__main__":
    unittest.main()
