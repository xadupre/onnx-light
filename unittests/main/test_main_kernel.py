# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``onnx-light kernel``."""

from __future__ import annotations

import argparse
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from onnx_light.__main__ import _build_parser, _parse_kernel_device, main
from onnx_light.ext_test_case import import_or_skip

runtime = import_or_skip("onnx_light.onnx_py._onnxpykernels", "runtime")


class TestMainKernel(unittest.TestCase):
    def test_parser_requires_mode(self):
        parser = _build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["kernel"])

    def test_filters_default_to_all(self):
        args = _build_parser().parse_args(["kernel", "--kernel", "Gemm"])
        self.assertIsNone(args.library)
        self.assertIsNone(args.device)
        self.assertIsNone(args.dtype)
        self.assertIsNone(args.impl)
        self.assertFalse(args.tune)
        self.assertFalse(args.verbose)

    def test_lists_registered_kernels(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            main(["kernel", "--list"])
        identifiers = buffer.getvalue().splitlines()
        self.assertEqual(identifiers, sorted(identifiers))
        self.assertIn("ai.onnx:Gemm", identifiers)

    def test_lists_registered_kernels_as_json(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            main(["kernel", "--list", "--json"])
        report = json.loads(buffer.getvalue())
        self.assertIn("ai.onnx:Gemm", report["kernels"])

    def test_shows_multiple_kernel_tunables(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            main(["kernel", "--kernel", "Abs", "--kernel", "Gemm"])
        output = buffer.getvalue()
        self.assertIn("ai.onnx:Abs", output)
        self.assertIn("ai.onnx:Gemm", output)
        self.assertIn("parallel.minimum_elements", output)
        self.assertIn("library=onnx_light", output)
        self.assertIn("device=CPU", output)

    def test_filters_tunables_by_library(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            main(["kernel", "--kernel", "Abs", "--library", "unknown_library", "--json"])
        report = json.loads(buffer.getvalue())
        self.assertEqual(report["kernels"][0]["library"], "unknown_library")
        self.assertEqual(report["kernels"][0]["device_name"], "CPU")
        self.assertEqual(report["kernels"][0]["tunables"], [])

    def test_filters_tunables_by_dtype_and_implementation(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            main(
                ["kernel", "--kernel", "Gemm", "--dtype", "FLOAT", "--impl", "portable", "--json"]
            )
        report = json.loads(buffer.getvalue())
        self.assertEqual(
            report["selection"],
            {"device": "all", "dtype": "FLOAT", "implementation": "portable", "library": "all"},
        )
        tunables = report["kernels"][0]["tunables"]
        self.assertGreater(len(tunables), 0)
        self.assertEqual({item["element_type"] for item in tunables}, {1})
        self.assertEqual({item["implementation"] for item in tunables}, {"portable"})

    def test_rejects_kernel_not_registered_for_device(self):
        with self.assertRaisesRegex(SystemExit, "device GPU0.*Abs"):
            main(["kernel", "--kernel", "Abs", "--device", "GPU0"])

    def test_parses_devices(self):
        self.assertEqual(_parse_kernel_device("CPU"), -1)
        self.assertEqual(_parse_kernel_device("Undefined"), -2)
        self.assertEqual(_parse_kernel_device("GPU0"), 0)
        self.assertEqual(_parse_kernel_device("GPU8191"), 8191)
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "unknown device"):
            _parse_kernel_device("GPU8192")

    def test_reports_fixed_kernel(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            main(["kernel", "--kernel", "Identity"])
        output = buffer.getvalue()
        self.assertIn("library=all device=CPU dtype=all impl=all", output)
        self.assertIn("no tunable parameters", output)

    def test_rejects_unknown_kernel(self):
        with self.assertRaisesRegex(SystemExit, "unknown kernel.*DoesNotExist"):
            main(["kernel", "--kernel", "DoesNotExist"])

    def test_tune_rejects_multiple_kernels(self):
        with self.assertRaisesRegex(SystemExit, "exactly one selected kernel"):
            main(["kernel", "--kernel", "Abs", "--kernel", "Gemm", "--tune"])

    def test_tune_rejects_fixed_kernel(self):
        with self.assertRaisesRegex(SystemExit, "no tunable parameters"):
            main(["kernel", "--kernel", "Identity", "--tune"])

    def test_tune_shows_before_after_and_progress(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        calibration = {
            "calibrated": [],
            "skipped": [],
            "unsupported": [],
            "diagnostics": [],
            "candidate_diagnostics": [],
            "cache_update": None,
            "published_generation": 0,
        }
        with (
            mock.patch(
                "onnx_light.kernel_tuning.calibrate_kernel_tuning", return_value=calibration
            ) as calibrate,
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            main(["kernel", "--kernel", "Abs", "--dtype", "FLOAT", "--tune", "--verbose"])
        self.assertIn("before:", stdout.getvalue())
        self.assertIn("after:", stdout.getvalue())
        self.assertIn("captured parameters before tuning", stderr.getvalue())
        self.assertIn("calibrating 1/1", stderr.getvalue())
        self.assertIn("captured parameters after tuning", stderr.getvalue())
        calibrate.assert_called_once()
        self.assertEqual(calibrate.call_args.kwargs["device"], -1)
        self.assertEqual(calibrate.call_args.kwargs["element_types"], [1])

    def test_tune_json_contains_before_and_after(self):
        stdout = io.StringIO()
        calibration = {
            "calibrated": [],
            "skipped": [],
            "unsupported": [],
            "diagnostics": [],
            "candidate_diagnostics": [],
            "cache_update": None,
            "published_generation": 0,
        }
        with (
            mock.patch(
                "onnx_light.kernel_tuning.calibrate_kernel_tuning", return_value=calibration
            ),
            redirect_stdout(stdout),
        ):
            main(["kernel", "--kernel", "Abs", "--dtype", "FLOAT", "--tune", "--json"])
        report = json.loads(stdout.getvalue())
        self.assertEqual(len(report["before"]), 1)
        self.assertEqual(len(report["calibrations"]), 1)
        self.assertEqual(len(report["after"]), 1)

    def test_selected_kernel_json_is_deterministic(self):
        arguments = ["kernel", "--kernel", "Gemm", "--kernel", "Abs", "--json"]
        outputs = []
        for _ in range(2):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                main(arguments)
            outputs.append(buffer.getvalue())
        self.assertEqual(outputs[0], outputs[1])
        report = json.loads(outputs[0])
        self.assertEqual(
            report["selection"],
            {"device": "all", "dtype": "all", "implementation": "all", "library": "all"},
        )
        self.assertEqual(
            [kernel["identifier"] for kernel in report["kernels"]],
            ["ai.onnx:Abs", "ai.onnx:Gemm"],
        )
        self.assertTrue(
            all(
                {"library", "device", "device_name"} <= tunable.keys()
                for kernel in report["kernels"]
                for tunable in kernel["tunables"]
            )
        )


if __name__ == "__main__":
    unittest.main()
