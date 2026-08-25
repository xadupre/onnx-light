# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``onnx-light kernel``."""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout

from onnx_light.__main__ import _build_parser, main
from onnx_light.ext_test_case import import_or_skip

runtime = import_or_skip("onnx_light.onnx_py._onnxpykernels", "runtime")


class TestMainKernel(unittest.TestCase):
    def test_parser_requires_mode(self):
        parser = _build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["kernel"])

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

    def test_reports_fixed_kernel(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            main(["kernel", "--kernel", "Identity"])
        self.assertIn("no tunable parameters", buffer.getvalue())

    def test_rejects_unknown_kernel(self):
        with self.assertRaisesRegex(SystemExit, "unknown kernel.*DoesNotExist"):
            main(["kernel", "--kernel", "DoesNotExist"])

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
            [kernel["identifier"] for kernel in report["kernels"]],
            ["ai.onnx:Abs", "ai.onnx:Gemm"],
        )


if __name__ == "__main__":
    unittest.main()
