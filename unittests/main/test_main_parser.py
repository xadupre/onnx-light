# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for the top-level command parser."""

import unittest

from onnx_light.ext_test_case import ExtTestCase
from contextlib import redirect_stderr
from io import StringIO

from onnx_light.__main__ import _build_parser


class TestMainParser(ExtTestCase):
    def test_kernel_help_contains_examples(self):
        help_text = _build_parser()._subparsers._group_actions[0].choices["kernel"].format_help()
        self.assertIn("examples:", help_text)
        self.assertIn("python -m onnx_light kernel --list", help_text)
        self.assertIn("--parameter parallel.minimum_tasks=default,2,4", help_text)

    def test_backend_help_contains_examples(self):
        help_text = _build_parser()._subparsers._group_actions[0].choices["backend"].format_help()
        self.assertIn("examples:", help_text)
        self.assertIn("--output backend-not.xlsx", help_text)
        self.assertIn("--save-models backend-models", help_text)
        self.assertIn("--parameter parallel.minimum_elements=default,16384,32768", help_text)

    def test_rejects_removed_kernel_baseline_command(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            _build_parser().parse_args(["kernel-baseline"])

    def test_rejects_renamed_commands(self):
        for command in ("tune", "tune-kernels", "backend-test"):
            with (
                self.subTest(command=command),
                redirect_stderr(StringIO()),
                self.assertRaises(SystemExit),
            ):
                _build_parser().parse_args([command])


if __name__ == "__main__":
    unittest.main()
