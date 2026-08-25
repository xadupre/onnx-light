# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for the top-level command parser."""

import unittest
from contextlib import redirect_stderr
from io import StringIO

from onnx_light.__main__ import _build_parser


class TestMainParser(unittest.TestCase):
    def test_rejects_removed_kernel_baseline_command(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            _build_parser().parse_args(["kernel-baseline"])


if __name__ == "__main__":
    unittest.main()
