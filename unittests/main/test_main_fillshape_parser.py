# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Parser tests for ``python -m onnx_light fillshape`` options."""

from __future__ import annotations

import unittest

from onnx_light.ext_test_case import ExtTestCase

from onnx_light.__main__ import _build_parser


class TestMainFillshapeParser(ExtTestCase):
    def test_fillshape_release_info_option(self):
        """Parses ``--release-info`` on the ``fillshape`` subcommand."""
        parser = _build_parser()
        args = parser.parse_args(["fillshape", "model.onnx", "--release-info"])
        self.assertTrue(args.release_info)
        self.assertFalse(args.inplace_info)


if __name__ == "__main__":
    unittest.main()
