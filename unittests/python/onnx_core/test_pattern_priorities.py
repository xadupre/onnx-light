# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests built-in pattern priorities without requiring runtime kernels."""

import unittest

from onnx_light.ext_test_case import ExtTestCase, import_or_skip

optim = import_or_skip("onnx_light.onnx_core.optimization")


class TestPatternPriorities(ExtTestCase):
    def test_builtin_pattern_priorities(self):
        for pattern_type in (
            optim.GroupedMatMulPattern,
            optim.GemmSumFusionPattern,
            optim.MatMulAddPattern,
            optim.MatMulBatchNormalizationFusionPattern,
            optim.MatMulScaleFusionPattern,
        ):
            with self.subTest(pattern=pattern_type.__name__):
                self.assertEqual(pattern_type().priority, 2)
                self.assertEqual(pattern_type(priority=3).priority, 3)
        self.assertEqual(optim.CastPattern().priority, 0)
        self.assertEqual(optim.CastCastPattern().priority, 1)


if __name__ == "__main__":
    unittest.main()
