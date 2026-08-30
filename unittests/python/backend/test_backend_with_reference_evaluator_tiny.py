# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests the ``tiny_llm`` decoder with :class:`onnx_light.reference.ReferenceEvaluator`."""

from __future__ import annotations

import unittest

import numpy as np

from onnx_light.ext_test_case import ExtTestCase, import_or_skip

# The kernels runtime and backend test registries are only available in the
# full build; skip this module on a reduced build (ONNX_LIGHT_BUILD_KERNELS=OFF).
collect_test_case = import_or_skip("onnx_light.onnx_lib.backend.test.case", "collect_test_case")
ReferenceEvaluator = import_or_skip("onnx_light.onnx.reference", "ReferenceEvaluator")


class TestReferenceEvaluatorTinyLlm(ExtTestCase):
    """Exercises the ``tiny_llm`` decoder through :class:`ReferenceEvaluator`.

    The ``test_cc_shape_inference_tiny_llm`` backend test case ships no
    ``data_sets``, so :func:`make_test_class` skips it. This class generates
    deterministic inputs and validates output shapes and dtypes.
    """

    @staticmethod
    def _tiny_llm_inputs():
        """Builds deterministic inputs for the ``tiny_llm`` decoder."""
        rng = np.random.RandomState(0)
        batch, seq, past = 1, 3, 2
        return {
            "input_ids": rng.randint(0, 32, (batch, seq)).astype(np.int64),
            "attention_mask": np.ones((batch, seq + past), dtype=np.int64),
            "past_key": rng.rand(batch, 4, past, 4).astype(np.float32),
            "past_value": rng.rand(batch, 4, past, 4).astype(np.float32),
        }

    def test_tiny_llm_end_to_end(self):
        """Runs the tiny_llm decoder end-to-end and checks output shapes."""
        tc = collect_test_case().get("test_cc_shape_inference_tiny_llm")
        self.assertIsNotNone(tc, "tiny_llm test case not found in backend registry")
        sess = ReferenceEvaluator(tc.model)
        outputs = sess.run(None, self._tiny_llm_inputs())
        self.assertEqual(len(outputs), 3)
        logits, present_key, present_value = outputs
        self.assertEqual(logits.dtype, np.float32)
        self.assertEqual(logits.shape, (1, 3, 32))
        self.assertEqual(present_key.shape, (1, 4, 5, 4))
        self.assertEqual(present_value.shape, (1, 4, 5, 4))

    def test_tiny_llm_deterministic(self):
        """Verifies that repeated runs produce identical results."""
        tc = collect_test_case().get("test_cc_shape_inference_tiny_llm")
        self.assertIsNotNone(tc)
        sess = ReferenceEvaluator(tc.model)
        inputs = self._tiny_llm_inputs()
        first = [np.array(out, copy=True) for out in sess.run(None, inputs)]
        second = sess.run(None, inputs)
        for expected, actual in zip(first, second):
            self.assertEqualArray(expected, actual)


if __name__ == "__main__":
    unittest.main(verbosity=2)
