# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Backend tests that exercise :class:`onnx_light.reference.ReferenceEvaluator`
against every backend test case.

This is the Python counterpart of
``test_backend_with_optim_shape_inference.py``: it walks the same backend
test registry produced by :func:`onnx_light.backend.test.case.make_test_class`
and validates that :class:`ReferenceEvaluator` reproduces the expected
outputs without discrepancies.

The evaluator wraps the
``RuntimeSession`` / ``ExecutionPlan`` execution machinery exposed by
:mod:`onnx_light.onnx_py._onnxpykernels`, so the
coverage here is a superset of ``test_backend_with_run_model.py`` (which
only exercises single-node graphs) and a useful cross-check of the Python
``ReferenceEvaluator`` facade against the upstream expected outputs.
"""

from __future__ import annotations

import unittest

import numpy as np

from onnx_light.ext_test_case import ExtTestCase, import_or_skip

import onnx_light.onnx as onnxl

# The kernels runtime and backend test registries are only available in the
# full build; skip this module on a reduced build (ONNX_LIGHT_BUILD_KERNELS=OFF).
_backend_case = import_or_skip("onnx_light.onnx_lib.backend.test.case")
collect_test_case = _backend_case.collect_test_case
make_test_class = _backend_case.make_test_class
ReferenceEvaluator = import_or_skip("onnx_light.onnx.reference", "ReferenceEvaluator")


def reference_evaluator_backend(model: onnxl.ModelProto, *inputs: np.ndarray) -> list[np.ndarray]:
    """Runs ``model`` through :class:`ReferenceEvaluator`.

    Mirrors the signature expected by :func:`make_test_class` (the same as
    ``onnxruntime_backend`` in ``test_backend_with_onnxruntime.py``):
    positional ``inputs`` are wired to the model's declared graph inputs by
    name and the outputs are returned as numpy arrays in graph-output order.
    """
    sess = ReferenceEvaluator(model)
    feed = dict(zip(sess.input_names, inputs))
    return sess.run(None, feed)


def _iter_ops(graph) -> list[str]:
    """Returns every ``op_type`` referenced (recursively) by ``graph``.

    Subgraphs nested inside attributes (``If``/``Loop``/``Scan`` bodies)
    are walked as well so the include filter rejects models whose
    control-flow bodies use unimplemented ops.
    """
    ops: list[str] = []
    for node in graph.node:
        ops.append(node.op_type)
        for attr in node.attribute:
            if attr.type == onnxl.AttributeProto.GRAPH:
                ops.extend(_iter_ops(attr.g))
            elif attr.type == onnxl.AttributeProto.GRAPHS:
                for g in attr.graphs:
                    ops.extend(_iter_ops(g))
    return ops


TestReferenceEvaluatorBackend = make_test_class(
    reference_evaluator_backend,
    exclude_regex=[
        "image_decoder_decode_jpeg_bgr",
        "image_decoder_decode_jpeg_grayscale",
        "image_decoder_decode_jpeg_rgb",
    ],
)


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
            np.testing.assert_array_equal(expected, actual)


if __name__ == "__main__":
    unittest.main(verbosity=2)
