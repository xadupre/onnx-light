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

from onnx_light.ext_test_case import import_or_skip

import onnx_light.onnx as onnxl

# The kernels runtime and backend test registries are only available in the
# full build; skip this module on a reduced build (ONNX_LIGHT_BUILD_KERNELS=OFF).
_backend_case = import_or_skip("onnx_light.onnx_lib.backend.test.case")
make_test_class = _backend_case.make_test_class
collect_test_case = _backend_case.collect_test_case
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


TestReferenceEvaluatorBackend = make_test_class(
    reference_evaluator_backend,
    exclude_regex=[
        "image_decoder_decode_jpeg_bgr",
        "image_decoder_decode_jpeg_grayscale",
        "image_decoder_decode_jpeg_rgb",
        "test_cc_release_partial_metadata",
    ],
)


class TestPartialReleaseMetadata(unittest.TestCase):
    """Checks the intentionally invalid release-metadata reproducer."""

    def test_incomplete_release_schedule_raises(self):
        """Raises when strict metadata omits an intermediate release."""
        case = collect_test_case(unload=False)["test_cc_release_partial_metadata"]
        self.addCleanup(case.unload)
        with self.assertRaisesRegex(
            RuntimeError, r"ExecutionPlan: result '?B'? is never released or unlocked"
        ):
            case.assert_allclose(reference_evaluator_backend, unload=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
