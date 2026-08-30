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

import numpy as np

from onnx_light.ext_test_case import import_or_skip

import onnx_light.onnx as onnxl

# The kernels runtime and backend test registries are only available in the
# full build; skip this module on a reduced build (ONNX_LIGHT_BUILD_KERNELS=OFF).
_backend_case = import_or_skip("onnx_light.onnx_lib.backend.test.case")
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
