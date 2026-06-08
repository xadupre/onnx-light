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

The evaluator wraps the C++ ``RunModel`` / ``RunGraph`` / ``RunFunction``
dispatcher exposed by :mod:`onnx_light.onnx_py._onnxpykernels`, so the
coverage here is a superset of ``test_backend_with_run_model.py`` (which
only exercises single-node graphs) and a useful cross-check of the Python
``ReferenceEvaluator`` facade against the upstream expected outputs.
"""

from __future__ import annotations

import re
import unittest

import numpy as np

import onnx_light.onnx as onnxl
from onnx_light.backend.test.case import collect_test_case, make_test_class
from onnx_light.reference import ReferenceEvaluator

# Operators currently registered in
# ``onnx_light/onnx_kernels/run_nodes.cc::KernelDispatchTable``. Backend
# test cases whose graph(s) only use these ops are the only ones
# :class:`ReferenceEvaluator` can execute today. The set mirrors
# ``_IMPLEMENTED_OPS`` in ``test_backend_with_run_model.py``.
_IMPLEMENTED_OPS: frozenset[str] = frozenset({"Abs", "Neg", "Add", "Sub", "Mul", "Div"})


def reference_evaluator_backend(
    model: onnxl.ModelProto, *inputs: np.ndarray
) -> list[np.ndarray]:
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


def _build_include_regex() -> list[str]:
    """Returns the include-regex list selecting every backend test case
    whose graph(s) only contain operators implemented by the C++
    ``KernelDispatchTable``.

    Building the list once at module load avoids re-walking the registry
    from inside :func:`make_test_class`.
    """
    names: list[str] = []
    for name, tc in collect_test_case().items():
        if tc.model is None:
            continue
        ops = _iter_ops(tc.model.graph)
        if ops and all(op in _IMPLEMENTED_OPS for op in ops):
            names.append(name)
    if not names:
        # Fallback: keep a regex that matches nothing so ``make_test_class``
        # generates an empty test class instead of every case in the registry.
        return [r"^$"]
    # Anchor each name to avoid accidental substring matches.
    return [r"^" + re.escape(n) + r"$" for n in names]


TestReferenceEvaluatorBackend = make_test_class(
    reference_evaluator_backend,
    include_regex=_build_include_regex(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
