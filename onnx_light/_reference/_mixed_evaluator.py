# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Mixed native and NumPy reference evaluator."""

from functools import partial

from ._evaluator import ReferenceEvaluator
from .ops import _attention, _conv, _gemm, _matmul


class MixedReferenceEvaluator(ReferenceEvaluator):
    """Uses NumPy for slow native operators and native kernels for the rest.

    Registers per-session NumPy implementations for ``Gemm``, ``MatMul``,
    ``Conv``, and ``Attention`` before the first run. The evaluator retains
    the :class:`ReferenceEvaluator` API; other operators still use native
    kernels. Actual performance depends on the model and tensor sizes.
    """

    def __init__(self, proto, **kwargs):
        """Initializes the evaluator and registers NumPy implementations."""
        super().__init__(proto, **kwargs)
        for op_type, kernel in (
            ("Gemm", partial(_gemm, opset=self._opsets.get("", self._opsets.get("ai.onnx", 13)))),
            ("MatMul", _matmul),
            ("Conv", _conv),
            ("Attention", _attention),
        ):
            self.register_custom_kernel("", op_type, kernel)
