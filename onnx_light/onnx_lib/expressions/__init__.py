"""Symbolic dimension expression utilities backed by a C++ library.

This package re-exports the expression functions exposed by the
``_onnxpy.expressions`` C++ extension submodule so they are accessible as
``onnx_light.onnx_lib.expressions.<name>`` (and via the backward-compatibility
alias ``onnx_light.onnx.expressions``).
"""

from __future__ import annotations

from ...onnx_proto._onnxpy import expressions as _C  # type: ignore[attr-defined]

simplify_expression = _C.simplify_expression
simplify_two_expressions = _C.simplify_two_expressions
evaluate_expression = _C.evaluate_expression
parse_expression_tokens = _C.parse_expression_tokens
rename_expression = _C.rename_expression
rename_dynamic_expression = _C.rename_dynamic_expression
dim_add = _C.dim_add
dim_sub = _C.dim_sub
dim_mul = _C.dim_mul
dim_multi_mul = _C.dim_multi_mul
dim_div = _C.dim_div
dim_mod = _C.dim_mod
dim_max = _C.dim_max
dim_min = _C.dim_min

__all__ = [
    "dim_add",
    "dim_div",
    "dim_max",
    "dim_min",
    "dim_mod",
    "dim_mul",
    "dim_multi_mul",
    "dim_sub",
    "evaluate_expression",
    "parse_expression_tokens",
    "rename_dynamic_expression",
    "rename_expression",
    "simplify_expression",
    "simplify_two_expressions",
]
