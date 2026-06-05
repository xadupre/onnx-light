"""Python re-export of the C++ ``backend_test`` bindings.

This sub-package exposes the runtime data model used by the C++ backend
test infrastructure (``onnx_backend_test``) through a stable Python
import path (``onnx_light.backend_test``).

The underlying objects are implemented in C++
(``onnx_light/onnx_backend_test/test_case.{h,cc}``,
``onnx_light/onnx_backend_test/simple_tensor.{h,cc}``) and bound to
Python via :mod:`onnx_light.onnx_py._onnxbackend`.

It mirrors :mod:`onnx_light.backend` (which exposes the deterministic
pseudo-random helpers) and is what ``onnx_light.backend.test.case``
builds higher-level helpers on top of.
"""

from __future__ import annotations

import re
from typing import Pattern, Union

from ..onnx_py._onnxpy import backend_test as _C  # type: ignore[attr-defined]

# Core data model.
DataSet = _C.DataSet
Tensor = _C.Tensor
TestCase = _C.TestCase

# Helper that collects every C++-registered backend test case.
collect_test_cases = _C.collect_test_cases


def collect_test_cases_by_name(pattern: Union[str, Pattern[str]]) -> list[TestCase]:
    """Returns the C++-implemented backend test cases whose name matches *pattern*.

    Args:
        pattern: A regular expression (as a string or a pre-compiled
            :class:`re.Pattern`) matched against :attr:`TestCase.name`
            with :func:`re.search`. Use ``"^...$"`` to require a full
            match.

    Returns:
        The list of :class:`TestCase` instances (in their natural
        registration order) whose ``name`` matches *pattern*.

    Raises:
        TypeError: If *pattern* is neither a string nor a compiled
            regular expression.
        re.error: If *pattern* is an invalid regular expression.
    """
    if isinstance(pattern, str):
        compiled = re.compile(pattern)
    elif isinstance(pattern, re.Pattern):
        compiled = pattern
    else:
        raise TypeError(
            "pattern must be a str or a compiled re.Pattern, "
            f"got {type(pattern).__name__}."
        )
    return [tc for tc in collect_test_cases() if compiled.search(tc.name)]


__all__ = [
    "DataSet",
    "Tensor",
    "TestCase",
    "collect_test_cases",
    "collect_test_cases_by_name",
]
