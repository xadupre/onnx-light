"""Python re-export of the C++ ``backend_test`` bindings.

This sub-package exposes the runtime data model used by the C++ backend
test infrastructure (``onnx_kernels``) through a stable Python
import path (``onnx_light.backend_test``).

The underlying objects are implemented in C++
(``onnx_light/onnx_kernels/test_case.{h,cc}``,
``onnx_light/onnx_kernels/simple_tensor.{h,cc}``) and bound to
Python via :mod:`onnx_light.onnx_py._onnxpybackend`.

It mirrors :mod:`onnx_light.backend` (which exposes the deterministic
pseudo-random helpers) and is what ``onnx_light.backend.test.case``
builds higher-level helpers on top of.
"""

from __future__ import annotations

import re
from typing import Any, Pattern, Union

from ..onnx_py._onnxpy import backend_test as _C  # type: ignore[attr-defined]

# Core data model.
DataSet: Any = _C.DataSet
Tensor: Any = _C.Tensor
TestCase: Any = _C.TestCase

# Helper that collects every C++-registered backend test case.
collect_test_cases = _C.collect_test_cases


def collect_test_cases_by_name(pattern: Union[str, Pattern[str]]) -> list[TestCase]:
    """Returns the C++-implemented backend test cases whose name matches *pattern*.

    The actual filtering happens in C++
    (``onnx_kernels::CollectTestCasesByName``) using
    ``std::regex_search`` with ECMAScript syntax. A compiled
    :class:`re.Pattern` is accepted for convenience and is forwarded as
    its source string.

    Args:
        pattern: A regular expression (as a string or a pre-compiled
            :class:`re.Pattern`) matched against :attr:`TestCase.name`.
            Use ``"^...$"`` to require a full match.

    Returns:
        The list of :class:`TestCase` instances (in their natural
        registration order) whose ``name`` matches *pattern*.

    Raises:
        TypeError: If *pattern* is neither a string nor a compiled
            regular expression.
        ValueError: If *pattern* is not a valid regular expression.
    """
    if isinstance(pattern, str):
        source = pattern
    elif isinstance(pattern, re.Pattern):
        source = pattern.pattern
    else:
        raise TypeError(
            "pattern must be a str or a compiled re.Pattern, got {type(pattern).__name__}."
        )
    return _C.collect_test_cases_by_name(source)


__all__ = ["DataSet", "Tensor", "TestCase", "collect_test_cases", "collect_test_cases_by_name"]
