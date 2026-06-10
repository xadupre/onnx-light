import re
from typing import Pattern, Union
from ..onnx_py._onnxpybackend.backend_test import TestCase, collect_test_cases  # type: ignore # noqa: F401
from ..onnx_lib.backend.test.case import collect_test_case, make_test_class  # type: ignore # noqa: F401


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
    from ..onnx_py._onnxpybackend import backend_test as _C  # type: ignore[attr-defined]

    return _C.collect_test_cases_by_name(source)
