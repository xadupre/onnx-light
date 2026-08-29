import re
from typing import Pattern, Union

try:
    from ..onnx_py._onnxpybackend.backend_test import (  # type: ignore # noqa: F401
        TestCase,
        TestMode,
        collect_test_cases,
    )
except ImportError as exc:  # pragma: no cover - exercised only in reduced builds
    raise ImportError(
        "onnx-light was built without the backend-test extensions "
        "(ONNX_LIGHT_BUILD_KERNELS=OFF); install the full build to use the "
        "backend test cases."
    ) from exc
from ..onnx_lib.backend.test.case import collect_test_case, get_test_case, make_test_class  # type: ignore # noqa: F401


def collect_test_cases_by_name(
    pattern: Union[str, Pattern[str]],
    include_big: bool = False,
    mode: "TestMode | None" = None,
    generate_benchmark_expected_outputs: bool = False,
) -> list[TestCase]:
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
        include_big: When ``True``, includes backend test cases whose name
            contains ``"_big_"``. Defaults to ``False``, which keeps these
            big cases excluded.
        mode: Selects the generation mode. ``TestMode.TEST`` (the default
            when ``None``) yields the standard correctness cases;
            ``TestMode.BENCHMARK`` yields large benchmark-sized cases where
            supported.

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

    if mode is None:
        mode = _C.TestMode.TEST
    return _C.collect_test_cases_by_name(
        source,
        include_big=include_big,
        mode=mode,
        generate_benchmark_expected_outputs=generate_benchmark_expected_outputs,
    )
