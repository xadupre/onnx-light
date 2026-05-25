import re
import unittest

import onnx_light.onnx as onnxl
import onnx_light.onnx.checker as checker
from onnx_light.backend.test.case import collect_test_case
from onnx_light.ext_test_case import ExtTestCase


def make_checker_test_class(
    include_regex: list[str] | None = None,
    exclude_regex: list[str] | None = None,
):
    """Builds a TestCase class running ``checker.check_model`` on every model
    produced by :func:`onnx_light.backend.test.case.collect_test_case`.

    Mirrors :func:`onnx_light.backend.test.case.make_test_class` but skips
    inference and only validates that the generated ``ModelProto`` passes the
    ONNX checker.
    """
    onnxl.defs.register_onnx_operator_set_schema()
    all_tests = collect_test_case()

    filtered = {}
    for name, tc in all_tests.items():
        if exclude_regex and any(re.search(p, name) for p in exclude_regex):
            continue
        if include_regex and not any(re.search(p, name) for p in include_regex):
            continue
        filtered[name] = tc

    class BackendCheckerTest(ExtTestCase):
        """Dynamically generated tests running the ONNX checker on backend
        test models."""

    for name, tc in filtered.items():

        def test_func(self, tc=tc):
            self.assertIsNotNone(tc.model)
            checker.check_model(tc.model)

        test_func.__name__ = f"test_{name}"
        test_func.__doc__ = f"Run check_model on the model of test case {name!r}."
        setattr(BackendCheckerTest, f"test_{name}", test_func)

    return BackendCheckerTest


TestCheckerBackend = make_checker_test_class()


if __name__ == "__main__":
    unittest.main(verbosity=2)
