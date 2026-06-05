import re
import unittest

from onnx_light.ext_test_case import ExtTestCase
import onnx_light.backend_test as bt


class TestCollectTestCasesByName(ExtTestCase):
    def test_collect_by_substring(self):
        cases = bt.collect_test_cases_by_name("abs")
        self.assertGreater(len(cases), 0)
        for tc in cases:
            self.assertIn("abs", tc.name)

    def test_collect_by_regex(self):
        cases = bt.collect_test_cases_by_name(r"^test_cc_add(_|$)")
        self.assertGreater(len(cases), 0)
        for tc in cases:
            self.assertRegex(tc.name, r"^test_cc_add(_|$)")

    def test_collect_no_match_returns_empty(self):
        cases = bt.collect_test_cases_by_name(r"__definitely_no_such_case__")
        self.assertEqual(cases, [])

    def test_collect_accepts_compiled_pattern(self):
        compiled = re.compile(r"abs")
        cases = bt.collect_test_cases_by_name(compiled)
        cases_str = bt.collect_test_cases_by_name("abs")
        self.assertEqual(
            sorted(tc.name for tc in cases),
            sorted(tc.name for tc in cases_str),
        )

    def test_collect_invalid_type_raises(self):
        with self.assertRaises(TypeError):
            bt.collect_test_cases_by_name(123)  # type: ignore[arg-type]

    def test_collect_invalid_regex_raises(self):
        with self.assertRaises(ValueError):
            bt.collect_test_cases_by_name("(")


if __name__ == "__main__":
    unittest.main(verbosity=2)
