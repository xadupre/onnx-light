import re
import unittest


from onnx_light.ext_test_case import ExtTestCase, import_or_skip

# The backend test registries are only available in the full build; skip this
# module on a reduced build (ONNX_LIGHT_BUILD_KERNELS=OFF).
bt = import_or_skip("onnx_light.onnx.backend")


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
        self.assertEqual(sorted(tc.name for tc in cases), sorted(tc.name for tc in cases_str))

    def test_collect_invalid_type_raises(self):
        with self.assertRaises(TypeError):
            bt.collect_test_cases_by_name(123)  # type: ignore[arg-type]

    def test_collect_invalid_regex_raises(self):
        with self.assertRaises(ValueError):
            bt.collect_test_cases_by_name("(")

    def test_collect_include_big(self):
        without_big = bt.collect_test_cases_by_name(
            r"^test_cc_shape_inference_big_qwen3_4_layers_like$"
        )
        with_big = bt.collect_test_cases_by_name(
            r"^test_cc_shape_inference_big_qwen3_4_layers_like$", include_big=True
        )
        self.assertEqual(without_big, [])
        self.assertEqual(
            [tc.name for tc in with_big], ["test_cc_shape_inference_big_qwen3_4_layers_like"]
        )

    def test_collect_test_case_accepts_mode(self):
        # ``mode`` is consumed by the C++ ``collect_test_cases`` binding, which
        # returns cases lazily (models and data sets are only materialized when
        # accessed). Exercise the real binding and read only ``name`` so the
        # test stays fast while still verifying that ``mode`` is accepted and
        # actually changes the generated cases.
        test_cases = bt.collect_test_cases(mode=bt.TestMode.TEST)
        self.assertGreater(len(test_cases), 0)
        test_names = {tc.name for tc in test_cases}
        # TEST mode does not emit the oversized benchmark cases.
        self.assertFalse(any(name.endswith("_benchmark") for name in test_names))

        benchmark_cases = bt.collect_test_cases(mode=bt.TestMode.BENCHMARK)
        self.assertGreater(len(benchmark_cases), 0)
        benchmark_names = {tc.name for tc in benchmark_cases}
        # BENCHMARK mode adds ``*_benchmark`` cases, proving ``mode`` reaches the
        # generator and changes its output.
        self.assertTrue(any(name.endswith("_benchmark") for name in benchmark_names))

        # The default (no ``mode``) matches ``TestMode.TEST``.
        default_names = {tc.name for tc in bt.collect_test_cases()}
        self.assertEqual(default_names, test_names)


if __name__ == "__main__":
    unittest.main(verbosity=2)
