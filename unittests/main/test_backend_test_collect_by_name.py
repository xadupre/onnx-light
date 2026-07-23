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
        from unittest import mock

        from onnx_light.onnx.backend import TestMode, collect_test_case
        from onnx_light.onnx_lib.backend.test.case import base as case_base

        # ``collect_test_case`` must accept ``mode`` and forward it to the C++
        # binding. Materializing every case (models + data sets, which are huge
        # for BENCHMARK) makes this test slow, so spy on the binding to assert
        # the forwarded mode without building any case.
        with mock.patch.object(
            case_base._backend_test_cc, "collect_test_cases", return_value=[]
        ) as spy:
            collect_test_case()
            collect_test_case(mode=TestMode.TEST)
            collect_test_case(mode=TestMode.BENCHMARK)
        forwarded = [call.kwargs["mode"] for call in spy.call_args_list]
        # A ``None`` mode defaults to TEST; BENCHMARK must be forwarded as-is.
        self.assertEqual(forwarded, [TestMode.TEST, TestMode.TEST, TestMode.BENCHMARK])

        # Smoke-check the real C++ binding lazily (``len`` only, no model/data
        # materialization) so that broken generation is still caught quickly.
        self.assertGreater(
            len(case_base._backend_test_cc.collect_test_cases(mode=TestMode.TEST)), 0
        )
        self.assertGreater(
            len(case_base._backend_test_cc.collect_test_cases(mode=TestMode.BENCHMARK)), 0
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
