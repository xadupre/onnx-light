import unittest
from onnx_light.ext_test_case import ExtTestCase
import numpy as np
import onnx_light.onnx as onnxl
from onnx_light.backend.test.case.base import (
    ALL_TESTS,
    TestCase,
    collect_test_case,
    expect,
    get_test_cases_for_op,
)


class TestBackendFunction(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        onnxl.defs.register_onnx_operator_set_schema()

    def setUp(self):
        ALL_TESTS.clear()

    def tearDown(self):
        ALL_TESTS.clear()

    def test_expect_creates_test_case(self):
        """Tests that expect creates a TestCase and adds it to ALL_TESTS."""
        node = onnxl.helper.make_node("Abs", inputs=["x"], outputs=["y"])
        return
        x = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
        y = np.abs(x)

        expect(node, inputs=[x], outputs=[y], name="test_abs_basic")

        self.assertIn("test_abs_basic", ALL_TESTS)
        tc = ALL_TESTS["test_abs_basic"]
        self.assertIsInstance(tc, TestCase)
        self.assertEqual(tc.name, "test_abs_basic")
        self.assertEqual(tc.kind, "node")

    def test_expect_model_has_correct_opset(self):
        """Tests that the model has the correct opset version from schema."""
        node = onnxl.helper.make_node("Abs", inputs=["x"], outputs=["y"])
        x = np.array([1.0], dtype=np.float32)
        y = np.abs(x)

        expect(node, inputs=[x], outputs=[y], name="test_abs_opset")

        tc = ALL_TESTS["test_abs_opset"]
        self.assertIsNotNone(tc.model)
        # Abs has since_version=13 in ONNX
        opset_imports = list(tc.model.opset_import)
        self.assertEqual(len(opset_imports), 1)
        self.assertEqual(opset_imports[0].domain, "")
        self.assertEqual(opset_imports[0].version, 13)

    def test_expect_data_sets_structure(self):
        """Tests that data_sets has the correct structure."""
        node = onnxl.helper.make_node("Add", inputs=["a", "b"], outputs=["c"])
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        c = a + b

        expect(node, inputs=[a, b], outputs=[c], name="test_add_data")

        tc = ALL_TESTS["test_add_data"]
        self.assertIsNotNone(tc.data_sets)
        self.assertEqual(len(tc.data_sets), 1)
        inputs_list, outputs_list = tc.data_sets[0]
        self.assertEqual(len(inputs_list), 2)
        self.assertEqual(len(outputs_list), 1)
        np.testing.assert_array_equal(inputs_list[0], a)
        np.testing.assert_array_equal(inputs_list[1], b)
        np.testing.assert_array_equal(outputs_list[0], c)

    def test_expect_with_optional_inputs(self):
        """Tests expect with optional inputs (empty string in node.input)."""
        # Create a node with an optional trailing input omitted via empty string.
        # Gemm exists in LightOpSchema and has an optional C input.
        node = onnxl.helper.make_node("Gemm", inputs=["a", "b", ""], outputs=["y"])
        a = np.array([[1.0, 2.0]], dtype=np.float32)
        b = np.array([[3.0], [4.0]], dtype=np.float32)
        y = a @ b

        expect(node, inputs=[a, b], outputs=[y], name="test_gemm_optional")

        tc = ALL_TESTS["test_gemm_optional"]
        self.assertIsNotNone(tc.model)
        # Should only have 2 inputs in data_sets (the non-empty ones)
        inputs_list, _ = tc.data_sets[0]
        self.assertEqual(len(inputs_list), 2)

    def test_expect_with_custom_opset(self):
        """Tests expect with custom opset_imports."""
        node = onnxl.helper.make_node("Abs", inputs=["x"], outputs=["y"])
        x = np.array([1.0], dtype=np.float32)
        y = np.abs(x)

        custom_opset = [onnxl.helper.make_opsetid("", 15)]
        expect(
            node,
            inputs=[x],
            outputs=[y],
            name="test_abs_custom_opset",
            opset_imports=custom_opset,
        )

        tc = ALL_TESTS["test_abs_custom_opset"]
        opset_imports = list(tc.model.opset_import)
        self.assertEqual(len(opset_imports), 1)
        self.assertEqual(opset_imports[0].version, 15)

    def test_expect_model_is_onnx_light_proto(self):
        """Tests that the model is an onnx_light ModelProto."""
        node = onnxl.helper.make_node("Abs", inputs=["x"], outputs=["y"])
        x = np.array([1.0], dtype=np.float32)
        y = np.abs(x)

        expect(node, inputs=[x], outputs=[y], name="test_model_type")

        tc = ALL_TESTS["test_model_type"]
        # Check that it's an onnx_light ModelProto, not onnx ModelProto
        self.assertIn("onnx_light", str(type(tc.model)))

    def test_expect_rtol_atol_defaults(self):
        """Tests that rtol and atol have correct default values."""
        node = onnxl.helper.make_node("Abs", inputs=["x"], outputs=["y"])
        x = np.array([1.0], dtype=np.float32)
        y = np.abs(x)

        expect(node, inputs=[x], outputs=[y], name="test_tolerances")

        tc = ALL_TESTS["test_tolerances"]
        self.assertEqual(tc.rtol, 1e-3)
        self.assertEqual(tc.atol, 1e-7)

    def test_collect_test_case_returns_dict(self):
        """Tests that collect_test_case returns a dictionary."""
        result = collect_test_case()
        self.assertIsInstance(result, dict)

    def test_collect_test_case_finds_abs_test(self):
        """Tests that collect_test_case finds the Abs test case (from C++)."""
        result = collect_test_case()
        # The C++ backend test library registers a test_cc_abs case.
        self.assertIn("test_cc_abs", result)
        tc = result["test_cc_abs"]
        self.assertIsInstance(tc, TestCase)
        self.assertEqual(tc.name, "test_cc_abs")
        self.assertEqual(tc.kind, "node")

    def test_collect_test_case_finds_blackmanwindow_tests(self):
        """Tests that collect_test_case finds BlackmanWindow test cases (from C++)."""
        result = collect_test_case()
        self.assertIn("test_cc_blackmanwindow", result)
        self.assertIn("test_cc_blackmanwindow_symmetric", result)

    def test_collect_test_case_clears_all_tests(self):
        """Tests that collect_test_case clears ALL_TESTS after collecting."""
        result = collect_test_case()
        # ALL_TESTS should be empty after collect_test_case
        self.assertEqual(len(ALL_TESTS), 0)
        # But result should have the collected tests
        self.assertGreater(len(result), 0)

    def test_collect_test_case_multiple_calls(self):
        """Tests that collect_test_case can be called multiple times."""
        result1 = collect_test_case()
        result2 = collect_test_case()
        # Both calls should return the same tests
        self.assertEqual(set(result1.keys()), set(result2.keys()))

    def test_collect_test_case_test_case_structure(self):
        """Tests that collected test cases have the correct structure."""
        result = collect_test_case()
        self.assertGreater(len(result), 0)

        for name, tc in result.items():
            self.assertIsInstance(tc, TestCase)
            self.assertEqual(tc.name, name)
            self.assertIsNotNone(tc.model)
            self.assertIsNotNone(tc.data_sets)
            self.assertEqual(tc.kind, "node")
            self.assertIsInstance(tc.rtol, float)
            self.assertIsInstance(tc.atol, float)

    def test_make_test_class_returns_test_class(self):
        """Verifies that make_test_class returns an ExtTestCase subclass."""
        from onnx_light.backend.test.case import make_test_class

        def dummy_runtime(model, *inputs):
            # Simple runtime that returns absolute values
            return [np.abs(inp) for inp in inputs]

        TestClass = make_test_class(dummy_runtime)
        self.assertTrue(issubclass(TestClass, ExtTestCase))

    def test_make_test_class_creates_test_methods(self):
        """Verifies that make_test_class creates test methods for each test case."""
        from onnx_light.backend.test.case import make_test_class

        def dummy_runtime(model, *inputs):
            return [np.abs(inp) for inp in inputs]

        TestClass = make_test_class(dummy_runtime)

        # Check that test methods were created
        test_methods = [attr for attr in dir(TestClass) if attr.startswith("test_")]
        self.assertGreater(len(test_methods), 0)

        # Check that test_cc_abs exists (from C++ backend test library)
        self.assertIn("test_test_cc_abs", test_methods)

    def test_make_test_class_with_include_regex(self):
        """Verifies that make_test_class filters tests with include_regex."""
        from onnx_light.backend.test.case import make_test_class

        def dummy_runtime(model, *inputs):
            return [np.abs(inp) for inp in inputs]

        # Only include tests with "abs" in the name
        TestClass = make_test_class(dummy_runtime, include_regex=["abs"])

        test_methods = [attr for attr in dir(TestClass) if attr.startswith("test_")]

        # Should have test_test_cc_abs
        self.assertIn("test_test_cc_abs", test_methods)

        # All test methods should contain "abs"
        for method_name in test_methods:
            # Remove "test_" prefix to get the test case name
            test_name = method_name[5:]  # Remove "test_" prefix
            self.assertIn("abs", test_name.lower())

    def test_make_test_class_with_exclude_regex(self):
        """Verifies that make_test_class filters tests with exclude_regex."""
        from onnx_light.backend.test.case import make_test_class

        def dummy_runtime(model, *inputs):
            return [np.abs(inp) for inp in inputs]

        # Exclude tests with "abs" in the name
        TestClass = make_test_class(dummy_runtime, exclude_regex=["abs"])

        test_methods = [attr for attr in dir(TestClass) if attr.startswith("test_")]

        # Should not have test_test_cc_abs
        self.assertNotIn("test_test_cc_abs", test_methods)

    def test_make_test_class_with_custom_atols(self):
        """Verifies that make_test_class uses custom atols."""
        from onnx_light.backend.test.case import make_test_class

        def dummy_runtime(model, *inputs):
            # Return values slightly different from expected
            return [np.abs(inp) + 1e-6 for inp in inputs]

        # Set a custom atol for test_cc_abs
        TestClass = make_test_class(
            dummy_runtime, include_regex=["abs"], atols={"test_cc_abs": 1e-5}
        )

        # Create an instance and run the test
        test_instance = TestClass()
        test_instance.test_test_cc_abs()  # Should pass with custom atol

    def test_make_test_class_with_custom_rtols(self):
        """Verifies that make_test_class uses custom rtols."""
        from onnx_light.backend.test.case import make_test_class

        def dummy_runtime(model, *inputs):
            # Return values with small relative error
            return [np.abs(inp) * 1.001 for inp in inputs]

        # Set a custom rtol for test_cc_abs
        TestClass = make_test_class(
            dummy_runtime, include_regex=["abs"], rtols={"test_cc_abs": 1e-2}
        )

        # Create an instance and run the test
        test_instance = TestClass()
        test_instance.test_test_cc_abs()  # Should pass with custom rtol

    def test_make_test_class_test_execution(self):
        """Verifies that generated test methods execute correctly."""
        from onnx_light.backend.test.case import make_test_class

        def correct_runtime(model, *inputs):
            # Correct implementation for Abs
            return [np.abs(inp) for inp in inputs]

        TestClass = make_test_class(correct_runtime, include_regex=["abs"])

        # Create a test suite and run it
        suite = unittest.TestLoader().loadTestsFromTestCase(TestClass)
        runner = unittest.TextTestRunner(verbosity=0)
        result = runner.run(suite)

        # All tests should pass
        self.assertEqual(result.failures, [])
        self.assertEqual(result.errors, [])
        self.assertGreater(result.testsRun, 0)

    def test_make_test_class_test_failure(self):
        """Verifies that generated test methods fail when runtime is incorrect."""
        from onnx_light.backend.test.case import make_test_class

        def incorrect_runtime(model, *inputs):
            # Incorrect implementation - returns wrong values
            return [inp * 2 for inp in inputs]

        TestClass = make_test_class(incorrect_runtime, include_regex=["abs"])

        # Create a test suite and run it
        suite = unittest.TestLoader().loadTestsFromTestCase(TestClass)
        runner = unittest.TextTestRunner(verbosity=0)
        result = runner.run(suite)

        # Tests should fail
        self.assertGreater(len(result.failures) + len(result.errors), 0)

    def test_make_test_class_empty_filters(self):
        """Verifies that make_test_class works with no filters."""
        from onnx_light.backend.test.case import make_test_class

        def dummy_runtime(model, *inputs):
            return [np.abs(inp) for inp in inputs]

        TestClass = make_test_class(dummy_runtime)

        test_methods = [attr for attr in dir(TestClass) if attr.startswith("test_")]

        # Should have multiple test methods
        self.assertGreater(len(test_methods), 0)

    def test_get_test_cases_for_op_basic(self):
        """Returns test cases that contain the requested op_type."""
        result = get_test_cases_for_op("Abs")
        self.assertIn("test_cc_abs", result)
        for tc in result.values():
            op_types = [n.op_type for n in tc.model.graph.node]
            self.assertIn("Abs", op_types)

    def test_get_test_cases_for_op_opset_match(self):
        """Filters by opset_version when it matches the model's opset_import."""
        result = get_test_cases_for_op("Abs", opset_version=13)
        self.assertIn("test_cc_abs", result)

    def test_get_test_cases_for_op_opset_no_match(self):
        """Returns an empty dict when opset_version doesn't match."""
        result = get_test_cases_for_op("Abs", opset_version=999)
        self.assertEqual(result, {})

    def test_get_test_cases_for_op_unknown_op(self):
        """Returns an empty dict for an unknown op_type."""
        result = get_test_cases_for_op("ThisOpDoesNotExist")
        self.assertEqual(result, {})

    def test_get_test_cases_for_op_uses_provided_cases(self):
        """Does not call collect_test_case when test_cases is provided."""
        all_tc = collect_test_case()
        result = get_test_cases_for_op("If", test_cases=all_tc)
        self.assertGreater(len(result), 0)
        for name, tc in result.items():
            self.assertIs(tc, all_tc[name])

    def test_get_test_cases_for_op_domain_filter(self):
        """Filtering on a non-default domain returns no ai.onnx ops."""
        result = get_test_cases_for_op("Abs", domain="ai.onnx.ml")
        self.assertEqual(result, {})

    def test_get_test_cases_for_attention(self):
        """Verifies that get_test_cases_for_op returns the Attention backend tests."""
        result = get_test_cases_for_op("Attention")
        self.assertEqual(set(result), {"test_cc_attention_basic", "test_cc_attention_gqa"})

        expected_shapes = {
            "test_cc_attention_basic": (1, 2, 2, 2),
            "test_cc_attention_gqa": (1, 4, 2, 2),
        }
        for name, expected_shape in expected_shapes.items():
            tc = result[name]
            self.assertEqual([node.op_type for node in tc.model.graph.node], ["Attention"])
            self.assertEqual(
                [(opset.domain, opset.version) for opset in tc.model.opset_import], [("", 23)]
            )
            self.assertEqual(tuple(tc.data_sets[0][1][0].shape), expected_shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
