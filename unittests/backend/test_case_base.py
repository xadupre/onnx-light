import unittest

import numpy as np

import onnx_light.onnx as onnxl
from onnx_light.backend.test.case.base import ALL_TESTS, TestCase, collect_test_case, expect


def _register_test_schemas():
    """Registers schemas needed for tests."""
    from onnx_light.onnx import defs

    # Register Abs schema (since_version=13)
    if not defs.has_schema("Abs"):
        abs_schema = defs.OpSchema("Abs", defs.ONNX_DOMAIN, 13, doc="Absolute value")
        defs.register_schema(abs_schema)

    # Register Add schema (since_version=14)
    if not defs.has_schema("Add"):
        add_schema = defs.OpSchema("Add", defs.ONNX_DOMAIN, 14, doc="Add two tensors")
        defs.register_schema(add_schema)

    # Register Clip schema (since_version=13)
    if not defs.has_schema("Clip"):
        clip_schema = defs.OpSchema("Clip", defs.ONNX_DOMAIN, 13, doc="Clip tensor values")
        defs.register_schema(clip_schema)


class TestExpectFunction(unittest.TestCase):
    """Tests for the expect function in backend test case base."""

    @classmethod
    def setUpClass(cls):
        """Registers schemas needed for tests."""
        _register_test_schemas()

    def setUp(self):
        """Clears ALL_TESTS before each test."""
        ALL_TESTS.clear()

    def tearDown(self):
        """Clears ALL_TESTS after each test."""
        ALL_TESTS.clear()

    def test_expect_creates_test_case(self):
        """Tests that expect creates a TestCase and adds it to ALL_TESTS."""
        node = onnxl.helper.make_node("Abs", inputs=["x"], outputs=["y"])
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
        # Create a node with an optional input (represented by empty string)
        node = onnxl.helper.make_node("Clip", inputs=["x", "", ""], outputs=["y"])
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y = np.clip(x, 0, 6)

        expect(node, inputs=[x], outputs=[y], name="test_clip_optional")

        tc = ALL_TESTS["test_clip_optional"]
        self.assertIsNotNone(tc.model)
        # Should only have 1 input in data_sets (the non-empty one)
        inputs_list, _ = tc.data_sets[0]
        self.assertEqual(len(inputs_list), 1)

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


class TestCollectTestCase(unittest.TestCase):
    """Tests for the collect_test_case function."""

    @classmethod
    def setUpClass(cls):
        """Registers schemas needed for tests."""
        _register_test_schemas()

    def setUp(self):
        """Clears ALL_TESTS before each test."""
        ALL_TESTS.clear()

    def tearDown(self):
        """Clears ALL_TESTS after each test."""
        ALL_TESTS.clear()

    def test_collect_test_case_returns_dict(self):
        """Tests that collect_test_case returns a dictionary."""
        result = collect_test_case()
        self.assertIsInstance(result, dict)

    def test_collect_test_case_finds_abs_test(self):
        """Tests that collect_test_case finds the Abs test case."""
        result = collect_test_case()
        # The abs.py module should have been imported and its export method called
        self.assertIn("test_abs", result)
        tc = result["test_abs"]
        self.assertIsInstance(tc, TestCase)
        self.assertEqual(tc.name, "test_abs")
        self.assertEqual(tc.kind, "node")

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
