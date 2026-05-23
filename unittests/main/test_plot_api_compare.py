"""Tests for the gallery example ``docs/examples/core/plot_api_compare.py``.

This exercises the comparison helpers (sub-module listing, function
listing, signature diffing) and asserts that a few core entry-points
are aligned between :mod:`onnx` and :mod:`onnx_light.onnx`.
"""

from __future__ import annotations

import importlib.util
import pathlib
import unittest

import onnx
import onnx_light.onnx as onnxl

from onnx_light.ext_test_case import ExtTestCase


def _load_example_module():
    """Imports ``docs/examples/core/plot_api_compare.py`` as a module."""
    root = pathlib.Path(__file__).resolve().parents[2]
    source_path = root / "docs" / "examples" / "core" / "plot_api_compare.py"
    spec = importlib.util.spec_from_file_location("plot_api_compare", source_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestPlotApiCompare(ExtTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.example = _load_example_module()

    def test_list_submodules_onnx(self):
        submods = self.example.list_submodules(onnx)
        # Sanity: the core onnx sub-modules should be discovered.
        for name in ("helper", "numpy_helper", "checker", "defs", "parser"):
            self.assertIn(name, submods)
        # Re-exported standard library modules must be filtered out.
        for name in ("os", "sys", "typing"):
            self.assertNotIn(name, submods)

    def test_list_submodules_onnxl(self):
        submods = self.example.list_submodules(onnxl)
        for name in (
            "helper",
            "numpy_helper",
            "checker",
            "defs",
            "parser",
            "shape_inference",
            "version_converter",
            "compose",
            "utils",
            "inliner",
            "io_helper",
        ):
            self.assertIn(name, submods)
        for name in ("sys",):
            self.assertNotIn(name, submods)

    def test_list_public_functions_filters_reexports(self):
        helper_funcs = self.example.list_public_functions(onnxl.helper)
        # NamedTuple is re-exported from ``typing`` and must be filtered out.
        self.assertNotIn("NamedTuple", helper_funcs)
        # A few well-known helper functions should be present.
        for name in ("make_node", "make_graph", "make_model", "make_tensor"):
            self.assertIn(name, helper_funcs)

    def test_compare_submodule_numpy_helper_is_aligned(self):
        report = self.example.compare_submodule("numpy_helper")
        self.assertEqual(report["missing_in_onnxl"], [])
        self.assertEqual(report["extra_in_onnxl"], [])
        self.assertEqual(report["signature_diffs"], [])
        # ``to_array`` / ``from_array`` are the canonical helpers.
        for name in ("to_array", "from_array"):
            self.assertIn(name, report["common"])

    def test_compare_submodule_helper_has_common_makers(self):
        report = self.example.compare_submodule("helper")
        for name in ("make_node", "make_graph", "make_tensor", "make_attribute"):
            self.assertIn(name, report["common"])

    def test_compare_submodule_parser_signature_diffs(self):
        # ``onnx`` uses ``model_text``/``graph_text``/... while
        # ``onnx_light`` uses ``text``.  Document this as expected.
        report = self.example.compare_submodule("parser")
        diff_names = {diff.name for diff in report["signature_diffs"]}
        self.assertIn("parse_model", diff_names)
        for diff in report["signature_diffs"]:
            if diff.name == "parse_model":
                self.assertEqual(diff.onnxl_params, ("text",))

    def test_compare_submodule_inliner_common_functions(self):
        report = self.example.compare_submodule("inliner")
        # Both packages expose the inlining helpers.
        for name in ("inline_local_functions", "inline_selected_functions"):
            self.assertIn(name, report["common"])

    def test_compare_submodule_unknown_module(self):
        report = self.example.compare_submodule("does_not_exist")
        self.assertEqual(report["common"], [])
        self.assertIn("<entire module>", report["missing_in_onnxl"])
        self.assertIn("<entire module>", report["extra_in_onnxl"])

    def test_signature_diff_namedtuple_fields(self):
        diff = self.example.SignatureDiff("foo", ("a",), ("b",))
        self.assertEqual(diff.name, "foo")
        self.assertEqual(diff.onnx_params, ("a",))
        self.assertEqual(diff.onnxl_params, ("b",))


if __name__ == "__main__":
    unittest.main(verbosity=2)
