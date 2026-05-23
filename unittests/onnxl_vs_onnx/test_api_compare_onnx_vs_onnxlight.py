"""Tests for :mod:`onnx_light.compatibility` that compare against :mod:`onnx`.

These exercise the comparison between :mod:`onnx` and
:mod:`onnx_light.onnx` and also smoke-test the gallery example
``docs/examples/core/plot_api_compare.py``.

The portion of the test suite that does not depend on the upstream
:mod:`onnx` package lives in
``unittests/main/test_plot_api_compare.py``.
"""

from __future__ import annotations

import importlib.util
import pathlib
import unittest

import onnx
import onnx.inliner  # noqa: F401  -- bind the inliner attribute on ``onnx``
import onnx_light.onnx as onnxl

from onnx_light.compatibility import (
    DEFAULT_SUBMODULES,
    compare_packages,
    compare_submodule,
    compare_top_level_functions,
    list_submodules,
)
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


class TestCompatibilityApiCompareVsOnnx(ExtTestCase):
    """Comparison tests that require the upstream ``onnx`` package."""

    def test_list_submodules_onnx(self):
        submods = list_submodules(onnx)
        for name in ("helper", "numpy_helper", "checker", "defs", "parser"):
            self.assertIn(name, submods)
        # Re-exported standard library modules must be filtered out.
        for name in ("os", "sys", "typing"):
            self.assertNotIn(name, submods)

    def test_compare_submodule_numpy_helper_is_aligned(self):
        report = compare_submodule("numpy_helper", onnx, onnxl)
        self.assertEqual(report["missing_in_onnxl"], [])
        self.assertEqual(report["extra_in_onnxl"], [])
        self.assertEqual(report["signature_diffs"], [])
        for name in ("to_array", "from_array"):
            self.assertIn(name, report["common"])

    def test_compare_submodule_helper_has_common_makers(self):
        report = compare_submodule("helper", onnx, onnxl)
        for name in ("make_node", "make_graph", "make_tensor", "make_attribute"):
            self.assertIn(name, report["common"])

    def test_compare_submodule_parser_signature_diffs(self):
        report = compare_submodule("parser", onnx, onnxl)
        diff_names = {diff.name for diff in report["signature_diffs"]}
        self.assertIn("parse_model", diff_names)
        for diff in report["signature_diffs"]:
            if diff.name == "parse_model":
                self.assertEqual(diff.onnxl_params, ("text",))

    def test_compare_submodule_inliner_common_functions(self):
        report = compare_submodule("inliner", onnx, onnxl)
        for name in ("inline_local_functions", "inline_selected_functions"):
            self.assertIn(name, report["common"])

    def test_compare_top_level_functions_load_save(self):
        report = compare_top_level_functions(onnx, onnxl)
        # ``load`` and ``save`` are exposed directly on both packages.
        self.assertIn("load", report["common"])
        self.assertIn("save", report["common"])
        # Both have signature mismatches: onnx_light.onnx.load/save expose
        # extra parameters (num_threads, no_copy, ...).
        diff_names = {diff.name for diff in report["signature_diffs"]}
        self.assertIn("load", diff_names)
        self.assertIn("save", diff_names)

    def test_compare_packages_overall(self):
        report = compare_packages(onnx, onnxl)
        self.assertIn("submodules", report)
        self.assertIn("top_level", report)
        self.assertIn("per_submodule", report)
        # Sub-module sets agree on the core sub-modules.
        for name in ("helper", "numpy_helper", "checker"):
            self.assertIn(name, report["submodules"]["common"])
        # Per-submodule entries are present for every requested sub-module.
        for name in DEFAULT_SUBMODULES:
            self.assertIn(name, report["per_submodule"])


class TestPlotApiCompareExample(ExtTestCase):
    """Smoke-test that the gallery example imports cleanly and uses the helpers."""

    def test_example_module_imports(self):
        example = _load_example_module()
        # The example is expected to compute and expose a ``report`` mapping.
        self.assertTrue(hasattr(example, "report"))
        report = example.report
        self.assertIn("submodules", report)
        self.assertIn("top_level", report)
        self.assertIn("per_submodule", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
