"""Tests for :mod:`onnx_light.compatibility` that do not depend on :mod:`onnx`.

These exercise the comparison helpers (sub-module listing, function
listing, signature diffing) using only :mod:`onnx_light.onnx` (and a
couple of synthetic in-memory modules).  Tests that need the upstream
:mod:`onnx` package live in
``unittests/onnxl_vs_onnx/test_plot_api_compare.py``.
"""

from __future__ import annotations

import types
import unittest

import onnx_light.onnx as onnxl

from onnx_light.compatibility import (
    DEFAULT_SUBMODULES,
    SignatureDiff,
    compare_signatures,
    compare_submodule,
    list_public_functions,
    list_submodules,
)
from onnx_light.ext_test_case import ExtTestCase


class TestCompatibilityApiCompare(ExtTestCase):
    """Direct tests for :mod:`onnx_light.compatibility` (no ``onnx`` dependency)."""

    def test_default_submodules_contains_core(self):
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
        ):
            self.assertIn(name, DEFAULT_SUBMODULES)

    def test_list_submodules_onnxl(self):
        submods = list_submodules(onnxl)
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
        self.assertNotIn("sys", submods)

    def test_list_public_functions_filters_reexports(self):
        helper_funcs = list_public_functions(onnxl.helper)
        # NamedTuple is re-exported from ``typing`` and must be filtered out.
        self.assertNotIn("NamedTuple", helper_funcs)
        for name in ("make_node", "make_graph", "make_model", "make_tensor"):
            self.assertIn(name, helper_funcs)

    def test_compare_submodule_unknown_module(self):
        report = compare_submodule("does_not_exist", onnxl, onnxl)
        self.assertEqual(report["common"], [])
        self.assertIn("<entire module>", report["missing_in_onnxl"])
        self.assertIn("<entire module>", report["extra_in_onnxl"])

    def test_compare_signatures_detects_param_rename(self):
        mod_a = types.ModuleType("mod_a")
        mod_b = types.ModuleType("mod_b")

        def f(model_text):  # pragma: no cover - introspected only
            return model_text

        def g(text):  # pragma: no cover - introspected only
            return text

        mod_a.f = f
        mod_b.f = g
        diffs = compare_signatures(mod_a, mod_b, ["f"])
        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs[0].name, "f")
        self.assertEqual(diffs[0].onnx_params, ("model_text",))
        self.assertEqual(diffs[0].onnxl_params, ("text",))

    def test_signature_diff_namedtuple_fields(self):
        diff = SignatureDiff("foo", ("a",), ("b",))
        self.assertEqual(diff.name, "foo")
        self.assertEqual(diff.onnx_params, ("a",))
        self.assertEqual(diff.onnxl_params, ("b",))


if __name__ == "__main__":
    unittest.main(verbosity=2)
