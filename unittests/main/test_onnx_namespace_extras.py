"""Tests that :mod:`onnx_light.onnx` re-exposes the onnx-light specific
sub-packages (``backend``, ``backend_test``, ``fuzz``, ``tools``).

These sub-packages live at ``onnx_light.<name>`` but are also reachable
from the ``onnx_light.onnx`` namespace (the API entry point that
mirrors the upstream :mod:`onnx` package) so that downstream callers can
write ``onnx_light.onnx.<name>`` consistently.
"""

from __future__ import annotations

import importlib
import sys
import unittest

import onnx_light.onnx as onnxl
from onnx_light.ext_test_case import ExtTestCase


class TestOnnxNamespaceExtras(ExtTestCase):
    """Verifies the re-exports added in ``onnx_light/onnx/__init__.py``."""

    def test_attributes_are_present(self):
        for name in ("backend", "backend_test", "fuzz", "tools"):
            self.assertTrue(
                hasattr(onnxl, name), f"onnx_light.onnx is missing attribute {name!r}"
            )
            mod = getattr(onnxl, name)
            self.assertEqual(mod.__name__, f"onnx_light.{name}")

    def test_dotted_imports_resolve(self):
        for name in ("backend", "backend_test", "fuzz", "tools"):
            full = f"onnx_light.onnx.{name}"
            mod = importlib.import_module(full)
            self.assertIs(mod, sys.modules[f"onnx_light.{name}"])

    def test_backend_test_exposes_data_model(self):
        import onnx_light.backend_test as bt

        for name in (
            "DataSet",
            "Tensor",
            "TestCase",
            "collect_test_cases",
            "collect_test_cases_by_name",
        ):
            self.assertTrue(hasattr(bt, name))


if __name__ == "__main__":
    unittest.main(verbosity=2)
