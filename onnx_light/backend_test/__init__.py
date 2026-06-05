"""Python re-export of the C++ ``backend_test`` bindings.

This sub-package exposes the runtime data model used by the C++ backend
test infrastructure (``onnx_backend_test``) through a stable Python
import path (``onnx_light.backend_test``).

The underlying objects are implemented in C++
(``onnx_light/onnx_backend_test/test_case.{h,cc}``,
``onnx_light/onnx_backend_test/simple_tensor.{h,cc}``) and bound to
Python via :mod:`onnx_light.onnx_py._onnxbackend`.

It mirrors :mod:`onnx_light.backend` (which exposes the deterministic
pseudo-random helpers) and is what ``onnx_light.backend.test.case``
builds higher-level helpers on top of.
"""

from __future__ import annotations

from ..onnx_py._onnxpy import backend_test as _C  # type: ignore[attr-defined]

# Core data model.
DataSet = _C.DataSet
Tensor = _C.Tensor
TestCase = _C.TestCase

# Helper that collects every C++-registered backend test case.
collect_test_cases = _C.collect_test_cases

__all__ = ["DataSet", "Tensor", "TestCase", "collect_test_cases"]
