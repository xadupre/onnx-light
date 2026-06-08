# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Lightweight :class:`ReferenceEvaluator` built on top of the C++ kernels
Python API exposed by :mod:`onnx_light.onnx_py._onnxpykernels`.

This module mirrors the public surface of
:class:`onnx.reference.ReferenceEvaluator` closely enough to be a drop-in
replacement for the common ``sess.run(None, {"x": ...})`` usage, while
delegating every operator evaluation to the static
``KernelDispatchTable`` of ``lib_onnx_kernels``.
"""

from __future__ import annotations

from ._evaluator import ReferenceEvaluator

__all__ = ["ReferenceEvaluator"]
