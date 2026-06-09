# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Python re-export of the C++ ``onnx_kernels`` runtime.

This module exposes, through the stable Python import path
:mod:`onnx_light.kernels`, the kernel-runtime primitives implemented in
C++ under ``onnx_light/onnx_kernels`` and bound to Python via the
:mod:`onnx_light.onnx_py._onnxpykernels` nanobind extension.

It provides:

* the :data:`runtime` submodule with the graph/node execution helpers
  (``RunNode``, ``RunNodes``, ``RunGraph``, ``RunFunction``,
  ``RunModel``, ``RunSubgraph``) and the supporting ``RuntimeContext``,
  ``KernelContext`` and ``OpsetId`` types,
* the :data:`backend` submodule with the deterministic pseudo-random
  helpers backing :mod:`onnx_light.backend.random`.

See :ref:`l-design-kernels` for an architectural overview of the C++
kernel layer.
"""

from __future__ import annotations

from ..onnx_py._onnxpykernels import backend, runtime  # type: ignore[missing-import]

__all__ = ["backend", "runtime"]
