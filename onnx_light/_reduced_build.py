# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers for the *reduced* onnx-light build.

The reduced wheel (built with ``ONNX_LIGHT_BUILD_KERNELS=OFF``) ships the
proto / schema / shape-inference / manipulation layers but omits the
operator-kernel runtime (``_onnxpykernels``) and the backend-test registries
(``_onnxpybackend``). Any feature that relies on those native extensions must
fail with a clear, explicit error instead of a bare
``ModuleNotFoundError`` mentioning an internal extension name.
"""

from __future__ import annotations

from typing import Optional


class ReducedBuildError(ImportError):
    """Raised when a kernels/backend feature is used in a reduced build."""


_MESSAGE = (
    "{feature} requires the onnx-light operator-kernel runtime and backend-test "
    "extensions (_onnxpykernels / _onnxpybackend), which were not compiled in "
    "this installation. This is a reduced 'onnx-light-reduced' build "
    "(ONNX_LIGHT_BUILD_KERNELS=OFF). Install the full 'onnx-light' wheel to use "
    "the reference runtime, the deterministic random helpers and the backend "
    "test cases."
)


def kernels_required(feature: str, exc: Optional[BaseException] = None) -> None:
    """Raises a :class:`ReducedBuildError` describing the missing feature.

    Args:
        feature: A short human-readable description of the requested feature.
        exc: The original :class:`ImportError`, chained for debugging.

    Raises:
        ReducedBuildError: Always.
    """
    raise ReducedBuildError(_MESSAGE.format(feature=feature)) from exc
