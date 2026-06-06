# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Proxy: re-exports :mod:`onnx_light.onnx_lib.compose` in the ``onnx`` namespace."""
from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    import onnx_light.onnx_lib.compose as _mod

    try:
        val = getattr(_mod, name)
        globals()[name] = val
        return val
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    import onnx_light.onnx_lib.compose as _mod

    return [n for n in dir(_mod) if not n.startswith("_")]
