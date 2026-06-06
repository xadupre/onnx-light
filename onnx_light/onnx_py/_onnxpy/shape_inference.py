# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Proxy module that exposes the merged ``shape_inference`` submodule.

The merged submodule collects bindings from both
:mod:`onnx_light.onnx_py._onnxpyprotolib` and
:mod:`onnx_light.onnx_py._onnxpyoptim`.  This module exists so that the
dotted import path ``onnx_light.onnx_py._onnxpy.shape_inference`` resolves
correctly without relying on ``sys.modules`` manipulation.

Example::

    from onnx_light.onnx_py._onnxpy.shape_inference import ShapesContext
"""

from __future__ import annotations

from typing import Any

_cached: Any = None


def _merged() -> Any:
    """Return the merged ``shape_inference`` submodule (loaded lazily)."""
    global _cached
    if _cached is None:
        import importlib

        _pkg = "onnx_light.onnx_py"
        _protolib = importlib.import_module(f"{_pkg}._onnxpyprotolib")
        _optim = importlib.import_module(f"{_pkg}._onnxpyoptim")
        si = _protolib.shape_inference
        for _attr in dir(_optim.shape_inference):
            if not _attr.startswith("_") and not hasattr(si, _attr):
                setattr(si, _attr, getattr(_optim.shape_inference, _attr))
        _cached = si
    return _cached


def __getattr__(name: str) -> Any:
    try:
        val = getattr(_merged(), name)
        globals()[name] = val
        return val
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return [n for n in dir(_merged()) if not n.startswith("_")]
