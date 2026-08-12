"""Initializes shared nanobind types used by the split extension modules."""

# The extension modules are generated during the native build.
# pyrefly: ignore-errors

from importlib.util import find_spec

import ml_dtypes

from . import _onnxpyprotoop

if find_spec(f"{__name__}._onnxpybackend") is not None:
    from . import _onnxpybackend
else:
    _onnxpybackend = None

__all__ = ["_onnxpybackend", "_onnxpyprotoop"]
