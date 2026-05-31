# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Backwards-compatibility shim that merges the public attributes of the
:mod:`onnx_light.onnx_py._onnxpyproto` and :mod:`onnx_light.onnx_py._onnxbackend`
compiled extensions into a single namespace.

The original ``_onnxpy`` extension was split into two nanobind modules:

* :mod:`onnx_light.onnx_py._onnxpyproto` exposes the proto bindings
  (``AddOnnxPyProto``).
* :mod:`onnx_light.onnx_py._onnxbackend` exposes the operator-schema,
  shape-inference, optimisation and backend-test bindings.

This module re-exports every public attribute of both extensions so that
existing callers writing ``onnx_light.onnx_py._onnxpy.<name>`` keep working.
"""

from . import _onnxbackend, _onnxpyproto  # type: ignore[attr-defined]

__all__: list[str] = []
for _mod in (_onnxpyproto, _onnxbackend):
    for _name in dir(_mod):
        if _name.startswith("_"):
            continue
        globals()[_name] = getattr(_mod, _name)
        if _name not in __all__:
            __all__.append(_name)

del _mod, _name
