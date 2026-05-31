# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Backward-compatibility shim that merges the public attributes of the
:mod:`onnx_light.onnx_py._onnxpyproto` and :mod:`onnx_light.onnx_py._onnxbackend`
compiled extensions into a single namespace.

The original ``_onnxpy`` extension was split into two nanobind modules:

* :mod:`onnx_light.onnx_py._onnxpyproto` exposes the proto bindings, the
  operator-schema (``onnx_op``) bindings, the optim (``expressions`` /
  ``shape_inference``) bindings and the onnx_lib bindings (``defs``,
  ``parser``, ``checker``, ``inliner``, ``shape_inference``,
  ``version_converter``).
* :mod:`onnx_light.onnx_py._onnxbackend` exposes the ``backend`` deterministic
  random helpers and the ``backend_test`` test-case utilities.

This module re-exports every public attribute of both extensions so that
existing callers writing ``onnx_light.onnx_py._onnxpy.<name>`` keep working.
"""

from . import _onnxpyproto, _onnxbackend  # type: ignore[attr-defined]

__all__: list[str] = []
for _mod in (_onnxpyproto, _onnxbackend):
    for _name in dir(_mod):
        if _name.startswith("_"):
            continue
        globals()[_name] = getattr(_mod, _name)
        if _name not in __all__:
            __all__.append(_name)  # noqa: PYI056

del _mod, _name
