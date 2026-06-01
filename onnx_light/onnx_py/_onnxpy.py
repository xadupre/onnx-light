# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Backward-compatibility shim that merges the public attributes of the
:mod:`onnx_light.onnx_py._onnxpyproto`,
:mod:`onnx_light.onnx_py._onnxpyoptim` and
:mod:`onnx_light.onnx_py._onnxbackend` compiled extensions into a single
namespace.

The original ``_onnxpy`` extension was split into three nanobind modules:

* :mod:`onnx_light.onnx_py._onnxpyproto` exposes the proto bindings, the
  operator-schema (``onnx_op``) bindings and the onnx_lib bindings
  (``defs``, ``parser``, ``checker``, ``inliner``, ``shape_inference``,
  ``version_converter``).
* :mod:`onnx_light.onnx_py._onnxpyoptim` exposes the optim bindings
  (``expressions`` and ``shape_inference``).
* :mod:`onnx_light.onnx_py._onnxbackend` exposes the ``backend`` deterministic
  random helpers and the ``backend_test`` test-case utilities.

This module re-exports every public attribute of all extensions so that
existing callers writing ``onnx_light.onnx_py._onnxpy.<name>`` keep working.
"""

from types import ModuleType

from . import _onnxpyproto, _onnxpyoptim, _onnxbackend  # type: ignore[attr-defined]


def _merge_submodule(existing: ModuleType, extra: ModuleType) -> ModuleType:
    """Copy public attributes from ``extra`` into ``existing`` in place."""
    for _attr in dir(extra):
        if _attr.startswith("_"):
            continue
        if not hasattr(existing, _attr):
            setattr(existing, _attr, getattr(extra, _attr))
    return existing


__all__: list[str] = []
_value = None
_existing = None
for _mod in (_onnxpyproto, _onnxpyoptim, _onnxbackend):
    for _name in dir(_mod):
        if _name.startswith("_"):
            continue
        _value = getattr(_mod, _name)
        _existing = globals().get(_name)
        if (
            _existing is not None
            and isinstance(_existing, ModuleType)
            and isinstance(_value, ModuleType)
        ):
            # Both extensions expose a submodule with the same name
            # (for example ``shape_inference`` is defined by both
            # ``_onnxpyproto`` and ``_onnxpyoptim``). Merge the public
            # attributes of the new submodule into the existing one so
            # callers can keep accessing every helper through a single
            # namespace.
            _merge_submodule(_existing, _value)
            continue
        globals()[_name] = _value
        if _name not in __all__:
            __all__.append(_name)  # noqa: PYI056

del _mod, _name, _value, _existing
