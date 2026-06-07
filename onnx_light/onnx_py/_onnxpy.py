# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Backward-compatibility shim that merges the public attributes of the
:mod:`onnx_light.onnx_py._onnxpyprotoop`,
:mod:`onnx_light.onnx_py._onnxpyprotolib`,
:mod:`onnx_light.onnx_py._onnxpyoptim`,
:mod:`onnx_light.onnx_py._onnxkernels` and
:mod:`onnx_light.onnx_py._onnxbackend` compiled extensions into a single
namespace.

The original ``_onnxpy`` extension was split into five nanobind modules:

* :mod:`onnx_light.onnx_py._onnxpyprotoop` exposes the proto bindings and
  operator-schema (``onnx_op``) bindings.
* :mod:`onnx_light.onnx_py._onnxpyprotolib` exposes the onnx_lib bindings
  (``defs``, ``parser``, ``checker``, ``inliner``, ``shape_inference``,
  ``version_converter``).
* :mod:`onnx_light.onnx_py._onnxpyoptim` exposes the optim bindings
  (``expressions`` and ``shape_inference``).
* :mod:`onnx_light.onnx_py._onnxkernels` exposes the ``backend``
  deterministic pseudo-random helpers backing :mod:`onnx_light.backend`.
* :mod:`onnx_light.onnx_py._onnxbackend` exposes the ``backend_test``
  test-case utilities.

This module re-exports every public attribute of all extensions so that
existing callers writing ``onnx_light.onnx_py._onnxpy.<name>`` keep working.

The compiled extensions are **loaded lazily**: the first access to an
attribute through this module triggers the import of the extension(s) that
provide it.  Callers that only need, for example, the proto bindings never
pay for importing the optim or backend extensions.
"""

from __future__ import annotations

import importlib
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Any

# Marking this module as a (namespace-like) package allows submodule import
# statements such as ``from onnx_light.onnx_py._onnxpy.shape_inference import
# ShapesContext`` to be handled by the meta-path finder installed below.
__path__: list[str] = []

# Ordered list of compiled extension modules to consult when an attribute is
# looked up.  The order matters: when several extensions expose a value with
# the same name (and the value is not a submodule), the first match wins,
# which mirrors the historical eager-merge behavior.
_EXTENSIONS: tuple[str, ...] = (
    "_onnxpyprotoop",
    "_onnxpyprotolib",
    "_onnxpyoptim",
    "_onnxkernels",
    "_onnxbackend",
)

# Attribute names that are exposed as submodules by more than one extension.
# Looking up such a name forces every listed extension to be imported so that
# their public attributes can be merged into a single namespace, matching the
# previous eager-merge behavior.
_COLLISIONS: dict[str, tuple[str, ...]] = {"shape_inference": ("_onnxpyprotolib", "_onnxpyoptim")}

_loaded: dict[str, ModuleType] = {}


def _load(ext_name: str) -> ModuleType:
    """Import the given compiled extension on first use and cache it."""
    mod = _loaded.get(ext_name)
    if mod is None:
        mod = importlib.import_module(f"{__package__}.{ext_name}")
        _loaded[ext_name] = mod
    return mod


def _merge_submodule(existing: ModuleType, extra: ModuleType) -> ModuleType:
    """Copy public attributes from ``extra`` into ``existing`` in place."""
    for _attr in dir(extra):
        if _attr.startswith("_"):
            continue
        if not hasattr(existing, _attr):
            setattr(existing, _attr, getattr(extra, _attr))
    return existing


def __getattr__(name: str) -> Any:
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    # Names known to be exposed as submodules by several extensions need every
    # contributing extension to be imported so their attributes can be merged.
    collision = _COLLISIONS.get(name)
    if collision is not None:
        result: ModuleType | None = None
        for ext_name in collision:
            mod = _load(ext_name)
            if not hasattr(mod, name):
                continue
            value = getattr(mod, name)
            if result is None:
                result = value
            elif isinstance(result, ModuleType) and isinstance(value, ModuleType):
                _merge_submodule(result, value)
        if result is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        globals()[name] = result
        return result

    # Otherwise consult the extensions in order and return the first match.
    # Only extensions that need to be inspected to find ``name`` are imported.
    for ext_name in _EXTENSIONS:
        mod = _load(ext_name)
        if hasattr(mod, name):
            value = getattr(mod, name)
            globals()[name] = value
            return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    names: set[str] = {n for n in globals() if not n.startswith("_")}
    for ext_name in _EXTENSIONS:
        mod = _load(ext_name)
        for n in dir(mod):
            if not n.startswith("_"):
                names.add(n)
    return sorted(names)


class _PreloadedLoader(Loader):
    """Loader that returns an already-resolved module object."""

    def __init__(self, module: ModuleType) -> None:
        self._module = module

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        return self._module

    def exec_module(self, module: ModuleType) -> None:  # pragma: no cover - no-op
        return None


class _SubmoduleFinder(MetaPathFinder):
    """Resolve submodules of :mod:`onnx_light.onnx_py._onnxpy` via ``__getattr__``.

    This allows statements such as ``from onnx_light.onnx_py._onnxpy.shape_inference
    import ShapesContext`` to work even though ``_onnxpy`` is implemented as a
    single ``.py`` file rather than a real package directory.
    """

    _prefix = __name__ + "."

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> ModuleSpec | None:
        if not fullname.startswith(self._prefix):
            return None
        sub = fullname[len(self._prefix) :]
        if "." in sub or sub.startswith("_"):
            return None
        try:
            value = __getattr__(sub)
        except AttributeError:
            return None
        if not isinstance(value, ModuleType):
            return None
        sys.modules[fullname] = value
        return ModuleSpec(fullname, _PreloadedLoader(value))


# Install the finder exactly once; re-imports of this module (e.g. during
# ``importlib.reload``) must not stack multiple finders.
if not any(isinstance(_f, _SubmoduleFinder) for _f in sys.meta_path):
    sys.meta_path.append(_SubmoduleFinder())
