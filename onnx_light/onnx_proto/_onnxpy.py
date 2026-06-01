"""Lazy re-export of :mod:`onnx_light.onnx_py._onnxpy`.

Attributes are forwarded on first access so that the underlying compiled
extensions are only imported when actually needed.
"""

from typing import Any

from ..onnx_py import _onnxpy  # type: ignore


def __getattr__(name: str) -> Any:
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_onnxpy, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    names = {n for n in globals() if not n.startswith("_")}
    names.update(n for n in dir(_onnxpy) if not n.startswith("_"))
    return sorted(names)
