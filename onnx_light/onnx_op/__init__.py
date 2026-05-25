"""Python re-export of the ``onnx_op`` C++ extension (``LightOpSchema``).

This module exposes the lightweight operator-schema types defined in the C++
``onnx_op`` library through a stable Python import path
(``onnx_light.onnx_op``).  The underlying objects are implemented in C++
(``onnx_light/onnx_op/light_op_schema.{h,cc}`` and the operator-set
registries under ``onnx_light/onnx_op/operator_sets_*.{h,cc}``) and bound to
Python via :mod:`onnx_light.onnx_py._onnxpy_op`.

The :class:`LightOpSchema` class is the ``onnx_light`` analogue of
:class:`onnx_light.onnx.defs.OpSchema`: a read-only description of a single
versioned ONNX operator that documentation and validation tools can consume
without depending on the full upstream ``onnx`` package.
"""

from __future__ import annotations

from ..onnx_py._onnxpy import onnx_op as _C  # type: ignore[attr-defined]

# Constants
kOnnxDomain = _C.kOnnxDomain

# Core schema types
LightOpSchema = _C.LightOpSchema
FormalParameter = _C.FormalParameter
TypeConstraintParam = _C.TypeConstraintParam
TensorType = _C.TensorType

# Helpers
ToTypeString = _C.ToTypeString
GetAllOnnxOpSchemasWithHistory = _C.GetAllOnnxOpSchemasWithHistory


def get_all_schemas_with_history() -> list[LightOpSchema]:
    """Returns every registered :class:`LightOpSchema`, including past versions.

    The returned list contains one :class:`LightOpSchema` per
    ``(domain, name, since_version)`` triple known to the lightweight
    operator registry shipped with :mod:`onnx_light`.  The ordering is the
    one used by the C++ extension and is stable across calls.

    Returns:
        list[LightOpSchema]: All registered lightweight schemas, including
        historical opset versions.
    """
    return list(_C.GetAllOnnxOpSchemasWithHistory())


def get_all_schemas() -> list[LightOpSchema]:
    """Returns the latest :class:`LightOpSchema` for every known operator.

    For each ``(domain, name)`` pair, only the schema with the highest
    ``since_version`` is kept.

    Returns:
        list[LightOpSchema]: The latest lightweight schema per operator.
    """
    latest: dict[tuple[str, str], LightOpSchema] = {}
    for s in _C.GetAllOnnxOpSchemasWithHistory():
        key = (s.domain, s.name)
        if key not in latest or s.since_version > latest[key].since_version:
            latest[key] = s
    return list(latest.values())


__all__ = [
    "FormalParameter",
    "GetAllOnnxOpSchemasWithHistory",
    "LightOpSchema",
    "TensorType",
    "ToTypeString",
    "TypeConstraintParam",
    "get_all_schemas",
    "get_all_schemas_with_history",
    "kOnnxDomain",
]
