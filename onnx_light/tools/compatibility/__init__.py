"""Compatibility helpers for ONNX operator schema comparisons."""

from __future__ import annotations

from .schema_diff import (  # noqa: F401
    AttributeDiff,
    ConstraintDiff,
    DeprecationDiff,
    DocDiff,
    ParameterDiff,
    SchemaDiff,
    compare_schemas,
)
