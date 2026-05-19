"""Schema helpers."""

from __future__ import annotations

from ...onnx_proto import _onnxpy as _C  # type: ignore[missing-module-attribute]
from .schema_diff import (  # noqa: F401
    AttributeDiff,
    ConstraintDiff,
    ParameterDiff,
    SchemaDiff,
    compare_schemas,
)

ONNX_DOMAIN = ""
ONNX_ML_DOMAIN = "ai.onnx.ml"
AI_ONNX_PREVIEW_TRAINING_DOMAIN = "ai.onnx.preview.training"

C = _C.defs  # type: ignore

has = C.has_schema
has_schema = C.has_schema
get_schema = C.get_schema
get_all_schemas = C.get_all_schemas
get_all_schemas_with_history = C.get_all_schemas_with_history
deregister_schema = C.deregister_schema
schema_version_map = C.schema_version_map

OpSchema = C.OpSchema
SchemaError = C.SchemaError


def onnx_opset_version() -> int:
    """Returns the current opset for domain ``ai.onnx``."""
    return C.schema_version_map()[ONNX_DOMAIN][1]


def onnx_ir_version() -> int:
    """Returns the ONNX IR version exported by the C++ bindings.

    Returns:
        The ONNX IR version.
    """
    return _C.IR_VERSION  # type: ignore


def onnx_ml_opset_version() -> int:
    """Returns the current opset for domain ``ai.onnx.ml``."""
    return C.schema_version_map()[ONNX_ML_DOMAIN][1]


def register_onnx_operator_set_schema() -> None:
    """Registers all built-in ONNX operator schemas into the schema registry.

    Registers every operator schema for every opset version that is compiled
    into onnx_light (i.e., the full ONNX standard operator set with history).
    Duplicate registrations are silently ignored, so the function is safe to
    call multiple times.

    Returns:
        None.
    """
    C.register_onnx_operator_set_schema()


def register_schema(schema: OpSchema) -> None:  # type: ignore
    """Registers a user-provided ``OpSchema``."""
    domain = schema.domain  # type: ignore
    version = schema.since_version  # type: ignore
    version_map = C.schema_version_map()
    min_version, max_version = version_map.get(domain, (version, version))
    if domain not in version_map or not (min_version <= version <= max_version):
        C.set_domain_to_version(domain, min(min_version, version), max(max_version, version))
    C.register_schema(schema)
