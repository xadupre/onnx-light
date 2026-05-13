"""Schema helpers."""

from ..onnx_proto import _onnxpy as _C  # type: ignore[missing-module-attribute]

ONNX_DOMAIN = ""
ONNX_ML_DOMAIN = "ai.onnx.ml"
AI_ONNX_PREVIEW_TRAINING_DOMAIN = "ai.onnx.preview.training"

C = _C.defs

has = C.has_schema
has_schema = C.has_schema
get_schema = C.get_schema
get_all_schemas = C.get_all_schemas
get_all_schemas_with_history = C.get_all_schemas_with_history
deregister_schema = C.deregister_schema
schema_version_map = C.schema_version_map

OpSchema = C.OpSchema
SchemaError = C.SchemaError
_ONNX_IR_BY_MIN_OPSET = (
    (25, 13),
    (24, 12),
    (23, 11),
    (21, 10),
    (19, 9),
    (15, 8),
    (12, 7),
    (11, 6),
    (10, 5),
    (9, 4),
)
_DEFAULT_IR_VERSION_FOR_LEGACY_OPSETS = 3


def onnx_opset_version() -> int:
    """Returns the current opset for domain ``ai.onnx``."""
    return C.schema_version_map()[ONNX_DOMAIN][1]


def onnx_ir_version() -> int:
    """Derives the ONNX IR version from the opset-threshold mapping table.

    The function searches `_ONNX_IR_BY_MIN_OPSET` for the first minimum opset
    that is lower than or equal to the current opset. For opsets below 9, it
    falls back to the legacy IR version value.

    Returns:
        The ONNX IR version corresponding to the current opset.
    """
    opset = onnx_opset_version()
    for min_opset, ir_version in _ONNX_IR_BY_MIN_OPSET:
        if opset >= min_opset:
            return ir_version
    return _DEFAULT_IR_VERSION_FOR_LEGACY_OPSETS


def onnx_ml_opset_version() -> int:
    """Returns the current opset for domain ``ai.onnx.ml``."""
    return C.schema_version_map()[ONNX_ML_DOMAIN][1]


def register_schema(schema: OpSchema) -> None:  # type: ignore
    """Registers a user-provided ``OpSchema``."""
    domain = schema.domain  # type: ignore
    version = schema.since_version  # type: ignore
    version_map = C.schema_version_map()
    min_version, max_version = version_map.get(domain, (version, version))
    if domain not in version_map or not (min_version <= version <= max_version):
        C.set_domain_to_version(domain, min(min_version, version), max(max_version, version))
    C.register_schema(schema)
