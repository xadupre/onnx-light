from ..onnx_lib.defs import (  # noqa: F401
    deregister_schema,
    get_all_schemas,
    get_all_schemas_with_history,
    get_function_ops,
    get_schema,
    has,
    has_schema,
    ONNX_DOMAIN,
    ONNX_ML_DOMAIN,
    onnx_ir_version,
    onnx_ml_opset_version,
    onnx_opset_version,
    OpSchema,
    register_onnx_operator_set_schema,
    register_schema,
    schema_version_map,
    SchemaError,
)

IR_VERSION = onnx_ir_version()
