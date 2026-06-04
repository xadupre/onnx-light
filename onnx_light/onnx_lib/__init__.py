# This module re-exports extension symbols that exist only at runtime, so Pyrefly errors are
# suppressed here.
# pyrefly: ignore-errors
from ..onnx_proto._onnxpy import (  # type: ignore
    AttributeProto,
    DeviceConfigurationProto,
    FileLoadMode,
    FunctionProto,
    GraphProto,
    IntIntListEntryProto,
    MapProto,
    Message,
    ModelProto,
    NodeDeviceConfigurationProto,
    NodeProto,
    OperatorSetIdProto,
    OperatorStatus,
    OptionalProto,
    ParseOptions,
    PrintOptions,
    SequenceProto,
    SerializeOptions,
    ShardedDimProto,
    ShardingSpecProto,
    SimpleShardedDimProto,
    SparseTensorProto,
    StringStringEntryProto,
    TensorAnnotation,
    TensorBufferOptions,
    TensorProto,
    TensorShapeProto,
    TypeProto,
    ValueInfoProto,
    IR_VERSION,
    align_external_data_streaming,
    consolidate_tensors_to_buffer,
    save_model_with_shared_external_data,
    utils_onnx_read_varint64,
)
from . import defs
from . import numpy_helper
from . import shape_inference
from .io_helper import (
    load,
    load_encrypted,
    load_encrypted_string,
    save,
    save_encrypted,
    save_encrypted_string,
)
