# This module re-exports extension symbols that exist only at runtime, so Pyrefly errors are
# suppressed here.
# pyrefly: ignore-errors
from ..onnx_py._onnxpyprotoop import (  # type: ignore
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
from ..onnx_proto import _helper as helper, _numpy_helper as numpy_helper
from ..onnx_proto._io_helper import (
    load,
    load_encrypted,
    load_encrypted_string,
    save,
    save_encrypted,
    save_encrypted_string,
)
from . import defs, external_data_helper, parser, shape_inference
