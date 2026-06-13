"""
This replicates the current Python API of the onnx package.
"""

from __future__ import annotations

from ..onnx_py._onnxpyprotoop import (  # type: ignore # noqa: F401
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
    SerializeFormat,
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
    align_external_data_streaming,
    collect_external_inputs,
    consolidate_tensors_to_buffer,
    save_model_with_shared_external_data,
    utils_onnx_read_varint64,
)
from ..onnx_proto._io_helper import (  # noqa: F401
    load,
    load_encrypted,
    load_encrypted_string,
    save,
    save_encrypted,
    save_encrypted_string,
)
