"""Compatibility shim for :mod:`onnx.onnx_pb`.

Re-exports all protobuf message types from the top-level ``onnx_light.onnx``
package so that ``from onnx_light.onnx.onnx_pb import ModelProto`` works.
"""

from . import (  # noqa: F401
    AttributeProto,
    DeviceConfigurationProto,
    FunctionProto,
    GraphProto,
    MapProto,
    ModelProto,
    NodeProto,
    OperatorSetIdProto,
    OptionalProto,
    SequenceProto,
    SparseTensorProto,
    TensorAnnotation,
    TensorProto,
    TensorShapeProto,
    TypeProto,
    ValueInfoProto,
)
