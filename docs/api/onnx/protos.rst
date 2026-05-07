protos
======

This page gives a high-level map of the ONNX protobuf message types exposed by
``onnx_light.onnx``.

Core graph structures
=====================

These classes represent the main model and graph topology.

.. autosummary::

    onnx_light.onnx.Message
    onnx_light.onnx.ModelProto
    onnx_light.onnx.GraphProto
    onnx_light.onnx.NodeProto
    onnx_light.onnx.FunctionProto
    onnx_light.onnx.OperatorSetIdProto
    onnx_light.onnx.AttributeProto
    onnx_light.onnx.ValueInfoProto

Tensor and type structures
==========================

These classes describe values, tensors, and type annotations.

.. autosummary::

    onnx_light.onnx.TensorProto
    onnx_light.onnx.SparseTensorProto
    onnx_light.onnx.TensorShapeProto
    onnx_light.onnx.TypeProto
    onnx_light.onnx.SequenceProto
    onnx_light.onnx.OptionalProto
    onnx_light.onnx.MapProto
    onnx_light.onnx.TensorAnnotation

Supporting structures
=====================

These classes cover metadata, sharding, and auxiliary entries.

.. autosummary::

    onnx_light.onnx.StringStringEntryProto
    onnx_light.onnx.IntIntListEntryProto
    onnx_light.onnx.DeviceConfigurationProto
    onnx_light.onnx.NodeDeviceConfigurationProto
    onnx_light.onnx.ShardedDimProto
    onnx_light.onnx.SimpleShardedDimProto
    onnx_light.onnx.ShardingSpecProto
    onnx_light.onnx.OperatorStatus
