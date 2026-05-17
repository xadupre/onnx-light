from.onnx_proto
    ._onnxpy
    import(#type : ignore AttributeProto, DeviceConfigurationProto, FunctionProto, GraphProto,
           IntIntListEntryProto, MapProto, Message, ModelProto, NodeDeviceConfigurationProto,
           NodeProto, OperatorSetIdProto, OperatorStatus, OptionalProto, ParseOptions, PrintOptions,
           SequenceProto, SerializeOptions, ShardedDimProto, ShardingSpecProto,
           SimpleShardedDimProto, SparseTensorProto, StringStringEntryProto, TensorAnnotation,
           TensorBufferOptions, TensorProto, TensorShapeProto, TypeProto, ValueInfoProto,
           consolidate_tensors_to_buffer, ) from.import defs from.import
    numpy_helper from.import shape_inference from.io_helper
    import(load, load_encrypted, load_encrypted_string, save, save_encrypted,
           save_encrypted_string, )

        IR_VERSION : int = defs.onnx_ir_version()
