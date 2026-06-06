======
protos
======

Relations between protos
========================

The following Mermaid graph shows containment relations between ONNX protos.
Each edge label is the attribute name (or names) that carries the nested proto.

.. runmermaid::

    %%{init: {'flowchart': {'useMaxWidth': false, 'nodeSpacing': 60, 'rankSpacing': 70}, 'themeVariables': {'fontSize': '18px'}}}%%
    flowchart TD
        ModelProto -->|graph| GraphProto
        ModelProto -->|opset_import| OperatorSetIdProto
        ModelProto -->|functions| FunctionProto
        ModelProto -->|configuration| DeviceConfigurationProto
        ModelProto -->|metadata_props| StringStringEntryProto

        GraphProto -->|node| NodeProto
        GraphProto -->|initializer| TensorProto
        GraphProto -->|sparse_initializer| SparseTensorProto
        GraphProto -->|input, output, value_info| ValueInfoProto
        GraphProto -->|quantization_annotation| TensorAnnotation
        GraphProto -->|metadata_props| StringStringEntryProto

        FunctionProto -->|attribute_proto| AttributeProto
        FunctionProto -->|node| NodeProto
        FunctionProto -->|opset_import| OperatorSetIdProto
        FunctionProto -->|value_info| ValueInfoProto
        FunctionProto -->|metadata_props| StringStringEntryProto

        NodeProto -->|attribute| AttributeProto
        NodeProto -->|device_configurations| NodeDeviceConfigurationProto
        NodeProto -->|metadata_props| StringStringEntryProto

        NodeDeviceConfigurationProto -->|sharding_spec| ShardingSpecProto
        NodeDeviceConfigurationProto -. configuration_id .-> DeviceConfigurationProto
        ShardingSpecProto -->|index_to_device_group_map| IntIntListEntryProto
        ShardingSpecProto -->|sharded_dim| ShardedDimProto
        ShardedDimProto -->|simple_sharding| SimpleShardedDimProto

        ValueInfoProto -->|type| TypeProto
        ValueInfoProto -->|metadata_props| StringStringEntryProto

        TensorShapeProto -->|dim| Dimension

        TypeProto -->|tensor_type| TypeProto.Tensor
        TypeProto -->|sparse_tensor_type| TypeProto.SparseTensor
        TypeProto -->|sequence_type| TypeProto.Sequence
        TypeProto -->|map_type| TypeProto.Map
        TypeProto -->|optional_type| TypeProto.Optional
        TypeProto.Tensor -->|shape| TensorShapeProto
        TypeProto.SparseTensor -->|shape| TensorShapeProto
        TypeProto.Sequence -->|elem_type| TypeProto
        TypeProto.Optional -->|elem_type| TypeProto
        TypeProto.Map -->|value_type| TypeProto

        TensorProto -->|segment| Segment
        TensorProto -->|external_data, metadata_props| StringStringEntryProto

        SparseTensorProto -->|values, indices| TensorProto

        AttributeProto -->|t, tensors| TensorProto
        AttributeProto -->|g, graphs| GraphProto
        AttributeProto -->|sparse_tensor, sparse_tensors| SparseTensorProto
        AttributeProto -->|tp, type_protos| TypeProto

        TensorAnnotation -->|quant_parameter_tensor_names| StringStringEntryProto

        SequenceProto -->|tensor_values| TensorProto
        SequenceProto -->|sparse_tensor_values| SparseTensorProto
        SequenceProto -->|sequence_values| SequenceProto
        SequenceProto -->|map_values| MapProto
        SequenceProto -->|optional_values| OptionalProto

        MapProto -->|values| SequenceProto

        OptionalProto -->|tensor_value| TensorProto
        OptionalProto -->|sparse_tensor_value| SparseTensorProto
        OptionalProto -->|sequence_value| SequenceProto
        OptionalProto -->|map_value| MapProto
        OptionalProto -->|optional_value| OptionalProto

Containment attributes
======================

Quick attribute list used in the graph:

* ``ModelProto``: ``graph``, ``opset_import``, ``functions``, ``configuration``, ``metadata_props``
* ``GraphProto``: ``node``, ``initializer``, ``sparse_initializer``, ``input``, ``output``, ``value_info``, ``quantization_annotation``, ``metadata_props``
* ``FunctionProto``: ``attribute_proto``, ``node``, ``opset_import``, ``value_info``, ``metadata_props``
* ``NodeProto``: ``attribute``, ``device_configurations``, ``metadata_props``
* ``NodeDeviceConfigurationProto``: ``sharding_spec`` (and ``configuration_id``, a name reference to a ``DeviceConfigurationProto`` declared in ``ModelProto.configuration``)
* ``ShardingSpecProto``: ``index_to_device_group_map``, ``sharded_dim``
* ``ShardedDimProto``: ``simple_sharding``
* ``ValueInfoProto``: ``type``, ``metadata_props``
* ``TensorShapeProto``: ``dim``
* ``TypeProto``: ``tensor_type`` (``TypeProto.Tensor``), ``sparse_tensor_type`` (``TypeProto.SparseTensor``), ``sequence_type`` (``TypeProto.Sequence``), ``map_type`` (``TypeProto.Map``), ``optional_type`` (``TypeProto.Optional``)
* ``TypeProto.Tensor`` / ``TypeProto.SparseTensor``: ``shape`` (a ``TensorShapeProto``)
* ``TypeProto.Sequence`` / ``TypeProto.Optional``: ``elem_type`` (a ``TypeProto``)
* ``TypeProto.Map``: ``value_type`` (a ``TypeProto``)
* ``TensorProto``: ``segment``, ``external_data``, ``metadata_props``
* ``SparseTensorProto``: ``values``, ``indices``
* ``AttributeProto``: ``t``, ``tensors``, ``g``, ``graphs``, ``sparse_tensor``, ``sparse_tensors``, ``tp``, ``type_protos``
* ``TensorAnnotation``: ``quant_parameter_tensor_names``
* ``SequenceProto``: ``tensor_values``, ``sparse_tensor_values``, ``sequence_values``, ``map_values``, ``optional_values``
* ``MapProto``: ``values``
* ``OptionalProto``: ``tensor_value``, ``sparse_tensor_value``, ``sequence_value``, ``map_value``, ``optional_value``

.. toctree::
    :maxdepth: 1

    attribute_proto
    device_configuration_proto
    function_proto
    graph_proto
    int_int_list_entry_proto
    map_proto
    message
    model_proto
    node_device_configuration_proto
    node_proto
    operator_set_id_proto
    operator_status
    optional_proto
    sequence_proto
    sharded_dim_proto
    sharding_spec_proto
    simple_sharded_dim_proto
    sparse_tensor_proto
    string_string_entry_proto
    tensor_annotation
    tensor_proto
    tensor_shape_proto
    type_proto
    value_info_proto
