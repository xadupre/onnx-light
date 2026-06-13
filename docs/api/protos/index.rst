======
protos
======

Relations between protos
========================

The following graph shows containment relations between ONNX protos.
Each edge label is the attribute name (or names) that carries the nested proto.
The SVG below is generated from
:download:`protos_relations.dot <_static/protos_relations.dot>` with
``dot -Tsvg`` (Graphviz); regenerate it after editing the ``.dot`` source.

Click the diagram to open it in a full-screen view where the mouse wheel zooms
and dragging pans, which makes the smaller labels easier to read.

.. image:: _static/protos_relations.svg
   :alt: Containment relations between ONNX protos
   :align: center
   :class: zoomable-svg

ASCII tree
==========

The same containment relations as a text tree, rooted at
:doc:`model_proto`. Edge labels are the attribute names; ``(↑)`` marks a proto
already expanded earlier in the tree (the graph contains cycles), and
``configuration_id`` is a name reference rather than a nested message.

.. code-block:: text

   ModelProto
   ├── graph → GraphProto
   │   ├── node → NodeProto
   │   │   ├── attribute → AttributeProto
   │   │   │   ├── t, tensors → TensorProto
   │   │   │   ├── g, graphs → GraphProto (↑)
   │   │   │   ├── sparse_tensor, sparse_tensors → SparseTensorProto
   │   │   │   └── tp, type_protos → TypeProto
   │   │   ├── device_configurations → NodeDeviceConfigurationProto
   │   │   │   ├── sharding_spec → ShardingSpecProto
   │   │   │   │   ├── index_to_device_group_map → IntIntListEntryProto
   │   │   │   │   └── sharded_dim → ShardedDimProto
   │   │   │   │       └── simple_sharding → SimpleShardedDimProto
   │   │   │   └── configuration_id → DeviceConfigurationProto (name reference)
   │   │   └── metadata_props → StringStringEntryProto
   │   ├── initializer → TensorProto
   │   │   ├── segment → Segment
   │   │   └── external_data, metadata_props → StringStringEntryProto
   │   ├── sparse_initializer → SparseTensorProto
   │   │   └── values, indices → TensorProto
   │   ├── input, output, value_info → ValueInfoProto
   │   │   ├── type → TypeProto
   │   │   │   ├── tensor_type → TypeProto.Tensor
   │   │   │   │   └── shape → TensorShapeProto
   │   │   │   │       └── dim → Dimension
   │   │   │   ├── sparse_tensor_type → TypeProto.SparseTensor
   │   │   │   │   └── shape → TensorShapeProto (↑)
   │   │   │   ├── sequence_type → TypeProto.Sequence
   │   │   │   │   └── elem_type → TypeProto (↑)
   │   │   │   ├── map_type → TypeProto.Map
   │   │   │   │   └── value_type → TypeProto (↑)
   │   │   │   └── optional_type → TypeProto.Optional
   │   │   │       └── elem_type → TypeProto (↑)
   │   │   └── metadata_props → StringStringEntryProto
   │   ├── quantization_annotation → TensorAnnotation
   │   │   └── quant_parameter_tensor_names → StringStringEntryProto
   │   └── metadata_props → StringStringEntryProto
   ├── opset_import → OperatorSetIdProto
   ├── functions → FunctionProto
   │   ├── attribute_proto → AttributeProto (↑)
   │   ├── node → NodeProto (↑)
   │   ├── opset_import → OperatorSetIdProto
   │   ├── value_info → ValueInfoProto (↑)
   │   └── metadata_props → StringStringEntryProto
   ├── configuration → DeviceConfigurationProto
   └── metadata_props → StringStringEntryProto

The runtime container protos form a separate cycle of their own:

.. code-block:: text

   SequenceProto
   ├── tensor_values → TensorProto
   ├── sparse_tensor_values → SparseTensorProto
   ├── sequence_values → SequenceProto (↑)
   ├── map_values → MapProto
   │   └── values → SequenceProto (↑)
   └── optional_values → OptionalProto
       ├── tensor_value → TensorProto
       ├── sparse_tensor_value → SparseTensorProto
       ├── sequence_value → SequenceProto (↑)
       ├── map_value → MapProto (↑)
       └── optional_value → OptionalProto (↑)

Containment attributes
======================

Quick attribute list used in the graph:

* :doc:`model_proto`: ``graph``, ``opset_import``, ``functions``, ``configuration``, ``metadata_props``
* :doc:`graph_proto`: ``node``, ``initializer``, ``sparse_initializer``, ``input``, ``output``, ``value_info``, ``quantization_annotation``, ``metadata_props``
* :doc:`function_proto`: ``attribute_proto``, ``node``, ``opset_import``, ``value_info``, ``metadata_props``
* :doc:`node_proto`: ``attribute``, ``device_configurations``, ``metadata_props``
* :doc:`node_device_configuration_proto`: ``sharding_spec`` (and ``configuration_id``, a name reference to a :doc:`device_configuration_proto` declared in ``ModelProto.configuration``)
* :doc:`sharding_spec_proto`: ``index_to_device_group_map``, ``sharded_dim``
* :doc:`sharded_dim_proto`: ``simple_sharding``
* :doc:`value_info_proto`: ``type``, ``metadata_props``
* :doc:`tensor_shape_proto`: ``dim``
* :doc:`type_proto`: ``tensor_type`` (:doc:`TypeProto.Tensor <type_proto>`), ``sparse_tensor_type`` (:doc:`TypeProto.SparseTensor <type_proto>`), ``sequence_type`` (:doc:`TypeProto.Sequence <type_proto>`), ``map_type`` (:doc:`TypeProto.Map <type_proto>`), ``optional_type`` (:doc:`TypeProto.Optional <type_proto>`)
* :doc:`TypeProto.Tensor <type_proto>` / :doc:`TypeProto.SparseTensor <type_proto>`: ``shape`` (a :doc:`tensor_shape_proto`)
* :doc:`TypeProto.Sequence <type_proto>` / :doc:`TypeProto.Optional <type_proto>`: ``elem_type`` (a :doc:`type_proto`)
* :doc:`TypeProto.Map <type_proto>`: ``value_type`` (a :doc:`type_proto`)
* :doc:`tensor_proto`: ``segment``, ``external_data``, ``metadata_props``
* :doc:`sparse_tensor_proto`: ``values``, ``indices``
* :doc:`attribute_proto`: ``t``, ``tensors``, ``g``, ``graphs``, ``sparse_tensor``, ``sparse_tensors``, ``tp``, ``type_protos``
* :doc:`tensor_annotation`: ``quant_parameter_tensor_names``
* :doc:`sequence_proto`: ``tensor_values``, ``sparse_tensor_values``, ``sequence_values``, ``map_values``, ``optional_values``
* :doc:`map_proto`: ``values``
* :doc:`optional_proto`: ``tensor_value``, ``sparse_tensor_value``, ``sequence_value``, ``map_value``, ``optional_value``

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
