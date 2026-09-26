======
protos
======

Relations between protos
========================

The following graph shows containment relations between ONNX protos and the
onnx-light extensions, including structured types, encoded values, persistent
bindings and paged caches. Solid edges represent nested messages; dashed edges
show selected model-scoped references, not containment or storage ownership.
Each edge label is the attribute name (or names) that carries the nested proto.
The SVG below is generated from
:download:`protos_relations.dot <_static/protos_relations.dot>` with
``dot -Tsvg`` (Graphviz); regenerate it after editing the ``.dot`` source.
``unittests/main/test_proto_relations_sync.py`` checks containment edges against
``onnx_light/onnx_proto/onnx.h`` and checks the generated SVG and text trees
against the DOT.

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
references are explicitly marked rather than expanded as nested messages.

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
   │   │   │   ├── optional_type → TypeProto.Optional
   │   │   │   │   └── elem_type → TypeProto (↑)
   │   │   │   ├── opaque_type → TypeProto.Opaque
   │   │   │   └── struct_type → StructTypeProto (expanded below)
   │   │   └── metadata_props → StringStringEntryProto
   │   ├── quantization_annotation → TensorAnnotation
   │   │   └── quant_parameter_tensor_names → StringStringEntryProto
   │   ├── encoded_initializer → EncodedValueProto
   │   │   ├── affine → AffineLayoutProto
   │   │   │   └── scale, zero_point → TensorProto (↑)
   │   │   ├── struct_type → StructTypeProto (expanded below)
   │   │   ├── logical_type → TypeProto (↑)
   │   │   ├── external_data → StringStringEntryProto
   │   │   └── parameter_ref → TensorAnnotation (model catalogue name reference)
   │   ├── persistent_bindings → PersistentBindingProto
   │   │   └── input_name, output_name → ValueInfoProto (graph IO name references)
   │   ├── paged_cache_initializer → PagedCacheProto
   │   │   └── blocks → PagedCacheBlockProto
   │   │       ├── key, value → TensorProto (↑)
   │   │       └── encoded_key, encoded_value → EncodedValueProto (↑)
   │   └── metadata_props → StringStringEntryProto
   ├── opset_import → OperatorSetIdProto
   ├── functions → FunctionProto
   │   ├── attribute_proto → AttributeProto (↑)
   │   ├── node → NodeProto (↑)
   │   ├── opset_import → OperatorSetIdProto
   │   ├── value_info → ValueInfoProto (↑)
   │   └── metadata_props → StringStringEntryProto
   ├── configuration → DeviceConfigurationProto
   ├── struct_types → StructTypeProto
   │   ├── array → StructTypeProto.Array
   │   │   └── element_type → TypeProto (↑)
   │   ├── structure → StructTypeProto.Structure
   │   │   └── field → StructTypeProto.Structure.Field
   │   │       ├── type → TypeProto (↑)
   │   │       └── constant → TensorProto (↑)
   │   ├── bit_packing → StructTypeProto.BitPacking
   │   │   └── component → StructTypeProto.BitPacking.Component
   │   ├── decoder, encoder → FunctionProto (↑)
   │   ├── metadata_props → StringStringEntryProto
   │   └── type_ref → StructTypeProto (model catalogue ID reference)
   └── metadata_props → StringStringEntryProto

``StructTypeProto.type_ref`` resolves an ID in ``ModelProto.struct_types``.
``EncodedValueProto.parameter_ref`` selects a shared numerical parameter set
declared by a root-graph ``TensorAnnotation`` whose ``tensor_name`` is
``onnx_light.quantization.parameters:<name>``. Its tensor-name mappings point
to graph initializers; those tensors are not embedded in the encoded value.
See :doc:`../../howto/quantized_values` for this catalogue.

The ``key``/``encoded_key`` and ``value``/``encoded_value`` edges are independent
oneof alternatives, not four simultaneously required payloads. The cache
itself is contained in the graph initializer, while its encoded pages may
reference model-level type and parameter catalogues.

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

* :doc:`model_proto`: ``graph``, ``opset_import``, ``functions``, ``configuration``, ``metadata_props``, ``struct_types``
* :doc:`graph_proto`: ``node``, ``initializer``, ``sparse_initializer``, ``input``, ``output``, ``value_info``, ``quantization_annotation``, ``metadata_props``, ``encoded_initializer``, ``persistent_bindings``, ``paged_cache_initializer``
* :doc:`function_proto`: ``attribute_proto``, ``node``, ``opset_import``, ``value_info``, ``metadata_props``
* :doc:`node_proto`: ``attribute``, ``device_configurations``, ``metadata_props``
* :doc:`node_device_configuration_proto`: ``sharding_spec`` (and ``configuration_id``, a name reference to a :doc:`device_configuration_proto` declared in ``ModelProto.configuration``)
* :doc:`sharding_spec_proto`: ``index_to_device_group_map``, ``sharded_dim``
* :doc:`sharded_dim_proto`: ``simple_sharding``
* :doc:`value_info_proto`: ``type``, ``metadata_props``
* :doc:`tensor_shape_proto`: ``dim``
* :doc:`type_proto`: ``tensor_type`` (:doc:`TypeProto.Tensor <type_proto>`), ``sparse_tensor_type`` (:doc:`TypeProto.SparseTensor <type_proto>`), ``sequence_type`` (:doc:`TypeProto.Sequence <type_proto>`), ``map_type`` (:doc:`TypeProto.Map <type_proto>`), ``optional_type`` (:doc:`TypeProto.Optional <type_proto>`), ``opaque_type`` (``TypeProto.Opaque``), ``struct_type`` (:doc:`struct_type_proto`)
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
* :doc:`struct_type_proto`: ``array``, ``structure``, ``bit_packing``, ``decoder``, ``encoder``, ``metadata_props`` (``type_ref`` is a catalogue ID reference)
* ``StructTypeProto.Array``: ``element_type``; ``StructTypeProto.Structure``: ``field``; ``StructTypeProto.Structure.Field``: ``type`` or ``constant``; ``StructTypeProto.BitPacking``: ``component``
* :doc:`encoded_value_proto`: ``affine``, ``struct_type``, ``logical_type``, ``external_data`` (``parameter_ref`` is a catalogue name reference)
* :doc:`affine_layout_proto`: ``scale``, ``zero_point``
* :doc:`paged_cache_proto`: ``blocks``; ``PagedCacheBlockProto``: ``key`` or ``encoded_key``, ``value`` or ``encoded_value``
* :doc:`PersistentBindingProto <graph_proto>` has no nested messages; its two names select graph IO

.. toctree::
    :maxdepth: 1

    affine_layout_proto
    attribute_proto
    device_configuration_proto
    encoded_value_proto
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
    paged_cache_proto
    sequence_proto
    sharded_dim_proto
    sharding_spec_proto
    simple_sharded_dim_proto
    sparse_tensor_proto
    string_string_entry_proto
    struct_type_proto
    tensor_annotation
    tensor_proto
    tensor_shape_proto
    type_proto
    value_info_proto
