.. _l-next-steps-graph-builder-quantized-tensor:

Quantized values in ``GraphBuilder``
====================================

:Date: 2026-08

**consolidated design reference**

.. note::

    GraphBuilder integration is PR05 of
    :ref:`l-next-steps-prepared-values-and-persistent-state`. The API sketches
    below are proposals, not existing interfaces. Use the unified
    plan for the representation, ownership and implementation sequence.
    In particular, preserve ``EncodedValueProto.storage_shape`` separately
    from the decoded logical shape and share ``StructTypeProto`` declarations
    across values. A different physical array size does not create a new type.

Objective
+++++++++

``GraphBuilder`` must preserve quantized initializers without converting them
to ``TensorProto`` or dequantizing them.

The representation is ``EncodedValueProto``. Its structured layout uses
``StructTypeProto`` as defined in :ref:`l-next-steps-custom-types`; the same
value container also supports the small built-in layout subset. There is no
separate structured or quantized value container.

Graph storage
+++++++++++++

``GraphProto`` needs a field parallel to ``initializer``:

.. code-block:: text

    repeated EncodedValueProto encoded_initializer = <N>;

``GraphBuilder`` stores these protos unchanged and exposes:

.. code-block:: cpp

    const std::string &MakeEncodedInitializer(const EncodedValueProto &value);
    const RepeatedProtoField<EncodedValueProto> &
    EncodedInitializers() const noexcept;

Names are shared with inputs, ordinary initializers, and node outputs.
External data remains external.

``ShapesContext``
+++++++++++++++++

``ShapesContext`` is the source of truth for value information. It already
contains tensors, sequences, opsets, functions, constraints, and subgraph
contexts. Quantized information must be added there, not in a second
``GraphBuilder`` registry.

Add a symbolic descriptor:

.. code-block:: cpp

    class SymEncodedValue {
    public:
      const EncodedLayoutRef &PhysicalLayout() const;
      const std::vector<int64_t> &StorageShape() const;
      uint64_t ByteSize() const;
      const TypeProto *LogicalType() const;
      const SymTensor *LogicalTensor() const;
    };

``ShapesContext`` then owns:

.. code-block:: cpp

    std::unordered_map<std::string, SymEncodedValue> encoded_values_;
    RepeatedProtoField<StructTypeProto> struct_types_;
    std::unordered_map<uint64_t, size_t> struct_type_positions_;

with ``SetEncodedValue``, ``HasEncodedValue``, ``GetEncodedValue``,
``AddStructType``, and ``GetStructType``. ``EncodedLayoutRef`` selects a
built-in layout or a resolved structured type. Type lookup uses stable
``type_id`` values; catalogue positions are internal lookup details.

``SymEncodedValue`` keeps both views of the value:

* physical layout, storage shape, and checked byte size;
* decoded logical type and, when applicable, its ``SymTensor``.

The payload itself remains in ``GraphBuilder``. A value name appears in only
one context map. Availability checks must cover tensors, sequences, and
encoded values.

Inference
+++++++++

``ComputeShapeModel`` registers model-level structured types before the graph
is processed. ``ComputeShapeGraph`` seeds encoded initializers through the
same helper used by ``GraphBuilder::MakeEncodedInitializer``.

For ``EncodedValueProto``, the helper:

1. resolves the selected built-in layout or inline/ID-referenced structured type;
2. binds ``EncodedValueProto.storage_shape``;
3. validates the payload size;
4. validates the optional logical type against the decoder or registered layout;
5. creates and stores ``SymEncodedValue``.

A tensor operator must not receive ``LogicalTensor()`` implicitly. It needs an
explicit decoder, unless its schema accepts the encoded value directly.
``LightOpSchema::SchemaInputValue`` must therefore support ``SymEncodedValue``.

Scopes
++++++

Subgraph contexts inherit outer encoded values and the structured-type
catalogue. Local functions inherit the catalogue but not outer values.
Function input and output binding copies the complete ``SymEncodedValue``.

Serialization and passes
++++++++++++++++++++++++

``ModelProto -> GraphBuilder -> ModelProto`` must preserve payloads, types,
storage and logical shapes, and stable type references. ``ToModel`` may
compact unused catalogue entries but preserves their ``type_id`` values;
conflicting definitions under the same ID are rejected, not silently
renumbered. ``ToGraph`` rejects remaining model-level references unless their
declarations are exported inline; a standalone graph has no model catalogue.

Passes handling initializers must include encoded initializers. Duplicate
removal compares the resolved physical layout, storage and logical shapes,
payload, and interpretation metadata; equal bytes alone are insufficient.

Implementation order
++++++++++++++++++++

1. Add the proto field and ``SymEncodedValue``.
2. Extend ``ShapesContext`` and schema validation.
3. Add ``GraphBuilder`` storage, import, and serialization.
4. Extend subgraphs, functions, and initializer passes.
5. Test incremental inference and model round-trips.
