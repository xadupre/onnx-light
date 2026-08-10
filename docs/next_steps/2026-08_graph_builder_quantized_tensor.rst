.. _l-next-steps-graph-builder-quantized-tensor:

Quantized values in ``GraphBuilder``
====================================

:Date: 2026-08

**in progress**

Objective
+++++++++

``GraphBuilder`` must preserve quantized initializers without converting them
to ``TensorProto`` or dequantizing them.

The recommended representation is ``StructProto`` plus ``StructTypeProto``,
as defined in :ref:`l-next-steps-custom-types`. If
``QuantizedTensorProto`` is retained, the same design applies with specialized
names.

Graph storage
+++++++++++++

``GraphProto`` needs a field parallel to ``initializer``:

.. code-block:: text

    repeated StructProto structured_initializer = <N>;

``GraphBuilder`` stores these protos unchanged and exposes:

.. code-block:: cpp

    const std::string &MakeStructuredInitializer(const StructProto &value);
    const RepeatedProtoField<StructProto> &
    StructuredInitializers() const noexcept;

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

    class SymStruct {
    public:
      const StructTypeRef &PhysicalType() const;
      const std::vector<uint64_t> &Dims() const;
      uint64_t ByteSize() const;
      const TypeProto *LogicalType() const;
      const SymTensor *LogicalTensor() const;
    };

``ShapesContext`` then owns:

.. code-block:: cpp

    std::unordered_map<std::string, SymStruct> structs_;
    RepeatedProtoField<StructTypeProto> struct_types_;

with ``SetStruct``, ``HasStruct``, ``GetStruct``, ``AddStructType``, and
``GetStructType``.

``SymStruct`` keeps both views of the value:

* physical type, dimensions, and checked byte size;
* decoded logical type and, when applicable, its ``SymTensor``.

The payload itself remains in ``GraphBuilder``. A value name appears in only
one context map. Availability checks must cover tensors, sequences, and
structures.

Inference
+++++++++

``ComputeShapeModel`` registers model-level structured types before the graph
is processed. ``ComputeShapeGraph`` seeds structured initializers through the
same helper used by ``GraphBuilder::MakeStructuredInitializer``.

For ``StructProto``, the helper:

1. resolves the inline or model-level physical type;
2. binds ``StructProto.dims``;
3. validates the payload size;
4. reads the decoder output type;
5. creates and stores ``SymStruct``.

A tensor operator must not receive ``LogicalTensor()`` implicitly. It needs an
explicit decoder, unless its schema accepts the structured value directly.
``LightOpSchema::SchemaInputValue`` must therefore support ``SymStruct``.

Scopes
++++++

Subgraph contexts inherit outer structured values and the structured-type
catalogue. Local functions inherit the catalogue but not outer values.
Function input and output binding copies the complete ``SymStruct``.

Serialization and passes
++++++++++++++++++++++++

``ModelProto -> GraphBuilder -> ModelProto`` must preserve payloads, types,
dimensions, and model-level references. ``ToModel`` compacts the type
catalogue and remaps indices. ``ToGraph`` rejects model-level references
because a standalone graph cannot resolve them.

Passes handling initializers must include structured initializers. Duplicate
removal compares the resolved physical type, dimensions, payload, and
interpretation metadata; equal bytes alone are insufficient.

Implementation order
++++++++++++++++++++

1. Add the proto field and ``SymStruct``.
2. Extend ``ShapesContext`` and schema validation.
3. Add ``GraphBuilder`` storage, import, and serialization.
4. Extend subgraphs, functions, and initializer passes.
5. Test incremental inference and model round-trips.
