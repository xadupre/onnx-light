.. _l-next-steps-graph-builder-quantized-tensor:

``QuantizedTensorProto`` in ``GraphBuilder``
============================================

:Date: 2026-08

Objective
+++++++++

``GraphBuilder`` must preserve quantized initializers without converting them
to ``TensorProto`` or dequantizing them.

Graph storage
+++++++++++++

``GraphProto`` adds:

.. code-block:: text

    repeated QuantizedTensorProto quantized_initializer = <N>;

``GraphBuilder`` stores these values unchanged and exposes:

.. code-block:: cpp

    const std::string &
    MakeQuantizedInitializer(const QuantizedTensorProto &value);

    const RepeatedProtoField<QuantizedTensorProto> &
    QuantizedInitializers() const noexcept;

Names are shared with inputs, ordinary initializers, and node outputs.
External payloads remain external.

``ShapesContext``
+++++++++++++++++

``ShapesContext`` is the source of truth for value information. It owns the
model quantization catalogue and one descriptor per quantized value:

.. code-block:: cpp

    class SymQuantizedTensor {
    public:
      QuantizationId QuantizedType() const;
      const SymShape &Shape() const;
      TensorType DecodedType() const;
      uint64_t ByteSize() const;
    };

    RepeatedProtoField<QuantizationProto> quantizations_;
    std::unordered_map<std::string, SymQuantizedTensor> quantized_tensors_;

The context exposes ``AddQuantization``, ``GetQuantization``,
``SetQuantizedTensor``, ``HasQuantizedTensor``, and
``GetQuantizedTensor``. A name appears in exactly one value map.

Inference
+++++++++

``ComputeShapeModel`` registers ``ModelProto.quantizations`` before processing
the graph. ``ComputeShapeGraph`` and
``GraphBuilder::MakeQuantizedInitializer`` use the same helper:

1. resolve ``quantized_type`` or the inline quantization;
2. validate dimensions and payload size;
3. derive the decoded element type and logical shape;
4. store ``SymQuantizedTensor`` in ``ShapesContext``.

An ordinary tensor operator cannot consume a quantized value implicitly. It
must use an explicit decoder or a schema accepting
``SymQuantizedTensor``. ``SchemaInputValue`` therefore adds this descriptor.

Scopes
++++++

Subgraphs inherit visible quantized values and the quantization catalogue.
Local functions inherit the catalogue but receive values only through formal
inputs. Input and output binding copies the complete descriptor.

Serialization and passes
++++++++++++++++++++++++

``ToModel`` writes referenced quantization declarations and remaps their
indices. ``ToGraph`` rejects model-level quantization references because a
standalone graph cannot resolve them.

Initializer passes include quantized initializers. Duplicate removal compares
the resolved quantization, dimensions, payload, and interpretation metadata;
equal bytes alone are insufficient.

Implementation order
++++++++++++++++++++

1. Add the graph field and ``SymQuantizedTensor``.
2. Extend ``ShapesContext`` and schema validation.
3. Add ``GraphBuilder`` import and serialization.
4. Extend subgraphs, functions, and initializer passes.
5. Test incremental inference and model round-trips.
