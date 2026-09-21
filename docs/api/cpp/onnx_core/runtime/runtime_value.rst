runtime_value
=============

``RuntimeValue`` represents named ``TypeProto.struct_type`` fields and
``EncodedValueProto`` payloads. Tensor leaves reuse ``Tensor``. Ordinary tensor
edges remain exclusively in ``RuntimeContext::tensors()``; ``values()`` is for
structured/encoded edges, not a parallel tensor store.

The representation is independent of persistence. Encoded messages have one
immutable shared owner from construction, not mutable inline storage plus a
second promoted representation. ``BorrowView()`` copies numeric metadata and
existing owner handles, without changing the source. Ordinary string fields
are materialized instead, as required by normal string execution.
``DeepCopy()`` copies all payloads explicitly.
``std::move(value).Retain()`` consumes a selected whole result, recursively moving
owned tensor buffers or exporting existing I/O allocations as ``IOLease`` handles.
An ownerless borrow or unleased execution-arena allocation cannot be retained.
String tensors cannot be retained, including nested fields and constants in an
encoded layout. ``Retain(catalogue)`` resolves encoded layout references through
the model catalogue; unresolved references and recursive types are rejected.

Structured ``Identity`` forwarding uses normal ``RuntimeValue`` copy semantics:
owned tensor storage is copied, while existing tensor owners and immutable
encoded messages are shared. Unlike ``BorrowView()``, this keeps inline-owned
intermediate fields valid when their source is released, including when there
are multiple consumers. It does not mutate or promote the source.

.. doxygenfile:: onnx_core/runtime/runtime_value.h
    :project: onnx-light
