runtime_value
=============

``RuntimeValue`` represents named ``TypeProto.struct_type`` fields and
``EncodedValueProto`` payloads. Tensor leaves reuse ``Tensor``. Ordinary tensor
edges remain exclusively in ``RuntimeContext::tensors()``; ``values()`` is for
structured/encoded edges, not a parallel tensor store.

The representation is independent of persistence. Encoded messages have one
immutable shared owner from construction, not mutable inline storage plus a
second promoted representation. ``BorrowView()`` copies only metadata and
existing owner handles, without changing the source. ``DeepCopy()`` is explicit.
``std::move(value).Retain()`` consumes a selected whole result, recursively moving
owned tensor buffers or exporting existing I/O allocations as ``IOLease`` handles.
An ownerless borrow or unleased execution-arena allocation cannot be retained.

Structured ``Identity`` forwarding uses normal ``RuntimeValue`` copy semantics:
owned tensor storage is copied, while existing tensor owners and immutable
encoded messages are shared. Unlike ``BorrowView()``, this keeps inline-owned
intermediate fields valid when their source is released, including when there
are multiple consumers. It does not mutate or promote the source.

.. doxygenfile:: onnx_core/runtime/runtime_value.h
    :project: onnx-light
