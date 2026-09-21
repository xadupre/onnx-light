runtime_session.h
=================

Ordinary construction eagerly snapshots initializers, independently of the
source graph. These cached tensors have explicit shared owners, so a selected
subgraph initializer output can outlive its session without a payload copy.
Nonpersistent outputs still materialize borrowed storage normally.

``InitializerMode::kBorrowed`` instead reads initializer storage from the
immutable graph. The context supplies its model lifetime token; ownerless
externally borrowed buffers cannot be retained as selected results.

.. doxygenfile:: onnx_core/runtime/runtime_session.h
   :project: onnx-light
