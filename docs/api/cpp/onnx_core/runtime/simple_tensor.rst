simple_tensor.h
===============

Explicit storage retention
--------------------------

``std::move(tensor).RetainStorage()`` transfers owned buffers into a read-only
shared view without copying payloads. Existing borrowed owners are reused and
``IOArena`` allocations acquire self-owning leases. Ownerless borrows and
unleased execution-arena allocations are rejected.

``BorrowView()`` is a const, nonmutating operation: it copies only metadata and
any existing owner token. Without such a token, the source must outlive the
view. ``BorrowStrings`` accepts an optional owner token for the same contract.
These APIs never move storage out of a const tensor.

.. doxygenfile:: onnx_core/runtime/memory/simple_tensor.h
   :project: onnx-light
