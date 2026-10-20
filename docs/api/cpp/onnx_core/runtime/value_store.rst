value_store.h
==========================

``ValueStore`` owns immutable values prepared synchronously while
a ``RuntimeSession`` hierarchy initializes its kernels. ``RuntimeContext``
always exposes a store, and child contexts inherit the same shared instance.
Each session owns a non-null store from construction and installs it into
the context before initializing kernels. Nested graph/function sessions
explicitly receive their parent's owner. Independent sessions do not adopt
the store of a reused context; cross-session sharing requires
``RuntimeSessionOptions::value_store``.
Kernels bind a ``ValueStore::Slot`` during initialization. This opaque handle
associates a stable ``uint32_t`` index with the originating store. ``IsReady``,
``Publish``, and ``Get`` reject handles from any other store, even when both
stores use the same numeric index. Raw integers cannot be used as handles;
``Slot::index()`` exposes the index only for diagnostics.

``Find(key) const`` looks up an existing slot without binding or allocating one.
It returns ``std::nullopt`` for an unknown key (including an empty key), and a
store-bound slot for a known key, even if its value has not been published.
Call ``IsReady(*slot)`` to check publication before ``Get`` or ``BindForRead``;
both reads reject unpublished slots. Estimators can use this lookup through
``const RuntimeContext::value_store()`` with the same execution cache key used
by preparation. An already published artifact needs no additional prepared
allocation: its storage is already included in ``prepared_bytes()``.
Lookups neither change that total nor create slots, and only search this store,
not independent stores with the same keys.

After publication, ``BindForRead(slot)`` checks ownership, bounds, and readiness
once and returns a ``ValueStore::BoundSlot``. It contains only the store pointer
and a ``uint32_t`` index. Its parameterless ``Get()`` reads the immutable value
directly: no identity comparison, ownership copy, bounds/readiness validation,
map lookup, mutex, readiness wait, or scheduler dispatch occurs on the hot path.
There is no public unchecked ``Get(uint32_t)`` that could combine an arbitrary
index with another store. ``Gemm`` retains this validated binding, not the
identity-carrying setup handle. All sessions sharing a store must
serialize their runs. Its allocations remain alive until the last shared
owner, including any context, is released.

A slot retains only a small identity token, not the store or its prepared
allocations. The token remains distinct if the original store is destroyed
and another is constructed at the same address. Default-constructed and
moved-from setup slots are rejected. Checked store operations take setup
handles by const reference, while replay bindings contain no shared pointer.
Bindings can only be constructed after successful validation and cannot be
default-constructed. They borrow the original store, which must remain alive
through every read; neither kind of handle extends its allocation lifetime.

The location under ``runtime/kernels/`` reflects its kernel-preparation role:
it is not the invocation value directory. "Static" means publication once per
slot followed by immutable replay, not a global C++ object or mutation-free
initialization. ``value_store()`` returns a reference;
``value_store_owner()`` returns shared ownership. Both session and context
accessors expose only read-only views through const objects.

This store does not schedule asynchronous loading, track generations, evict
values, or persist prepared payloads.

``prepared_bytes()`` returns an incrementally maintained live-byte total in
constant time, including allocated but unpublished values. Each allocation
and release updates the total, including reuse through a tensor's allocator.
``accounting_allocator_visits()`` exposes allocator live-byte query visits for
deterministic scaling diagnostics; reading the total adds no visits.
The total excludes retained free-buffer
capacity and execution/I/O arenas. There is no static-store admission budget or
eviction policy.

.. doxygenfile:: onnx_core/runtime/kernels/value_store.h
   :project: onnx-light
