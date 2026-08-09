.. _l-next-steps-mutable-cache:

Mutable execution cache
=======================

:Date: 2026-08

Objective
+++++++++

A KV-cache may occupy most of the available memory. Updating it must not
allocate an output cache of the same size. The runtime needs a guaranteed
in-place update, not the current best-effort buffer-reuse heuristic.

Value model
+++++++++++

The cache is a mutable graph input owned by the caller. The graph keeps SSA
names, but two names may identify the same storage:

.. code-block:: text

    updated_cache = CacheUpdate(cache, values, position)

``updated_cache`` must alias ``cache``. The operator changes the valid cache
content but does not allocate another full buffer.

The cache descriptor separates:

* capacity: allocated shape or number of pages;
* valid length: portion currently containing values;
* physical type: dense tensor or ``QuantizedTensorProto`` page;
* mutability and storage identity.

``ShapesContext``
+++++++++++++++++

``ShapesContext`` keeps this information because it already owns value shapes,
types, and symbolic dimensions. A cache descriptor may wrap a tensor, a
quantized tensor, or a sequence of quantized pages:

.. code-block:: cpp

    class SymCache {
    public:
      const SymValue &Value() const;
      const SymDim &Capacity() const;
      const SymDim &ValidLength() const;
      CacheStorageId StorageId() const;
      bool IsMutable() const;
    };

Shape inference for ``CacheUpdate``:

1. checks that the input is mutable;
2. verifies ``position + update_length <= capacity``;
3. returns a descriptor with the same storage identity;
4. updates the valid length without changing capacity.

Subgraphs inherit captured caches. Local functions receive and return them
explicitly. Merging control-flow branches is valid only when both branches
preserve the same storage identity and compatible valid lengths.

Required alias
++++++++++++++

The execution plan needs a mandatory alias distinct from opportunistic
``inplace_reuse``:

.. code-block:: text

    output 0 MUST_ALIAS input 0

``MUST_ALIAS`` means:

* no output allocation is planned;
* the kernel receives a writable view of the input storage;
* pointer identity is checked by the runtime;
* execution fails if the caller supplied immutable or shared storage;
* there is no silent copy fallback.

The existing last-use rule does not apply: the cache is persistent state and
remains live after the node. Safety comes from explicit mutable ownership, not
from the input becoming dead.

Proto additions
+++++++++++++++

Two small additions make the contract serializable:

.. code-block:: text

    message ValueAliasProto {
        enum Kind {
            UNDEFINED = 0;
            MUST_ALIAS = 1;
        }
        int32 output_index = 1;
        int32 input_index = 2;
        Kind kind = 3;
    }

    message NodeProto {
        ...
        repeated ValueAliasProto output_alias = <N>;
    }

    message ValueInfoProto {
        enum Access {
            READ_ONLY = 0;
            READ_WRITE = 1;
        }
        ...
        Access access = <N>;
    }

``output_alias`` declares the storage relation independently of the operator
implementation. ``READ_WRITE`` is valid for a graph input and tells the caller
that execution may modify its storage.

No ``CacheProto`` is needed. Dense caches remain ``TensorProto`` values and
quantized pages are ``QuantizedTensorProto`` values. Capacity and valid length
are described by shapes or explicit scalar inputs.

Model example
+++++++++++++

The model exposes the cache as a mutable input and returns a different SSA
name backed by the same storage:

.. code-block:: text

    graph input:
        cache: FLOAT16[2, batch, heads, max_length, head_size]
            access: READ_WRITE
        values: FLOAT16[2, batch, heads, new_length, head_size]
        position: INT64[]

    node:
        op_type: "CacheUpdate"
        input: ["cache", "values", "position"]
        output: ["updated_cache"]
        output_alias: {
            output_index: 0
            input_index: 0
            kind: MUST_ALIAS
        }

    graph output:
        updated_cache: FLOAT16[2, batch, heads, max_length, head_size]

Before execution, the caller allocates ``cache`` for ``max_length`` and binds
it as writable storage. The runtime binds ``updated_cache`` to the same
address, and ``CacheUpdate`` writes only the range starting at ``position``.
On the next invocation, that same storage is bound again as ``cache``.

``position + new_length`` is checked against ``max_length``. The execution
plan allocates only ordinary outputs and kernel workspace; it never allocates
a second cache.

Heterogeneous paged cache
+++++++++++++++++++++++++

A paged cache is a sequence or map of ``QuantizedTensorProto`` pages. The
container accepts several quantization types, while every page selects one
exact type:

.. code-block:: text

    cache: Sequence<QuantizedTensor {
        allowed_quantized_type: []  // any registered quantization
        elem_type: FLOAT16
        shape: [2, heads, page_size, head_size]
    }>
        access: READ_WRITE

    pages: [
        QuantizedTensorProto {
            quantized_type: INT4_PAGE
            dims: [2, heads, page_size, head_size]
            raw_data: ...
        },
        QuantizedTensorProto {
            quantized_type: FP8_PAGE
            dims: [2, heads, page_size, head_size]
            raw_data: ...
        },
        QuantizedTensorProto {
            quantized_type: INT2_PAGE
            dims: [2, heads, page_size, head_size]
            raw_data: ...
        }
    ]

All page types must decode to the same logical element type and page shape,
but their physical layouts and byte sizes may differ.

The update node aliases the page table:

.. code-block:: text

    updated_cache = PagedCacheUpdate(cache, page_id, values, position)
    updated_cache MUST_ALIAS cache

``ShapesContext`` keeps one ``SymQuantizedTensor`` and one storage identity
per page. The kernel dispatches from ``quantized_type`` and quantizes the new
values directly into that page.

Updating an existing page allocates nothing. Creating a page allocates only
that page. Changing a page to a quantization type requiring a different byte
size replaces only that page and updates the aliased page table; it never
copies the complete cache.

``ModelProto.quantizations`` contains the referenced quantization
declarations. Different physical page formats remain valid because they all
decode to the container's common logical element type and shape.

Runtime contract
++++++++++++++++

The caller opts into mutation when binding the cache. One writable cache
binding cannot be used concurrently by multiple sessions. Read-only model
initializers are never accepted as mutable caches.

The kernel writes only inside the declared capacity. An overflow is an error;
the runtime must not grow the cache by allocating and copying a full
replacement.

For a paged cache, only a missing page may be allocated. Existing pages are
updated in place, and page-table growth must not duplicate page payloads.
Each quantized page keeps its own ``QuantizedTensorProto.quantized_type``.

Memory planning
+++++++++++++++

Peak-memory analysis counts an in-place cache update as zero additional cache
bytes. It includes only temporary kernel workspace and newly allocated pages.
The execution plan keeps the cache alive across graph invocations and excludes
it from ordinary release and reuse candidates.

Implementation order
++++++++++++++++++++

1. Add ``SymCache`` and storage identity to ``ShapesContext``.
2. Add ``MUST_ALIAS`` to schema and execution-plan metadata.
3. Add mutable caller bindings and writable kernel views.
4. Support dense and heterogeneous quantized paged-cache updates.
5. Test pointer identity, capacity errors, concurrency rejection, and peak
   memory without a second cache allocation.
