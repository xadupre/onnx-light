.. _l-next-steps-mutable-cache:

Mutable execution cache
=======================

:Date: 2026-08

**discussion**

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

The cache remains an ordinary tensor or sequence. Its capacity is part of its
shape, while its valid length and update position are explicit scalar inputs.
Mutability and storage aliasing are execution contracts, not new value types.

``ShapesContext``
+++++++++++++++++

No ``SymCache`` class is needed. ``ShapesContext`` handles the cache with its
existing ``SymTensor`` or sequence representation and infers that
``updated_cache`` has the same type and shape as ``cache``. When the relevant
dimensions are known, it also checks
``position + update_length <= capacity``.

``ValueInfoProto.access`` records whether the graph input is mutable.
``NodeProto.output_alias`` records the relation between operator ports, while
the output's ``ValueInfoProto.alias_of`` records the same relation by value
name. The execution plan and runtime track the concrete storage identity and
verify pointer equality. Subgraphs and local functions propagate the same type
and alias annotations.

Shape inference maintains alias equivalence classes over value names. For each
alias it:

1. resolves the node port indices to input and output names;
2. verifies that ``ValueInfoProto.alias_of`` names the same input;
3. merges the input and output type and shape constraints;
4. rejects incompatible element types, ranks, dimensions, or alias cycles;
5. records the common alias root for memory planning.

Type and shape information learned for either name is therefore visible
through the other. Control-flow branches may merge an aliased output only when
they resolve it to the same alias root. ``ShapesContext`` stores names and
constraints only; concrete pointer identity remains a runtime check.

Required alias
++++++++++++++

The execution plan needs a mandatory alias distinct from opportunistic
``inplace_reuse``:

.. code-block:: text

    updated_cache.alias_of = "cache"

``alias_of`` means:

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

The port-level and value-level contracts are serialized as follows:

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
        string alias_of = <N+1>;  // empty when the value owns distinct storage
    }

``output_alias`` identifies the input and output ports of the producing node.
``alias_of`` names the value whose storage must be shared and exposes the
relation at graph or function boundaries. When both annotations describe the
same output, a checker requires them to identify the same input.
``READ_WRITE`` is valid for a graph input and tells the caller that execution
may modify its storage.

No ``CacheProto`` is needed. Dense caches remain ``TensorProto`` values and
quantized pages are ``QuantizedTensorProto`` values. Capacity and valid length
are described by shapes or explicit scalar inputs.

Alternative alias designs
+++++++++++++++++++++++++

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Design
     - Advantage
     - Limitation
   * - ``ValueInfoProto.alias_of``
     - Relates named values directly and works at graph boundaries.
     - Requires ``ValueInfoProto`` for every aliased output.
   * - ``NodeProto.output_alias``
     - Describes the operator contract using stable port indices.
     - Does not alone expose the alias at a graph boundary.
   * - Operator-schema alias rule
     - Declared once for ``CacheUpdate``.
     - Requires the runtime schema; less suitable for dynamic rules.
   * - ``FunctionProto`` parameter alias
     - Preserves aliases through local functions.
     - Does not describe primitive custom nodes.
   * - Mutable reference type
     - Naturally represents persistent state.
     - Introduces reference semantics into ONNX types.
   * - Graph state slot
     - Hides cache threading between graph calls.
     - Adds state, concurrency, and checkpoint semantics.
   * - Metadata property
     - Requires no proto change and is easy to prototype.
     - Weak contract that generic tools may ignore.

The recommended design keeps ``NodeProto.output_alias`` for the local operator
contract and ``ValueInfoProto.alias_of`` for the named graph contract.
Metadata is suitable only for experimentation.

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
            alias_of: "cache"

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
    updated_cache.alias_of = "cache"

``ShapesContext`` keeps only the common logical page type and shape. The
runtime owns the page storage identities. The kernel dispatches from
``quantized_type`` and quantizes the new values directly into the selected
page.

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

Existing implementations
++++++++++++++++++++++++

ONNX Runtime GenAI
^^^^^^^^^^^^^^^^^^

ONNX Runtime GenAI exposes the cache as ``past_key_values.*`` inputs and
``present.*`` outputs. In the basic mode, each ``present`` becomes the next
iteration's ``past``.

Buffer sharing is requested in ``genai_config.json``:

.. code-block:: json

    {
      "search": {
        "past_present_share_buffer": true,
        "num_beams": 1,
        "max_length": 4096
      }
    }

GenAI then allocates one capacity-sized ``OrtValue`` per K/V layer and binds
the same object to both names:

.. code-block:: text

    past_key_values.0.key ──┐
                            ├── OrtValue B
    present.0.key ──────────┘

The compatible attention operator writes new values at the current position.
``DefaultKeyValueCache::Update`` returns immediately because neither the
pointer nor the allocation changes. The ONNX model itself contains no alias
annotation; GenAI creates the alias while binding session inputs and outputs.

The option is enabled only when requested and when ``num_beams == 1``, except
for the specialized Whisper case. Graph capture requires shared buffers.

It also supports windowed caches, model-managed state, and paged-attention
metadata such as ``block_table`` and ``past_sequence_lengths``. Quantized
caches use a cache-wide or layer-wide representation rather than a different
type for every page.

Its ``PagedKeyValueCache`` is logically, not physically, paged. For each layer
it preallocates two contiguous tensors:

.. code-block:: text

    key_cache:   [num_blocks, block_size, num_kv_heads, head_size]
    value_cache: [num_blocks, block_size, num_kv_heads, head_size]

The ``block_table`` maps each request's logical pages to slices on the
``num_blocks`` axis. Blocks can be assigned and released independently, but
the backing tensors never grow after allocation. Once the free block pool is
empty, no additional cache capacity is available without creating a larger
cache and copying the old content.

All blocks of one cache tensor have the same element type, layout, and
physical size. Quantization parameters may differ by layer, but not by page.
Independent page allocations and heterogeneous page quantization are not
supported. With one active request, continuous batching provides no batching
benefit; paging still avoids contiguous cache growth, but it does not reduce
the preallocated physical pool.

See `kv_cache.cpp
<https://github.com/microsoft/onnxruntime-genai/blob/ff2009e71bd625d3c5ed7a6cbb410cf2e2dbaf48/src/models/kv_cache.cpp#L531-L564>`_,
`kv_cache.h
<https://github.com/microsoft/onnxruntime-genai/blob/ff2009e71bd625d3c5ed7a6cbb410cf2e2dbaf48/src/models/kv_cache.h>`_,
`paged_key_value_cache.h
<https://github.com/microsoft/onnxruntime-genai/blob/ff2009e71bd625d3c5ed7a6cbb410cf2e2dbaf48/src/engine/paged_key_value_cache.h>`_,
the `activation rule
<https://github.com/microsoft/onnxruntime-genai/blob/ff2009e71bd625d3c5ed7a6cbb410cf2e2dbaf48/src/generators.cpp#L430-L435>`_,
and the `model builder
<https://github.com/microsoft/onnxruntime-genai/blob/ff2009e71bd625d3c5ed7a6cbb410cf2e2dbaf48/src/python/py/models/builders/base.py>`_.

llama.cpp
^^^^^^^^^

llama.cpp keeps the KV-cache in runtime-owned ``llama_kv_cache`` state rather
than model inputs and outputs. It allocates K/V storage per layer, gives the
compute graph views over the destination ranges, and writes new values
directly through ``cpy_k`` and ``cpy_v``. Sequence operations primarily update
cell metadata; defragmentation moves only required ranges.

The capacity ``kv_size`` is fixed when the context is created. Each layer owns
K/V tensors covering that capacity, while ``llama_kv_cells`` maps tokens and
sequences to slots. A sequence may therefore occupy non-contiguous cells, but
the underlying tensors do not grow. ``find_slot`` reuses free cells;
context shifting, eviction, and defragmentation recover space. If no suitable
slot is available, cache preparation fails rather than extending the
allocation.

``type_k`` and ``type_v`` are selected for the cache as a whole. They may
differ from each other, but not from cell to cell. Specialized block caches
exist for some architectures, but there is no generic page table with
independently allocated or heterogeneously quantized pages.

See `llama-kv-cache.h
<https://github.com/ggml-org/llama.cpp/blob/7ba604f1cb61cd14898138e9abc0b4ff2601f180/src/llama-kv-cache.h>`_
and `llama-kv-cache.cpp
<https://github.com/ggml-org/llama.cpp/blob/7ba604f1cb61cd14898138e9abc0b4ff2601f180/src/llama-kv-cache.cpp>`_.

Design consequence
^^^^^^^^^^^^^^^^^^

``output_alias`` and ``alias_of`` serialize the shared-buffer optimization
used by ONNX Runtime GenAI instead of leaving it to runtime configuration. A
mutable state handle would be closer to llama.cpp but would move cache
semantics outside the graph. ``Sequence<QuantizedTensorProto>`` additionally
permits each page to select a different ``quantized_type``.

Memory planning
+++++++++++++++

Peak-memory analysis counts an in-place cache update as zero additional cache
bytes. It includes only temporary kernel workspace and newly allocated pages.
The execution plan keeps the cache alive across graph invocations and excludes
it from ordinary release and reuse candidates.

Implementation order
++++++++++++++++++++

1. Add ``NodeProto.output_alias``, ``ValueInfoProto.alias_of``, and
   execution-plan alias metadata.
2. Add mutable caller bindings and writable kernel views.
3. Extend shape inference with alias propagation and cache-capacity checks.
4. Support dense and heterogeneous quantized paged-cache updates.
5. Test pointer identity, capacity errors, concurrency rejection, and peak
   memory without a second cache allocation.
