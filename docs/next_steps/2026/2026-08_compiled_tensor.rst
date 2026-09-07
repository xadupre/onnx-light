.. _l-next-steps-compiled-tensor:

Prepared values in EncodedValueProto
====================================

:Date: 2026-08

**consolidated design reference**

.. note::

    The current design is
    :ref:`l-next-steps-prepared-values-and-persistent-state`. In particular,
    preparation keys cover all source operands and their semantics. Reuse
    the implemented prepared object store and disk cache. The proposed
    typed serialization below is not implemented: it uses
    ``EncodedValueProto`` with optional preparation metadata, not a separate
    ``CompiledTensorProto`` or a wrapper around a ``StructProto``.

Motivation
++++++++++

A runtime may transform an initializer into a device-specific packed
representation. Persisting that representation avoids repeating an expensive
prepacking step when the same model is loaded again.

A compiled tensor is a cache, not a new tensor semantics. The graph continues
to reference the original initializer, which remains the portable fallback.
The cached bytes use :ref:`l-next-steps-custom-types` and are ignored when the
current runtime or device is incompatible.

Value and preparation metadata
+++++++++++++++++++++++++++++++++++++++++++++++++

The prepared payload is an ``EncodedValueProto``. Preparation metadata
records its provenance and compatibility without owning another payload.
The following descriptive sketch does not freeze wire field numbers or
metadata field names:

.. code-block:: text

    EncodedValueProto {
        struct_type: { type_ref: 3001 }
        storage_shape: [tile_count]
        logical_type: FLOAT[K, N]
        raw_data: ...
        preparation: {
            source_lineage: ...       // ordered operands and their semantics
            source_lineage_digest: ...
            digest_algorithm: "blake3"
            recipe: ...               // operator, packing and tuning choices
            device: 1                 // index into ModelProto.devices
        }
    }

The structured branch resolves a stable ``type_ref`` ID or an inline
concrete ``struct_type``. An unconstrained static type is not valid for a
payload. A built-in layout may instead use the same value container without
a structured declaration.

Source lineage resolves every operand in its graph scope, including scales,
zero points, bias and other inputs when they contribute to preparation.
Its digest prevents stale prepared data from being used after any dependency
changes. The digest algorithm is explicit. The key covers canonical source
content, types, dimensions, type constants and interpretation, together with
the recipe and compatibility metadata; it is not merely a digest of one
weight buffer. Tensor names and external-data locations are not sufficient
content identities.

DeviceProto
+++++++++++

The device descriptor identifies compatibility, not only a device ordinal:

.. code-block:: text

    message DeviceProto {
        string type = 1;              // "cpu", "cuda", "rocm", ...
        optional int32 index = 2;     // exact ordinal only when required
        string architecture = 3;      // "x86_64-avx2", "sm_80", ...
        string runtime = 4;           // producer/runtime domain
        string runtime_version = 5;   // compatible runtime version
        repeated StringStringEntryProto metadata_props = 6;
    }

``architecture`` records the instruction-set or accelerator compatibility
needed by the packed representation. ``runtime`` and ``runtime_version``
identify the implementation ABI that interprets the type. Additional
compatibility keys may be stored in ``metadata_props``.

ModelProto extension
++++++++++++++++++++

An illustrative model extension stores the same value messages directly:

.. code-block:: text

    message ModelProto {
        ...
        repeated StructTypeProto struct_types = <N>;
        repeated DeviceProto devices = <N+1>;
        repeated EncodedValueProto prepared_values = <N+2>;
    }

Several prepared entries may use the same source lineage for different
architectures, runtimes, or packing strategies. Stable structured type IDs,
resolved definitions and compatibility metadata distinguish physical formats;
display names alone do not. The collection is a cache attachment, not a new
value category or a replacement for portable graph initializers.

Loading rules
+++++++++++++

A runtime uses a compiled entry only when all of the following hold:

* every source operand resolves unambiguously in its scope;
* the source-lineage digest and preparation recipe match current dependencies;
* ``device`` is an in-range model-level index;
* device type, architecture, runtime, version, and required metadata are
  compatible;
* the encoded value, selected layout and storage shape pass structural and
  payload-size validation;
* the runtime recognizes that physical type and compiled-format version.

If a compatibility condition or digest comparison fails, the runtime treats
the entry as a cache miss and rebuilds it from the portable source operands.
Invalid compiled data must never change graph results or make an otherwise
valid portable model unloadable. Malformed indices, payloads, or digest
declarations are checker errors; ordinary incompatibility or a stale digest
is only a cache miss.

Quantized and tiled tensors
+++++++++++++++++++++++++++

No dependency on ``QuantizedTensorProto`` is needed. A quantized, tiled, or
otherwise packed cache entry is represented by the same
``EncodedValueProto`` mechanism:

.. code-block:: text

    EncodedValueProto {
        struct_type: { type_ref: 3002 } // stable packed CUDA type ID
        storage_shape: [tile_count]
        logical_type: FLOAT[K, N]
        raw_data: ...
        preparation: {
            source_lineage: ...        // includes weight and all other operands
            source_lineage_digest: ...
            digest_algorithm: "blake3"
            recipe: ...
            device: 1                  // e.g. CUDA sm_80 + runtime ABI
        }
    }

The referenced structured type describes the complete byte layout. Its
optional decoder describes portable interpretation for inspectable formats.
A runtime-specific prepack may omit the decoder when only the named runtime
can consume it; the original initializer still guarantees portability.

Validation
++++++++++

A checker validates:

* unique device descriptors and valid device indices;
* unique complete preparation keys, including lineage, recipe, layout and device;
* source operand existence and unambiguous scope;
* non-empty digest and algorithm fields;
* exact layout resolution, stable structured IDs and payload size;
* absence of unconstrained static structured types;
* metadata keys are unique.

Digest comparison and runtime compatibility may be deferred until load time,
but structural errors are rejected independently of hardware availability.

Relationship to other proposals
+++++++++++++++++++++++++++++++

Preparation reuses the physical representation in ``EncodedValueProto``.
The specialized hierarchy in
:ref:`l-next-steps-quantization` may remain a format catalogue, but it is not a
storage dependency. Proto inheritance and wrapper containers are unnecessary:
the preparation recipe is optional metadata on the same encoded value.
