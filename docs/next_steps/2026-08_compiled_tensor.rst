.. _l-next-steps-compiled-tensor:

CompiledTensorProto
===================

:Date: 2026-08

**discussion**

Motivation
++++++++++

A runtime may transform an initializer into a device-specific packed
representation. Persisting that representation avoids repeating an expensive
prepacking step when the same model is loaded again.

A compiled tensor is a cache, not a new tensor semantics. The graph continues
to reference the original initializer, which remains the portable fallback.
The cached bytes use :ref:`l-next-steps-custom-types` and are ignored when the
current runtime or device is incompatible.

Stable contract
+++++++++++++++

.. code-block:: text

    message CompiledTensorProto {
        string source_name = 1;        // original graph initializer
        StructProto value = 2;   // prepacked physical representation
        int32 device = 3;              // index into ModelProto.devices
        bytes source_digest = 4;       // digest of the canonical source tensor
        string digest_algorithm = 5;   // for example "blake3"
        repeated StringStringEntryProto metadata_props = 6;
    }

``value`` is always an exact structured value: it uses either a non-negative
model-level ``type`` index or an inline concrete ``struct_type``. An
unconstrained static type is not valid here.

``source_name`` refers to one initializer in ``ModelProto.graph``. The first
version does not address initializers owned by nested subgraphs.
``source_digest`` prevents stale compiled data from being used after that
initializer changes. The digest algorithm is explicit so the cache format
does not depend on one hard-coded hash implementation. The digest covers the
element type, dimensions, and canonical tensor content, but not the tensor
name, external-data location, or storage method.

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

.. code-block:: text

    message ModelProto {
        ...
        repeated StructTypeProto struct_types = <N>;
        repeated DeviceProto devices = <N+1>;
        repeated CompiledTensorProto compiled_tensors = <N+2>;
    }

Several compiled entries may reference the same ``source_name`` for different
architectures, runtimes, or packing strategies. Their
``StructTypeProto.name`` and metadata distinguish the physical formats.

Loading rules
+++++++++++++

A runtime uses a compiled entry only when all of the following hold:

* ``source_name`` resolves to exactly one initializer;
* ``source_digest`` matches the canonical initializer content;
* ``device`` is an in-range model-level index;
* device type, architecture, runtime, version, and required metadata are
  compatible;
* ``value`` and its concrete ``StructTypeProto`` pass structural and
  payload-size validation;
* the runtime recognizes that physical type and compiled-format version.

If a compatibility condition or digest comparison fails, the runtime treats
the entry as a cache miss and rebuilds it from the original initializer.
Invalid compiled data must never change graph results or make an otherwise
valid portable model unloadable. Malformed indices, payloads, or digest
declarations are checker errors; ordinary incompatibility or a stale digest
is only a cache miss.

Quantized and tiled tensors
+++++++++++++++++++++++++++

No dependency on ``QuantizedTensorProto`` is needed. A quantized, tiled, or
otherwise packed cache entry is represented by the same
``StructProto`` mechanism:

.. code-block:: text

    CompiledTensorProto {
        source_name: "decoder.layers.0.attn.weight"
        value: StructProto {
            type: 3                    // model-level packed CUDA type
            raw_data: ...
        }
        device: 1                      // e.g. CUDA sm_80 + runtime ABI
        source_digest: ...
        digest_algorithm: "blake3"
    }

The referenced structured type describes the complete byte layout. Its
optional decoder describes portable interpretation for inspectable formats.
A runtime-specific prepack may omit the decoder when only the named runtime
can consume it; the original initializer still guarantees portability.

Validation
++++++++++

A checker validates:

* unique device descriptors and valid device indices;
* unique ``(source_name, device, physical type)`` cache keys;
* source initializer existence;
* non-empty digest and algorithm fields;
* exact structured type resolution and payload size;
* absence of unconstrained static structured types;
* metadata keys are unique.

Digest comparison and runtime compatibility may be deferred until load time,
but structural errors are rejected independently of hardware availability.

Relationship to other proposals
+++++++++++++++++++++++++++++++

``CompiledTensorProto`` depends only on the stable physical representation in
``StructProto``. The specialized hierarchy in
:ref:`l-next-steps-quantization` may remain a format catalogue, but it is not a
storage dependency. Proto inheritance is likewise unnecessary because the
compiled value is composed from a ``StructProto`` rather than derived
from it.
