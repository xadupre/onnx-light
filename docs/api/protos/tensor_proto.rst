===========
TensorProto
===========

.. autoclass:: onnx_light.onnx.TensorProto
    :members:

DLPack import
-------------

``TensorProto.from_dlpack(array, name="")`` constructs a tensor without allocating
or copying its payload. ``array`` must implement both ``__dlpack__`` and
``__dlpack_device__``. For example::

    import numpy
    from onnx_light.onnx import TensorProto

    array = numpy.arange(12, dtype=numpy.float32).reshape(3, 4)
    tensor = TensorProto.from_dlpack(array, name="weights")
    del array
    serialized = tensor.SerializeToString()

The explicit import API does not change ``numpy_helper.from_array``: that
compatibility path still copies. Reading the Python ``tensor.raw_data`` property
also returns a bytes copy; use ``numpy.from_dlpack(tensor)`` to inspect shared
storage without copying.

Import accepts only CPU device ``(1, 0)``, compact row-major storage, and the
whole-byte dtypes listed under export below. Dimensions and byte counts are
validated, including overflow. Singleton strides are ignored, as are strides of
empty tensors. Nonzero byte offsets are supported when element-aligned, with
pointer overflow checks. The producer must supply a valid allocation covering
the described storage; DLPack does not convey its allocation size.

DLPack describes native-endian storage. Producers must reject non-native-endian
arrays (as NumPy does); multi-byte imports on big-endian hosts are rejected
because ONNX requires little-endian bytes. Unsupported layouts, devices, dtypes,
misalignment, copied versioned exports, and unknown flags fail rather than being
converted. Read-only versioned exports are accepted.

The consumer requests ``stream=None``, ``max_version=(1, 0)``, and ``copy=False``.
For legacy producers that reject negotiation keywords, it retries with only
``stream=None``. It never retries a synchronization failure. CPU data must be
ready for synchronous access when export returns: no accelerator streams,
events, or asynchronous synchronization are supported.

Both legacy ``dltensor`` and versioned ``dltensor_versioned`` capsules are consumed
exactly once. Metadata is copied, while ``raw_data`` borrows the allocation.
The managed DLPack deleter runs under the Python GIL after the last shared owner
is released, including on validation failure after capsule consumption. This
also applies to zero-sized tensors with null data pointers. The original Python
array may be deleted immediately; the producer's managed tensor retains storage.

``copy.copy(tensor)``, ``TensorProto.CopyFrom``, repeated tensor insertion, and
GraphBuilder's ``make_initializer`` / ``make_initializer_move`` preserve the
borrowed owner, as do the builder's returned graph and model. ``CopyFrom`` still
copies owned payloads. Serialization creates independent bytes; reparsing those
bytes does not retain the DLPack owner. Treat shared storage as read-only while
the tensor, builder, or model is in use, including through the original array.

DLPack export
-------------

``numpy.from_dlpack(tensor)`` creates a read-only, zero-copy CPU view of a
``TensorProto`` with dense, row-major ``raw_data``. ``__dlpack_device__()``
returns ``(1, 0)`` (CPU). Export allocates only bookkeeping and shape metadata,
never a payload buffer, and does not call ``numpy_helper.to_array``.

Supported types are signed and unsigned 8/16/32/64-bit integers, BOOL,
FLOAT16/FLOAT/DOUBLE, COMPLEX64/COMPLEX128, BFLOAT16, and all five ONNX FLOAT8
formats. Each uses its exact DLPack dtype, including format-specific FLOAT8
codes; consumers such as NumPy may not support every exported dtype.
STRING, UNDEFINED, packed sub-byte types, segmented tensors, typed-field-only
payloads, and external data not yet loaded into ``raw_data`` are rejected.
Dimensions must be non-negative, the rank at most 128, and shape/byte counts
must fit the platform's size and signed 64-bit limits. The payload must match
the shape exactly. Empty tensors require an explicitly present empty
``raw_data`` field.

ONNX raw bytes are little-endian. On big-endian hosts, multi-byte types raise
``BufferError`` instead of being byte-swapped. Nonempty buffers must be aligned
to the element width (the component width for complex types); unaligned
buffers raise ``BufferError`` instead of being copied.

Each export returns a fresh, single-consumption legacy ``dltensor`` capsule.
``stream`` must be ``None``; ``dl_device`` must be ``None`` or ``(1, 0)``.
``copy=None`` and ``copy=False`` are accepted; ``copy=True`` or another device
raises ``BufferError``. ``max_version`` is accepted, but export still uses the
legacy capsule ABI. Consumers must not write into exported storage, which may
be backed by a read-only memory mapping.

The capsule retains the source Python proto and a separate reference to its
raw-data owner token until the consumer releases the tensor (or an unused
capsule is destroyed). Deleting the source variable is safe; copied borrowed
protos share the original owner. Do not clear, resize, replace, or reparse
**owned** source storage while a view exists, including through a containing
model. Borrowed storage with no owner token remains the caller's lifetime
responsibility. Shape metadata is snapshotted at export.

Explicit destructive transfer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``tensor.release_dlpack()`` transfers the existing allocation or shared
borrowed-storage owner to an independent, single-consumption legacy
``dltensor`` capsule. Unlike ``__dlpack__``, it does not retain the source.
The native equivalent is ``ReleaseDLPack(TensorProto&)`` in ``onnx_dlpack.h``.

.. warning::

   **DESTRUCTIVE:** ``release_dlpack()`` immediately removes the source
   ``raw_data``: its pointer becomes null, size becomes zero, and
   ``HasField("raw_data")`` becomes false. The tensor and any initializer/model
   containing it cannot use that payload until it is reassigned.
   All other proto fields (including name, dimensions, dtype, and external
   metadata) remain unchanged.

The descriptor snapshots dimensions and dtype, and its storage survives
source destruction or reuse. The deleter never accesses the source.
User-provided storage callbacks must likewise be independent of the source.
Owned and aligned-owned allocations transfer without copying; borrowed data
requires a lifetime owner and shares exactly-once cleanup with other owners.
Validation and allocation failures leave the source unchanged, including
failure to create the Python capsule.

This method applies the same dtype, shape, endian, alignment, and explicit
empty-payload restrictions as ``__dlpack__``. Consumers must not write to the
buffer. The return value is a capsule, not an array or protocol object;
consumers accepting capsules directly can consume it once.

.. warning::

   Transferring owned storage while earlier ``__dlpack__`` views or capsules
   are still alive raises an exception without modifying the source. Release
   those consumers first. Borrowed storage with a shared lifetime owner can
   be transferred while views remain alive; each view retains that owner.

Binary-size measurement
~~~~~~~~~~~~~~~~~~~~~~~

The following historical measurement covers the original non-destructive
Python-only export, **before** native destructive transfer was added.
The current implementation shares validation and dtype mapping through
``lib_onnx_proto`` and vendors the header-only DLPack C ABI; it adds no DLPack
runtime library. These size deltas do not measure the native transfer API.
A Linux x86-64
Release build with GCC 13.3.0, Python 3.13.15 and nanobind 3.1.0, using
``python setup.py build_ext --inplace --no-kernels``, measured as follows
(bytes, after ``strip --strip-unneeded``; ``.text`` from ``size -A``):

.. list-table::
   :header-rows: 1

   * - Artifact / section
     - Before
     - After
     - Delta
   * - ``_onnxpyprotoop.abi3.so`` stripped file
     - 2,404,304
     - 2,412,528
     - +8,224 (+0.34%)
   * - ``_onnxpyprotoop.abi3.so`` ``.text``
     - 1,517,797
     - 1,523,861
     - +6,064 (+0.40%)
   * - ``liblib_onnx_proto.so`` stripped file
     - 1,371,656
     - 1,371,656
     - 0
   * - ``liblib_onnx_proto.so`` ``.text``
     - 948,058
     - 948,058
     - 0

In that historical measurement the stripped proto library was byte-identical
before and after, and the extension's ``DT_NEEDED`` entries were unchanged.

Native transfer build budget
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With native transfer and the out-of-line export guard, the same Linux x86-64
GCC 13.3.0 Release build initially exceeded the CI ``.text`` limit. Compiling
``onnx_dlpack.cc`` for size and sharing validation-error construction reduces
the footprint without changing validation conditions or exception types:

.. list-table::
   :header-rows: 1

   * - ``liblib_onnx_proto.so`` metric
     - Before size fix
     - After size fix
     - CI maximum
   * - Stripped file bytes
     - 1,383,976
     - 1,379,880
     - 1,390,552 (unchanged)
   * - ``.text`` bytes
     - 956,186
     - 952,826
     - 953,706 (unchanged)
   * - Defined dynamic symbols
     - 793
     - 793
     - 793 (previously 792)

The one-symbol allowance accounts for ``ByteSpan::acquire_export_guard()``
moving out of line. No public API is hidden to satisfy the count. The shared
diagnostic helper keeps the ``[onnx-light]`` prefix and descriptive message,
but omits the redundant stringified C++ condition.

Measurements use ``strip --strip-unneeded`` and
``.github/scripts/report_proto_binary_size.py`` on the shared proto library,
configured with ``CMAKE_BUILD_TYPE=Release``, ``ONNX_LIGHT_BUILD_PYTHON=ON``
and ``ONNX_LIGHT_BUILD_KERNELS=OFF``. The baseline reproduces the failing
full-build CI measurements exactly. The dependency allowlist is unchanged:
``libcrypto.so.3``, ``libstdc++.so.6``, ``libgcc_s.so.1``, ``libc.so.6``,
and ``ld-linux-x86-64.so.2``.
