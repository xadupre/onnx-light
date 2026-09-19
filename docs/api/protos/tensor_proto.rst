===========
TensorProto
===========

.. autoclass:: onnx_light.onnx.TensorProto
    :members:

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
