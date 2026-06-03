"""Binary NNEF tensor file (``*.dat``) reader and writer.

NNEF stores each parameter tensor as a self-contained file made of a
128-byte header followed by the raw item data.  The header layout, as
documented in section *5. Binary Quantization Storage Format* of the
NNEF 1.0 specification, is the following (little-endian):

================  ============  =====================================
Offset (bytes)    Size (bytes)  Field
================  ============  =====================================
0                 2             magic = ``0x4E, 0xEF``
2                 2             major / minor version (1, 0)
4                 4             ``data_length`` – payload bytes
8                 4             ``rank``
12                32            ``extents[8]`` – shape (uint32 each)
44                4             ``bits_per_item``
48                2             ``item_type`` (0=float, 1=quantized,
                                2=signed int, 3=unsigned int, 4=bool)
50                2             ``item_type_data_length`` (0)
52                76            padding (zeros) → total 128 bytes
================  ============  =====================================

This module supports the dense (non-quantized) item types that are
needed to round-trip ONNX initializers: float16, float32, float64,
int8/16/32/64, uint8/16/32/64 and bool.
"""

from __future__ import annotations

import os
import struct
from typing import BinaryIO, Iterable

import numpy as np

#: NNEF binary tensor magic bytes.
NNEF_MAGIC: bytes = b"\x4e\xef"
#: NNEF binary tensor format major version implemented here.
NNEF_VERSION_MAJOR: int = 1
#: NNEF binary tensor format minor version implemented here.
NNEF_VERSION_MINOR: int = 0
#: Size of the fixed NNEF binary tensor header.
NNEF_HEADER_SIZE: int = 128
#: Maximum rank that fits in the fixed-size ``extents`` field.
NNEF_MAX_RANK: int = 8

# Item type codes as defined by the NNEF specification.
ITEM_TYPE_FLOAT: int = 0
ITEM_TYPE_QUANT: int = 1
ITEM_TYPE_SIGNED: int = 2
ITEM_TYPE_UNSIGNED: int = 3
ITEM_TYPE_BOOL: int = 4


def _classify_dtype(dtype: np.dtype) -> tuple[int, int]:
    """Returns ``(item_type, bits_per_item)`` for a numpy dtype.

    Raises:
        ValueError: if the dtype has no NNEF mapping.
    """
    kind = dtype.kind
    bits = dtype.itemsize * 8
    if kind == "f":
        return ITEM_TYPE_FLOAT, bits
    if kind == "i":
        return ITEM_TYPE_SIGNED, bits
    if kind == "u":
        return ITEM_TYPE_UNSIGNED, bits
    if kind == "b":
        return ITEM_TYPE_BOOL, 8
    raise ValueError(f"Unsupported numpy dtype for NNEF: {dtype!r}")


def _dtype_from_item_type(item_type: int, bits: int) -> np.dtype:
    """Inverse of :func:`_classify_dtype` used by the reader."""
    if item_type == ITEM_TYPE_FLOAT:
        if bits == 16:
            return np.dtype(np.float16)
        if bits == 32:
            return np.dtype(np.float32)
        if bits == 64:
            return np.dtype(np.float64)
    elif item_type == ITEM_TYPE_SIGNED:
        return np.dtype(f"int{bits}")
    elif item_type == ITEM_TYPE_UNSIGNED:
        return np.dtype(f"uint{bits}")
    elif item_type == ITEM_TYPE_BOOL:
        return np.dtype(np.bool_)
    raise ValueError(f"Unsupported NNEF item_type={item_type} bits={bits}")


def _pack_header(shape: Iterable[int], item_type: int, bits: int, data_length: int) -> bytes:
    """Packs the 128-byte NNEF tensor header."""
    extents = list(shape)
    rank = len(extents)
    if rank > NNEF_MAX_RANK:
        raise ValueError(f"NNEF binary tensor format supports rank ≤ {NNEF_MAX_RANK}, got {rank}")
    extents = extents + [0] * (NNEF_MAX_RANK - rank)
    header = bytearray(NNEF_HEADER_SIZE)
    header[0:2] = NNEF_MAGIC
    header[2] = NNEF_VERSION_MAJOR
    header[3] = NNEF_VERSION_MINOR
    struct.pack_into("<I", header, 4, data_length)
    struct.pack_into("<I", header, 8, rank)
    struct.pack_into("<8I", header, 12, *extents)
    struct.pack_into("<I", header, 44, bits)
    struct.pack_into("<H", header, 48, item_type)
    struct.pack_into("<H", header, 50, 0)  # item_type_data_length
    return bytes(header)


def write_nnef_tensor(path_or_file: str | os.PathLike[str] | BinaryIO, array: np.ndarray) -> None:
    """Serialises ``array`` to a NNEF ``*.dat`` file.

    Args:
        path_or_file: Destination path (``str``/:class:`os.PathLike`) or
            an already-opened binary file object.
        array: A :class:`numpy.ndarray` whose dtype is one of the
            supported NNEF item types (float16/32/64, signed/unsigned
            integers, bool).

    Raises:
        ValueError: when the dtype or rank cannot be represented by
            the NNEF binary tensor format.
    """
    original_shape = tuple(array.shape)
    array = np.ascontiguousarray(array)
    if array.shape != original_shape:
        # ``np.ascontiguousarray`` promotes 0-d arrays to 1-d; preserve
        # the caller's original rank/shape so it round-trips correctly.
        array = array.reshape(original_shape)
    item_type, bits = _classify_dtype(array.dtype)
    data_length = (
        int(array.size) * (bits // 8) if item_type != ITEM_TYPE_BOOL else int(array.size)
    )
    header = _pack_header(array.shape, item_type, bits, data_length)
    if item_type == ITEM_TYPE_BOOL:
        payload = array.astype(np.uint8).tobytes()
    else:
        payload = array.tobytes()
    if hasattr(path_or_file, "write"):
        path_or_file.write(header)
        path_or_file.write(payload)
    else:
        with open(path_or_file, "wb") as f:
            f.write(header)
            f.write(payload)


def _read_from_stream(stream: BinaryIO) -> np.ndarray:
    header = stream.read(NNEF_HEADER_SIZE)
    if len(header) != NNEF_HEADER_SIZE or header[0:2] != NNEF_MAGIC:
        raise ValueError("Not a NNEF binary tensor file (bad magic)")
    data_length = struct.unpack_from("<I", header, 4)[0]
    rank = struct.unpack_from("<I", header, 8)[0]
    extents = struct.unpack_from("<8I", header, 12)
    bits = struct.unpack_from("<I", header, 44)[0]
    item_type = struct.unpack_from("<H", header, 48)[0]
    dtype = _dtype_from_item_type(item_type, bits)
    shape = tuple(int(x) for x in extents[:rank])
    payload = stream.read(data_length)
    if len(payload) != data_length:
        raise ValueError(
            f"Truncated NNEF tensor: expected {data_length} bytes, got {len(payload)}"
        )
    if item_type == ITEM_TYPE_BOOL:
        array = np.frombuffer(payload, dtype=np.uint8).astype(np.bool_)
    else:
        array = np.frombuffer(payload, dtype=dtype)
    if shape:
        array = array.reshape(shape)
    else:
        array = array.reshape(())
    return array.copy()


def read_nnef_tensor(path_or_file: str | os.PathLike[str] | BinaryIO) -> np.ndarray:
    """Reads a NNEF ``*.dat`` file and returns a :class:`numpy.ndarray`.

    This is mainly used by the tests to round-trip tensors written by
    :func:`write_nnef_tensor`.
    """
    if hasattr(path_or_file, "read"):
        return _read_from_stream(path_or_file)
    with open(path_or_file, "rb") as f:
        return _read_from_stream(f)
