"""Thin re-export shim for the NNEF binary tensor IO (C++ implementation)."""

from __future__ import annotations

from onnx_light.onnx_py._onnxpy import nnef as _n

write_nnef_tensor = _n.write_nnef_tensor
read_nnef_tensor = _n.read_nnef_tensor

NNEF_MAGIC: bytes = b"\x4e\xef"
NNEF_VERSION_MAJOR: int = 1
NNEF_VERSION_MINOR: int = 0
NNEF_HEADER_SIZE: int = 128
NNEF_MAX_RANK: int = 8

ITEM_TYPE_FLOAT: int = 0
ITEM_TYPE_QUANT: int = 1
ITEM_TYPE_SIGNED: int = 2
ITEM_TYPE_UNSIGNED: int = 3
ITEM_TYPE_BOOL: int = 4

__all__ = [
    "ITEM_TYPE_BOOL",
    "ITEM_TYPE_FLOAT",
    "ITEM_TYPE_QUANT",
    "ITEM_TYPE_SIGNED",
    "ITEM_TYPE_UNSIGNED",
    "NNEF_HEADER_SIZE",
    "NNEF_MAGIC",
    "NNEF_MAX_RANK",
    "NNEF_VERSION_MAJOR",
    "NNEF_VERSION_MINOR",
    "read_nnef_tensor",
    "write_nnef_tensor",
]
