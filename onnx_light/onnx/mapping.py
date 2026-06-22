"""Compatibility shim for :mod:`onnx.mapping`.

Provides ``TENSOR_TYPE_TO_NP_TYPE`` for legacy code that imports it directly.
Prefer ``onnx_light.onnx.helper.tensor_dtype_to_np_dtype`` instead.
"""

from __future__ import annotations

import contextlib

import numpy

from . import TensorProto
from .helper import tensor_dtype_to_np_dtype

# Build the mapping dict from all known elem_type integer values.
TENSOR_TYPE_TO_NP_TYPE: dict[int, numpy.dtype] = {}
for _field in dir(TensorProto):
    _val = getattr(TensorProto, _field)
    if not isinstance(_val, int) or _field.startswith("_") or _val == 0:
        continue
    with contextlib.suppress(KeyError, ValueError, TypeError):
        TENSOR_TYPE_TO_NP_TYPE[_val] = tensor_dtype_to_np_dtype(_val)
