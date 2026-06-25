"""Public helpers that complement :mod:`onnx_light.onnx`."""

from __future__ import annotations

import numpy as np

from ..onnx_lib.backend.random import rand, randint
from .helper import tensor_dtype_to_np_dtype

try:
    from ..onnx_py._onnxpyprotoop import TensorProto  # type: ignore[attr-defined]
except ImportError:
    from . import TensorProto  # type: ignore[assignment]

_FLOAT_TYPES = frozenset(
    {
        int(TensorProto.FLOAT),
        int(TensorProto.DOUBLE),
        int(TensorProto.FLOAT16),
        int(TensorProto.BFLOAT16),
        int(TensorProto.FLOAT8E4M3FN),
        int(TensorProto.FLOAT8E4M3FNUZ),
        int(TensorProto.FLOAT8E5M2),
        int(TensorProto.FLOAT8E5M2FNUZ),
        int(TensorProto.FLOAT8E8M0),
        int(TensorProto.FLOAT4E2M1),
        int(TensorProto.COMPLEX64),
        int(TensorProto.COMPLEX128),
    }
)
_INT_TYPES = frozenset(
    {
        int(TensorProto.INT8),
        int(TensorProto.INT16),
        int(TensorProto.INT32),
        int(TensorProto.INT64),
        int(TensorProto.UINT8),
        int(TensorProto.UINT16),
        int(TensorProto.UINT32),
        int(TensorProto.UINT64),
        int(TensorProto.INT4),
        int(TensorProto.UINT4),
        int(TensorProto.INT2),
        int(TensorProto.UINT2),
    }
)

__all__ = ["make_random_input"]


def make_random_input(elem_type: int, shape: list[int], seed: int) -> np.ndarray:
    """Creates a random NumPy array for an ONNX tensor input.

    Uses the same deterministic pseudo-random generators as the onnx-light
    runtime (``onnx_light.onnx_lib.backend.random``).

    Args:
        elem_type: ``TensorProto`` data-type integer (e.g.
            ``TensorProto.FLOAT``).
        shape: Concrete non-negative integer dimensions.
        seed: Integer seed forwarded to the random generator.

    Returns:
        A ``numpy.ndarray`` of the appropriate dtype and shape.

    Raises:
        NotImplementedError: For unsupported element types (e.g. STRING).
    """
    np_dtype = tensor_dtype_to_np_dtype(elem_type)

    if elem_type == int(TensorProto.BOOL):
        values = randint(0, 2, size=shape, seed=seed, dtype=np.int32)
        return values.astype(bool)

    if elem_type in _FLOAT_TYPES:
        if elem_type in (int(TensorProto.COMPLEX64), int(TensorProto.COMPLEX128)):
            real = rand(*shape, seed=seed)
            imag = rand(*shape, seed=seed + 1)
            return (real + 1j * imag).astype(np_dtype)
        values = rand(*shape, seed=seed)
        return values.astype(np_dtype)

    if elem_type in _INT_TYPES:
        return randint(0, 10, size=shape, seed=seed, dtype=np_dtype)

    if elem_type == int(TensorProto.STRING):
        raise NotImplementedError(
            "STRING inputs are not supported by the run subcommand's random input generator."
        )

    raise NotImplementedError(
        f"Unsupported element type {elem_type} for random input generation."
    )
