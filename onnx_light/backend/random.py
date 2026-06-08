"""Deterministic pseudo-random helpers.

This module is a thin Python wrapper around the C++ implementation in
``onnx_light.onnx_kernels`` exposed through the ``_onnxpy.backend``
nanobind submodule (compiled from the ``_onnxpykernels`` extension).
The behavior is bit-identical to the prior pure-Python implementation.
"""

from typing import Iterable

import numpy as np

from ..onnx_py._onnxpy import backend as _C  # type: ignore[attr-defined]

_UINT64_MASK = (1 << 64) - 1


def _normalize_seed(seed: int | np.integer | None) -> int | None:
    """Returns a normalized 64-bit seed, or ``None`` to use the default seed."""
    if seed is None:
        return None
    if isinstance(seed, np.integer):
        seed = int(seed)
    if not isinstance(seed, int):
        raise TypeError(f"seed must be an integer or None, not {type(seed)!r}.")
    return seed & _UINT64_MASK


def _normalize_size(size: int | Iterable[int] | None) -> tuple[int, ...]:
    """Converts a numpy-like size argument into a validated shape tuple."""
    if size is None:
        return ()
    if isinstance(size, int):
        if size < 0:
            raise ValueError(f"size cannot be negative, got {size!r}.")
        return (size,)
    shape = tuple(int(dim) for dim in size)
    if any(dim < 0 for dim in shape):
        raise ValueError(f"size cannot contain negative dimensions, got {shape!r}.")
    return shape


def rand(*shape: int, seed: int | np.integer | None = None) -> np.ndarray:
    """Returns deterministic uniform random values in [0, 1).

    Args:
        shape: Output dimensions. When empty, returns a scalar.
        seed: Optional integer seed.

    Returns:
        A ``np.ndarray`` of float64 values with the requested shape.
    """
    normalized_shape = _normalize_size(shape)
    values = _C.rand(list(normalized_shape), _normalize_seed(seed))
    return np.asarray(values, dtype=np.float64).reshape(normalized_shape)


def randint(
    low: int,
    high: int | None = None,
    size: int | Iterable[int] | None = None,
    seed: int | np.integer | None = None,
    dtype=np.int64,
) -> np.ndarray:
    """Returns deterministic pseudo-random integers.

    Args:
        low: Lower bound, inclusive. If ``high`` is ``None``, this becomes the
            exclusive upper bound and the lower bound becomes 0.
        high: Exclusive upper bound.
        size: Output size. Must not be ``None``.
        seed: Optional integer seed.
        dtype: Integer dtype of the output.

    Returns:
        A ``np.ndarray`` of integers with the requested shape.
    """
    assert size is not None, "size cannot be None"
    if high is None:
        high = low
        low = 0
    low = int(low)
    high = int(high)
    output_dtype = np.dtype(dtype)
    if output_dtype.kind not in {"i", "u"}:
        raise TypeError(f"dtype must be an integer dtype, not {dtype!r}.")
    normalized_shape = _normalize_size(size)
    values = _C.randint(low, high, list(normalized_shape), _normalize_seed(seed))
    return np.asarray(values, dtype=output_dtype).reshape(normalized_shape)


def randn(*shape: int, seed: int | np.integer | None = None) -> np.ndarray:
    """Returns deterministic pseudo-random values with an approximate normal distribution.

    Samples are produced using the Irwin-Hall approximation (sum of 12 uniform
    values minus 6), which approximates a normal distribution.

    Args:
        shape: Output dimensions. When empty, returns a scalar.
        seed: Optional integer seed.

    Returns:
        A ``np.ndarray`` of float64 values with the requested shape.
    """
    normalized_shape = _normalize_size(shape)
    values = _C.randn(list(normalized_shape), _normalize_seed(seed))
    return np.asarray(values, dtype=np.float64).reshape(normalized_shape)
