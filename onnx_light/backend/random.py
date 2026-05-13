from typing import Iterable

import numpy as np

_UINT64_MASK = (1 << 64) - 1
_UINT64_MODULO = 1 << 64
_INV_TWO_POW_53 = 1.0 / (1 << 53)
_DEFAULT_SEED = 0


def _normalize_seed(seed: int | np.integer | None) -> int:
    """Returns a normalized 64-bit seed.

    Converts ``None`` to the default seed (0) and masks integer seeds to
    64 bits for platform-stable behavior.
    """
    if seed is None:
        return _DEFAULT_SEED
    if isinstance(seed, np.integer):
        seed = int(seed)
    if not isinstance(seed, int):
        raise TypeError(f"seed must be an integer or None, not {type(seed)!r}.")
    return seed & _UINT64_MASK


def _next_uint64(state: int) -> tuple[int, int]:
    """Computes the next SplitMix64 state and output value.

    Returns:
        A tuple ``(next_state, random_value)``.
    """
    state = (state + 0x9E3779B97F4A7C15) & _UINT64_MASK
    mixed = state
    mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    mixed = mixed ^ (mixed >> 31)
    return state, mixed & _UINT64_MASK


def _normalize_size(size: int | Iterable[int] | None) -> tuple[int, ...]:
    """Converts a numpy-like size argument into a validated shape tuple.

    Returns ``()`` when *size* is ``None`` and raises if any dimension is
    negative.
    """
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


def _shape_to_count(shape: tuple[int, ...]) -> int:
    """Returns the number of elements in a shape."""
    if not shape:
        return 1
    return int(np.prod(shape, dtype=np.int64))


def rand(*shape: int, seed: int | np.integer | None = None) -> np.ndarray:
    """Returns deterministic uniform random values in [0, 1).

    Args:
        shape: Output dimensions. When empty, returns a scalar.
        seed: Optional integer seed.

    Returns:
        A float when no shape is provided, otherwise a ``np.ndarray``.
    """
    normalized_shape = _normalize_size(shape)
    count = _shape_to_count(normalized_shape)
    values = np.empty(count, dtype=np.float64)
    state = _normalize_seed(seed)
    for i in range(count):
        state, value = _next_uint64(state)
        values[i] = float(value >> 11) * _INV_TWO_POW_53
    if not normalized_shape:
        return float(values[0])
    return values.reshape(normalized_shape)


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
        size: Output size. ``None`` returns a scalar.
        seed: Optional integer seed.
        dtype: Integer dtype of the output.

    Returns:
        An integer scalar when ``size`` is ``None``, otherwise a ``np.ndarray``.
    """
    if high is None:
        high = low
        low = 0
    low = int(low)
    high = int(high)
    if high <= low:
        raise ValueError(f"high must be greater than low, got low={low!r} and high={high!r}.")
    output_dtype = np.dtype(dtype)
    if output_dtype.kind not in {"i", "u"}:
        raise TypeError(f"dtype must be an integer dtype, not {dtype!r}.")
    normalized_shape = _normalize_size(size)
    count = _shape_to_count(normalized_shape)
    span = high - low
    limit = _UINT64_MODULO - (_UINT64_MODULO % span)
    values = np.empty(count, dtype=np.uint64)
    state = _normalize_seed(seed)
    for i in range(count):
        while True:
            state, candidate = _next_uint64(state)
            if candidate < limit:
                values[i] = candidate % span
                break
    values = values.astype(output_dtype, copy=False)
    values += output_dtype.type(low)
    if not normalized_shape:
        return values.reshape(()).item()
    return values.reshape(normalized_shape)


def randn(*shape: int, seed: int | np.integer | None = None) -> np.ndarray:
    """Returns deterministic pseudo-random values with an approximate normal distribution.

    Samples are produced using the Irwin-Hall approximation (sum of 12 uniform
    values minus 6), which approximates a normal distribution.

    Args:
        shape: Output dimensions. When empty, returns a scalar.
        seed: Optional integer seed.

    Returns:
        A float when no shape is provided, otherwise a ``np.ndarray``.
    """
    normalized_shape = _normalize_size(shape)
    count = _shape_to_count(normalized_shape)
    values = np.empty(count, dtype=np.float64)
    state = _normalize_seed(seed)
    for i in range(count):
        sample = 0.0
        for _ in range(12):
            state, value = _next_uint64(state)
            sample += float(value >> 11) * _INV_TWO_POW_53
        values[i] = sample - 6.0
    if not normalized_shape:
        return float(values[0])
    return values.reshape(normalized_shape)
