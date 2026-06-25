"""Public helpers that complement :mod:`onnx_light.onnx`."""

from __future__ import annotations

import numpy as np

from ..onnx_lib.backend.random import rand, randint
from .helper import tensor_dtype_to_np_dtype

try:
    from ..onnx_py._onnxpyprotoop import TensorProto  # type: ignore[attr-defined]
except ImportError:
    from . import TensorProto  # type: ignore[assignment]

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
    dtype_name = np.dtype(np_dtype).name

    if elem_type == int(TensorProto.BOOL):
        values = randint(0, 2, size=shape, seed=seed, dtype=np.int32)
        return values.astype(bool)

    if np.issubdtype(np_dtype, np.complexfloating) or dtype_name.startswith("complex"):
        real = rand(*shape, seed=seed)
        imag = rand(*shape, seed=seed + 1)
        return (real + 1j * imag).astype(np_dtype)

    if np.issubdtype(np_dtype, np.floating) or dtype_name.startswith(("float", "bfloat")):
        values = rand(*shape, seed=seed)
        return values.astype(np_dtype)

    if np.issubdtype(np_dtype, np.integer):
        return randint(0, 10, size=shape, seed=seed, dtype=np_dtype)

    if dtype_name.startswith(("int", "uint")):
        values = randint(0, 10, size=shape, seed=seed, dtype=np.int32)
        return values.astype(np_dtype)

    if elem_type == int(TensorProto.STRING):
        raise NotImplementedError(
            "STRING inputs are not supported by the run subcommand's random input generator."
        )

    raise NotImplementedError(
        f"Unsupported element type {elem_type} for random input generation."
    )
