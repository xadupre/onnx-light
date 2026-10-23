# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Computes descriptive tensor statistics using the C++ runtime."""

from collections.abc import Sequence
from typing import TypeAlias

import numpy

from ..onnx import EncodedValueProto, ModelProto, TensorProto
from ..onnx_py._onnxpybackend import backend_test  # type: ignore[missing-import]
from ..onnx_py._onnxpykernels import runtime  # type: ignore[missing-import]

TensorStatistics: TypeAlias = runtime.TensorStatistics


def compute_tensor_statistics(
    value: (
        numpy.ndarray
        | backend_test.Tensor
        | TensorProto
        | EncodedValueProto
        | runtime.RuntimeValue
    ),
    *,
    bins: int = 0,
    quantiles: Sequence[float] = (),
    median: bool = False,
    rectangle_shape: Sequence[int] = (),
    model: ModelProto | None = None,
    max_output_bytes: int = 64 * 1024 * 1024,
) -> TensorStatistics:
    """Computes descriptive statistics and optional summaries in C++.

    Args:
        value: A NumPy array, runtime ``Tensor``, ``TensorProto``,
            ``EncodedValueProto``, or tensor/encoded ``RuntimeValue``.
            ``SharedQuantizedValue`` is an alias of ``RuntimeValue`` and is
            also accepted. Supports real numeric and boolean tensors, including
            noncontiguous NumPy arrays; strings and complex types are unsupported.
            Encoded inputs describe reconstructed numeric values.
        bins: Nonnegative histogram bin count, default 0 (automatic: 50 bins for
            more than 500 elements, 10 otherwise). A positive value overrides
            the automatic count. No histogram is computed for fewer than
            10 elements, regardless of this option.
        quantiles: Probabilities in [0, 1], default () (no quantiles).
            Computes values in the requested order using linear interpolation
            at ``q * (count - 1)``, preserving exact endpoints.
        median: Whether to compute the median independently of ``quantiles``,
            default False.
        rectangle_shape: Positive [height, width], default () (no rectangle).
            Reshapes the tensor to ``[shape[0], product(shape[1:])]`` (or
            ``[1, count]`` for a vector or scalar), then resizes using
            fractional cell overlap to produce area averages.
        model: Optional ``ModelProto``, default None. Resolves the type catalogue
            and shared parameters of encoded values. A shared ``RuntimeValue``
            can also carry its own parameter ownership.
        max_output_bytes: Per-call allocation budget, default 64 MiB, for the
            combined option-driven numeric output payloads: ``8*bins + 8*(bins+1)``
            histogram bytes, ``8*height*width`` rectangle bytes, 8 bytes per
            quantile, and 8 bytes for the median. ``bins=0`` charges 50 bins
            (808 bytes), the largest automatic histogram. The budget can be
            raised or lowered explicitly.

    Returns:
        TensorStatistics: Statistics with ``min``, ``max``, ``mean``, ONNX
        ``dtype``, ``shape``, element ``count``, and ``size_bytes``.
        Integer, boolean, and encoded inputs also report ``distinct_count``;
        other inputs leave it as None. Encoded inputs report the logical tensor
        dtype and shape, and their original encoded payload length in
        ``size_bytes``, excluding descriptor metadata and shared parameters.

        ``distribution`` contains histogram counts and ``bin_edges`` the
        corresponding boundaries. Bins are left-inclusive and right-exclusive,
        except for the final bin, which includes the maximum. Only finite
        values enter the histogram; no finite values produce no histogram.
        Absent histograms have empty counts and edges.

        ``quantiles`` contains the requested values, or an empty list.
        ``median`` is None unless requested. ``rectangle_map`` is a flat
        row-major list with the requested ``rectangle_shape``; both are empty
        when no map is produced.

    Raises:
        TypeError: If an input or option cannot be converted to its native type.
        ValueError: If a tensor dtype or runtime value kind is unsupported,
            storage or encoded metadata is malformed, required encoded types or
            shared parameters cannot be resolved, ``bins`` is negative or too
            large, a quantile is nonfinite or outside [0, 1], or
            ``rectangle_shape`` is neither empty nor two positive dimensions
            with a representable output size, or the combined requested payload
            exceeds ``max_output_bytes``.

    Notes:
        NaNs propagate to the minimum, maximum, mean, and requested quantiles
        and median. Infinities follow numeric ordering and IEEE arithmetic.
        Nonfinite values propagate through overlapping rectangle cells.
        Empty tensors return NaN numeric statistics (including each requested
        quantile and the median if requested), no histogram or rectangle map,
        and a zero distinct count when applicable. Options are validated even
        for empty tensors.

        All requested outputs are charged even when no histogram or rectangle
        would be produced. Oversized options are rejected before output allocation
        or input conversion/dequantization. This is not a total-memory limit:
        input storage, decoding, input-dependent workspaces, container
        capacity/metadata, and Python result copies are excluded.

        For unencoded tensors, basic reductions and histograms read the tensor
        directly; only order statistics and rectangle maps materialize a full
        numeric copy.
        Sorting is performed only when quantiles or the median are requested.
        Exact distinct counts use storage proportional to the number of
        distinct values.
    """
    return runtime.compute_tensor_statistics(
        value,
        bins=bins,
        quantiles=quantiles,
        median=median,
        rectangle_shape=rectangle_shape,
        model=model,
        max_output_bytes=max_output_bytes,
    )


__all__ = ["TensorStatistics", "compute_tensor_statistics"]
