# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Computes descriptive tensor statistics using the C++ runtime.

``compute_tensor_statistics(value, *, bins=0, quantiles=[], median=False,
rectangle_shape=[], model=None)`` accepts a NumPy array, runtime ``Tensor``,
``TensorProto``, ``EncodedValueProto``, ``SharedQuantizedValue``, or
``RuntimeValue`` and returns
a :class:`TensorStatistics` instance. ``model`` optionally resolves the type
catalogue and shared parameters of encoded values.

The result includes ``min``, ``max``, ``mean``, the ONNX ``dtype``, ``shape``,
element ``count``, and ``size_bytes``. Integer and quantized inputs also report
``distinct_count``. Other inputs leave it as ``None``.
For encoded values, ``dtype`` is the logical tensor dtype and ``size_bytes``
is the encoded payload length, excluding descriptor metadata and shared parameters.

``distribution`` contains histogram counts and ``bin_edges`` the corresponding
boundaries. No histogram is computed for fewer than 10 elements. By default,
there are 50 bins for more than 500 elements and 10 otherwise; a positive
``bins`` overrides this choice. Bins are left-inclusive and right-exclusive,
except for the final bin, which includes the maximum.
Only finite values enter the histogram; no finite values produces no histogram.

``quantiles`` requests probabilities in [0, 1] and returns values in the same
order, using linear interpolation. ``median=True`` requests the median
independently. Neither requires sorting unless requested.
NaNs propagate to the minimum, maximum, mean and requested quantiles. Infinities
follow numeric ordering and IEEE arithmetic. Empty tensors return NaN numeric
statistics and no histogram or rectangle map.

``rectangle_shape=[height, width]`` requests an area-averaged rectangle map.
The tensor is reshaped to ``[shape[0], product(shape[1:])]`` (or ``[1, count]``
for a vector or scalar), then resized using fractional cell overlap. The result
is a flat row-major ``rectangle_map`` with the requested ``rectangle_shape``.
"""

from ..onnx_py._onnxpykernels import runtime  # type: ignore[missing-import]

TensorStatistics = runtime.TensorStatistics
compute_tensor_statistics = runtime.compute_tensor_statistics

__all__ = ["TensorStatistics", "compute_tensor_statistics"]
