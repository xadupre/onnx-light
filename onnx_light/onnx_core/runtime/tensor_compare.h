// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Public tolerance-based tensor comparison helper. onnx-light's backend
// run-model tests compare kernel outputs bit-for-bit because the expected
// outputs are produced by the very same kernels the runtime dispatches to.
// External callers (and tests that compare against a reference implementation)
// often need a tolerance-based comparison instead; :cpp:func:`CompareTensors`
// provides one with the same semantics as ``numpy.allclose``.

#pragma once

#include "onnx_core/runtime/simple_tensor.h"

#include <string>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

/**
 * Outcome of :cpp:func:`CompareTensors`.
 *
 * ``close`` is true when the two tensors match within the requested
 * tolerance. When it is false, ``message`` holds a human-readable description
 * of the first mismatch (differing data type, shape, string value, or numeric
 * element).
 */
struct TensorComparison {
  /// Whether the tensors match within tolerance.
  bool close = false;
  /// Human-readable description of the first mismatch (empty when ``close``).
  std::string message;
};

/**
 * Compares two tensors element-wise within an absolute and relative tolerance,
 * mirroring ``numpy.allclose``.
 *
 * The comparison first requires ``actual`` and ``expected`` to share the same
 * ``data_type`` and ``shape``. ``STRING`` tensors are then compared for exact
 * equality of their string values. Numeric tensors are compared element-wise:
 * two finite values ``a`` and ``b`` are considered close when
 * ``|a - b| <= atol + rtol * |b|``. Infinities must match in both value and
 * sign. ``NaN`` values compare unequal unless ``equal_nan`` is true, in which
 * case two ``NaN`` values are treated as equal.
 *
 * Half-precision (``FLOAT16``/``BFLOAT16``) elements are decoded to ``float``
 * before comparison. Element types that cannot be represented as ``double``
 * here (the ``FLOAT8*`` variants and the 4-bit / 2-bit packed types) fall back
 * to an exact byte-for-byte comparison.
 *
 * @param actual The computed tensor.
 * @param expected The reference tensor.
 * @param rtol Relative tolerance (default ``1e-5``).
 * @param atol Absolute tolerance (default ``1e-8``).
 * @param equal_nan When true, ``NaN`` values in matching positions compare
 *                  equal (default ``false``).
 * @return A :cpp:class:`TensorComparison` describing the outcome.
 */
TensorComparison CompareTensors(const Tensor &actual, const Tensor &expected, double rtol = 1e-5,
                                double atol = 1e-8, bool equal_nan = false);

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
