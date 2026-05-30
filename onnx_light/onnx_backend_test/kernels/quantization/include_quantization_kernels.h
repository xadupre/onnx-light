// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/simple_tensor.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

// ---------------------------------------------------------------------------
// Reference implementations of the ``quantization`` backend test kernels.
//
// Each kernel is exposed as a small class whose constructor takes a
// :ref:`KernelContext` (carrying the opset against which the kernel must
// behave) and whose ``operator()`` performs the computation.
//
// Two flavors of ``operator()`` are provided:
//
//   * The returning overload (``Tensor operator()(...) const``) allocates a
//     fresh ``Tensor`` whose data buffer is owned by the returned value.
//   * The in-place overload (``void operator()(..., Tensor &output) const``)
//     writes results into a caller-supplied output tensor whose buffer has
//     already been allocated. The caller is responsible for setting
//     ``output.data_type``, ``output.shape`` and sizing ``output.data`` to
//     match the operator's expected output; the kernel validates these
//     attributes and throws ``std::invalid_argument`` on mismatch.
//
// ``QuantizeLinear`` mirrors the ONNX ``QuantizeLinear`` operator restricted
// to its most widely supported case: per-tensor (scalar ``y_scale`` and
// optional scalar ``y_zero_point``) quantization of a FLOAT input ``x`` to an
// 8-bit integer output ``y`` whose element type is taken from
// ``y_zero_point`` (UINT8 or INT8). When ``y_zero_point`` is omitted the
// output is UINT8 with a zero point of 0, matching the ONNX default. The
// kernel implements the saturating round-half-to-even rule used by ONNX:
// ``y = saturate(round(x / y_scale) + y_zero_point)``.
//
// ``DequantizeLinear`` mirrors the ONNX ``DequantizeLinear`` operator
// restricted to the per-tensor case: an 8-bit integer input ``x`` (UINT8 or
// INT8), a scalar FLOAT ``x_scale`` and an optional scalar ``x_zero_point``
// of the same element type as ``x``. The output ``y`` is FLOAT with the same
// shape as ``x``: ``y = (x - x_zero_point) * x_scale``. When
// ``x_zero_point`` is omitted the zero point defaults to 0.
//
// Each kernel class also exposes a ``static constexpr bool CanRunInPlace()``
// query indicating whether the output tensor's data buffer may alias one of
// the input tensors' buffers. ``QuantizeLinear``'s output element type
// (UINT8/INT8) differs from its FLOAT input, so storage cannot be shared
// with an input.
// ---------------------------------------------------------------------------

/// Per-tensor linear quantization of a FLOAT input ``x`` to an 8-bit integer
/// output. The output element type is taken from ``y_zero_point`` (UINT8 or
/// INT8); if ``y_zero_point`` is omitted the output is UINT8 with a zero
/// point of 0.
class QuantizeLinear : public KernelBase {
public:
  using KernelBase::KernelBase;

  /// Omitted ``y_zero_point``: output is UINT8 with zero point 0.
  Tensor operator()(const Tensor &x, const Tensor &y_scale) const;
  void operator()(const Tensor &x, const Tensor &y_scale, Tensor &output) const;

  /// Explicit ``y_zero_point``: its data_type drives the output element type.
  Tensor operator()(const Tensor &x, const Tensor &y_scale, const Tensor &y_zero_point) const;
  void operator()(const Tensor &x, const Tensor &y_scale, const Tensor &y_zero_point,
                  Tensor &output) const;

  /// Output element type (UINT8/INT8) differs from the FLOAT input element
  /// type, so storage can never be shared with an input.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Per-tensor linear dequantization of an 8-bit integer input ``x`` to a
/// FLOAT output ``y`` using ``y = (x - x_zero_point) * x_scale``. When
/// ``x_zero_point`` is omitted the zero point defaults to 0.
class DequantizeLinear : public KernelBase {
public:
  using KernelBase::KernelBase;

  /// Omitted ``x_zero_point``: zero point defaults to 0.
  Tensor operator()(const Tensor &x, const Tensor &x_scale) const;
  void operator()(const Tensor &x, const Tensor &x_scale, Tensor &output) const;

  /// Explicit ``x_zero_point``: must have the same element type as ``x``.
  Tensor operator()(const Tensor &x, const Tensor &x_scale, const Tensor &x_zero_point) const;
  void operator()(const Tensor &x, const Tensor &x_scale, const Tensor &x_zero_point,
                  Tensor &output) const;

  /// Output element type (FLOAT) differs from the 8-bit integer input
  /// element type, so storage can never be shared with an input.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
