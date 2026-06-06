// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/simple_tensor.h"

#include <string>
#include <tuple>
#include <vector>

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
// integer output ``y`` whose element type is taken from
// ``y_zero_point`` (UINT8, INT8, UINT16 or INT16). When ``y_zero_point`` is
// omitted the output defaults to UINT8 with a zero point of 0, matching the
// ONNX default. The
// kernel implements the saturating round-half-to-even rule used by ONNX:
// ``y = saturate(round(x / y_scale) + y_zero_point)``.
//
// ``DequantizeLinear`` mirrors the ONNX ``DequantizeLinear`` operator
// restricted to the per-tensor case: an integer or float8 input ``x``
// (UINT8, INT8, UINT16, INT16, INT32, FLOAT8E4M3FN, FLOAT8E4M3FNUZ,
// FLOAT8E5M2 or FLOAT8E5M2FNUZ), a scalar FLOAT ``x_scale`` and an
// optional scalar ``x_zero_point`` of the same element type as ``x``. The
// output ``y`` is FLOAT with the same shape as ``x``:
// ``y = (x - x_zero_point) * x_scale``. When ``x_zero_point`` is omitted
// the zero point defaults to 0.
//
// Each kernel class also exposes a ``static constexpr bool CanRunInPlace()``
// query indicating whether the output tensor's data buffer may alias one of
// the input tensors' buffers. ``QuantizeLinear``'s output element type
// (UINT8/INT8) differs from its FLOAT input, so storage cannot be shared
// with an input.
// ---------------------------------------------------------------------------

/// Per-tensor linear quantization of a FLOAT input ``x`` to an integer
/// output. The output element type is taken from ``y_zero_point`` (UINT8,
/// INT8, UINT16 or INT16); if ``y_zero_point`` is omitted the output defaults
/// to UINT8 with a zero point of 0.
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

  /// Output element type differs from the FLOAT input element type, so storage
  /// can never be shared with an input.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Per-tensor linear dequantization of an integer or float8 input ``x`` to a
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

  /// Output element type (FLOAT) differs from the integer/float8 input
  /// element type, so storage can never be shared with an input.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Per-tensor dynamic linear quantization of a FLOAT input ``x`` to a UINT8
/// output ``y`` together with the derived scalar ``y_scale`` (FLOAT) and
/// scalar ``y_zero_point`` (UINT8). Matches the ONNX ``DynamicQuantizeLinear``
/// operator (opset 11) for the canonical uint8 case:
///   ``y_scale = (max(0, max(x)) - min(0, min(x))) / 255``
///   ``y_zero_point = saturate(round(-min(0, min(x)) / y_scale))``
///   ``y = saturate(round(x / y_scale) + y_zero_point)``
class DynamicQuantizeLinear : public KernelBase {
public:
  using KernelBase::KernelBase;

  /// Returning overload: produces ``{y, y_scale, y_zero_point}``.
  std::tuple<Tensor, Tensor, Tensor> operator()(const Tensor &x) const;

  /// In-place overload writing into caller-allocated outputs.
  void operator()(const Tensor &x, Tensor &y, Tensor &y_scale, Tensor &y_zero_point) const;

  /// Output element type (UINT8) differs from the FLOAT input element type,
  /// so storage can never be shared with the input.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference ``QLinearMatMul`` kernel for the per-tensor (scalar scales and
/// scalar zero-points) case, restricted to INT8/UINT8 inputs and outputs and
/// FLOAT scales. Implements
/// ``y = saturate(round(((a - a_zp) * a_scale) * ((b - b_zp) * b_scale) /
/// y_scale) + y_zp)``. Matrix multiplication follows the standard
/// :cpp:class:`MatMul` broadcasting rules.
class QLinearMatMul : public KernelBase {
public:
  using KernelBase::KernelBase;

  /// Returning overload. Output element type is taken from ``y_zero_point``.
  Tensor operator()(const Tensor &a, const Tensor &a_scale, const Tensor &a_zero_point,
                    const Tensor &b, const Tensor &b_scale, const Tensor &b_zero_point,
                    const Tensor &y_scale, const Tensor &y_zero_point) const;

  /// In-place overload writing into a caller-allocated output.
  void operator()(const Tensor &a, const Tensor &a_scale, const Tensor &a_zero_point,
                  const Tensor &b, const Tensor &b_scale, const Tensor &b_zero_point,
                  const Tensor &y_scale, const Tensor &y_zero_point, Tensor &output) const;

  /// Output dtype/shape generally differs from any input, so storage cannot be shared.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference N-D ``QLinearConv`` kernel for the per-tensor (or per-output-channel
/// ``w_scale`` / ``w_zero_point``) case, restricted to INT8/UINT8 ``x``, ``w``
/// and ``y`` and FLOAT scales. Implements
/// ``y = saturate(round(((x - x_zp) * x_scale) * ((w - w_zp) * w_scale) /
/// y_scale + bias * x_scale * w_scale / y_scale) + y_zp)`` using the standard
/// :cpp:class:`Conv` shape/padding/dilation rules.
class QLinearConv : public KernelBase {
public:
  /// Attributes carried by the ONNX ``QLinearConv`` operator.
  struct Attributes {
    std::vector<int64_t> kernel_shape;
    std::vector<int64_t> strides;
    std::vector<int64_t> pads;
    std::vector<int64_t> dilations;
    int64_t group = 1;
    std::string auto_pad = "NOTSET";
  };

  using KernelBase::KernelBase;

  /// Returning overload. ``B`` may be a default-constructed (empty) ``Tensor``
  /// to indicate the optional bias is missing.
  Tensor operator()(const Tensor &x, const Tensor &x_scale, const Tensor &x_zero_point,
                    const Tensor &w, const Tensor &w_scale, const Tensor &w_zero_point,
                    const Tensor &y_scale, const Tensor &y_zero_point, const Tensor &B,
                    const Attributes &attrs) const;

  void operator()(const Tensor &x, const Tensor &x_scale, const Tensor &x_zero_point,
                  const Tensor &w, const Tensor &w_scale, const Tensor &w_zero_point,
                  const Tensor &y_scale, const Tensor &y_zero_point, const Tensor &B,
                  const Attributes &attrs, Tensor &output) const;

  /// Output dtype/shape generally differs from any input, so storage cannot be shared.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
