// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/simple_tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

// ---------------------------------------------------------------------------
// Reference implementations of the ``nn`` (neural network) backend test
// kernels.
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
// ``AveragePool`` mirrors the ONNX ``AveragePool`` operator restricted to
// FLOAT tensors with a non-empty ``kernel_shape``. It supports an
// arbitrary number of spatial dimensions (N, C, D1, ..., Dk), the
// ``strides`` attribute (default 1 along every spatial axis), explicit
// ``pads`` (default 0 along every spatial begin/end), ``dilations``
// (default 1 along every spatial axis), ``ceil_mode`` (default 0 —
// floor), ``count_include_pad`` (default 0), and the ``auto_pad`` attribute
// (one of ``NOTSET`` (default), ``SAME_UPPER``, ``SAME_LOWER`` or
// ``VALID``; when ``auto_pad`` is not ``NOTSET`` the ``pads`` argument
// must be empty and the begin/end padding is computed from ``auto_pad``).
// The kernel exposes only the primary output ``y``; the optional second
// ``Indices`` output (added in opset 22) is not produced.
//
// Each kernel class also exposes a ``static constexpr bool CanRunInPlace()``
// query indicating whether the output tensor's data buffer may alias one of
// the input tensors' buffers. ``AveragePool``'s output shape generally
// differs from its input, so storage cannot be shared with an input.
// ---------------------------------------------------------------------------

/// N-D average pooling on a FLOAT tensor laid out as ``(N, C, D1, ..., Dk)``.
/// ``kernel_shape`` must have ``k`` entries; ``strides``, ``pads`` and
/// ``dilations`` (lengths ``k``, ``2 * k`` and ``k`` respectively) default
/// to all-ones / all-zeros / all-ones when omitted. ``auto_pad`` defaults
/// to ``NOTSET`` (use explicit ``pads``); when set to ``SAME_UPPER``,
/// ``SAME_LOWER`` or ``VALID`` the ``pads`` argument must be empty and
/// the begin/end padding is computed from the input shape.
class AveragePool {
public:
  explicit AveragePool(const KernelContext &ctx) : ctx_(ctx) {}

  /// All attributes explicit. ``strides`` may be empty (treated as all 1),
  /// ``pads`` may be empty (treated as all 0) and ``dilations`` may be
  /// empty (treated as all 1).
  Tensor operator()(const Tensor &x, const std::vector<int64_t> &kernel_shape,
                    const std::vector<int64_t> &strides = {}, const std::vector<int64_t> &pads = {},
                    bool ceil_mode = false, bool count_include_pad = false,
                    const std::vector<int64_t> &dilations = {},
                    const std::string &auto_pad = "NOTSET") const;

  void operator()(const Tensor &x, const std::vector<int64_t> &kernel_shape,
                  const std::vector<int64_t> &strides, const std::vector<int64_t> &pads,
                  bool ceil_mode, bool count_include_pad, Tensor &output,
                  const std::vector<int64_t> &dilations = {},
                  const std::string &auto_pad = "NOTSET") const;

  /// Output shape generally differs from the input shape, so the output
  /// buffer cannot in general alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return false; }

private:
  const KernelContext &ctx_;
};

/// Inference-mode BatchNormalization on a FLOAT input laid out as
/// ``(N, C, D1, ..., Dk)`` (any rank >= 2; rank 1 is also accepted with
/// ``C`` treated as 1). All four extra inputs (``scale``, ``B``,
/// ``input_mean``, ``input_var``) must be 1-D FLOAT tensors of length
/// ``C``. The kernel implements the inference formula
/// ``Y = (X - input_mean) / sqrt(input_var + epsilon) * scale + B``
/// using NumPy-style broadcasting along the channel axis. Training mode
/// (``training_mode = 1``, opset 14+) is not supported because the
/// reference backend test cases registered today exercise only the
/// inference path.
class BatchNormalization {
public:
  explicit BatchNormalization(const KernelContext &ctx) : ctx_(ctx) {}

  /// Returns the inference-mode primary output ``Y``. ``epsilon`` defaults
  /// to 1e-5f, the upstream default.
  Tensor operator()(const Tensor &x, const Tensor &scale, const Tensor &bias,
                    const Tensor &input_mean, const Tensor &input_var, float epsilon = 1e-5f) const;

  void operator()(const Tensor &x, const Tensor &scale, const Tensor &bias,
                  const Tensor &input_mean, const Tensor &input_var, Tensor &output,
                  float epsilon = 1e-5f) const;

  /// Output ``Y`` has the same shape as ``X`` so the output buffer may
  /// alias the input ``X`` buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }

private:
  const KernelContext &ctx_;
};

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
