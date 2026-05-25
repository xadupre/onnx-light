// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/simple_tensor.h"

#include <cstdint>
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
// FLOAT tensors with a non-empty ``kernel_shape`` and an ``auto_pad`` value
// of ``NOTSET`` (explicit ``pads``). It supports an arbitrary number of
// spatial dimensions (N, C, D1, ..., Dk), the ``strides`` attribute
// (default 1 along every spatial axis), explicit ``pads`` (default 0 along
// every spatial begin/end), ``ceil_mode`` (default 0 — floor), and
// ``count_include_pad`` (default 0). The kernel exposes only the primary
// output ``y``; the optional second ``Indices`` output (added in opset 22)
// is not produced.
//
// Each kernel class also exposes a ``static constexpr bool CanRunInPlace()``
// query indicating whether the output tensor's data buffer may alias one of
// the input tensors' buffers. ``AveragePool``'s output shape generally
// differs from its input, so storage cannot be shared with an input.
// ---------------------------------------------------------------------------

/// N-D average pooling on a FLOAT tensor laid out as ``(N, C, D1, ..., Dk)``.
/// ``kernel_shape`` must have ``k`` entries; ``strides`` and ``pads``
/// (lengths ``k`` and ``2 * k`` respectively) default to all-ones and
/// all-zeros when omitted.
class AveragePool {
public:
  explicit AveragePool(const KernelContext &ctx) : ctx_(ctx) {}

  /// All attributes explicit. ``strides`` may be empty (treated as all 1)
  /// and ``pads`` may be empty (treated as all 0).
  Tensor operator()(const Tensor &x, const std::vector<int64_t> &kernel_shape,
                    const std::vector<int64_t> &strides = {}, const std::vector<int64_t> &pads = {},
                    bool ceil_mode = false, bool count_include_pad = false) const;

  void operator()(const Tensor &x, const std::vector<int64_t> &kernel_shape,
                  const std::vector<int64_t> &strides, const std::vector<int64_t> &pads,
                  bool ceil_mode, bool count_include_pad, Tensor &output) const;

  /// Output shape generally differs from the input shape, so the output
  /// buffer cannot in general alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return false; }

private:
  KernelContext ctx_;
};

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
