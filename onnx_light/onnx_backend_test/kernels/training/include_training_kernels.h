// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/simple_tensor.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

// ---------------------------------------------------------------------------
// Reference implementations of the ``training`` backend test kernels.
//
// These kernels implement reference computations for operators in the
// ``ai.onnx.preview.training`` domain. Each kernel is exposed as a small
// class whose constructor takes a :ref:`KernelContext` (carrying the opset
// against which the kernel must behave) and whose ``operator()`` performs
// the computation.
//
// Two flavors of ``operator()`` are provided:
//
//   * The returning overload (``std::vector<Tensor> operator()(...) const``)
//     allocates fresh ``Tensor`` objects whose data buffers are owned by the
//     returned vector.
//   * The in-place overload (``void operator()(..., std::vector<Tensor>
//     &outputs) const``) writes results into caller-supplied output tensors
//     whose buffers have already been allocated. The caller is responsible
//     for setting each ``output.data_type``, ``output.shape`` and sizing
//     ``output.data`` to match the operator's expected output; the kernel
//     validates these attributes and throws ``std::invalid_argument`` on
//     mismatch.
// ---------------------------------------------------------------------------

/// Reference implementation of ``ai.onnx.preview.training::Adam`` (v1).
///
/// Computes one iteration of the Adam stochastic gradient based
/// optimization algorithm for an arbitrary number ``N >= 1`` of optimized
/// tensors. For each optimized tensor ``X_i`` (with gradient ``G_i``,
/// accumulated gradient ``V_i`` and accumulated squared gradient ``H_i``)
/// the kernel evaluates the pseudo-code documented in the ONNX schema:
///
/// ```
///   G_reg = norm_coefficient * X + G
///   V_new = alpha * V + (1 - alpha) * G_reg
///   H_new = beta  * H + (1 - beta)  * G_reg * G_reg
///   H_sqrt = sqrt(H_new) + epsilon
///   R_adjusted = T > 0 ? R * sqrt(1 - beta^T) / (1 - alpha^T) : R
///   X_new = X - R_adjusted * V_new / H_sqrt
///   X_final = (1 - norm_coefficient_post) * X_new
/// ```
///
/// and returns ``{X_final_1..N, V_new_1..N, H_new_1..N}``. All optimized
/// tensors share the same scalar inputs ``R`` (FLOAT) and ``T`` (INT64).
/// Only FLOAT tensors are supported.
class Adam {
public:
  explicit Adam(const KernelContext &ctx) : ctx_(ctx) {}

  /// Computes one Adam iteration for ``N == Xs.size()`` optimized tensors
  /// and allocates fresh output tensors. ``Xs``, ``Gs``, ``Vs`` and ``Hs``
  /// must all have the same length and pairwise-matching FLOAT shapes.
  /// ``R`` must be a scalar FLOAT tensor and ``T`` a scalar INT64 tensor.
  /// The trailing ``alpha``, ``beta``, ``epsilon``, ``norm_coefficient`` and
  /// ``norm_coefficient_post`` parameters mirror the Adam ONNX schema
  /// attributes; defaults match the schema defaults.
  std::vector<Tensor> operator()(const Tensor &R, const Tensor &T, const std::vector<Tensor> &Xs,
                                 const std::vector<Tensor> &Gs, const std::vector<Tensor> &Vs,
                                 const std::vector<Tensor> &Hs, float alpha = 0.9f,
                                 float beta = 0.999f, float epsilon = 1e-6f,
                                 float norm_coefficient = 0.0f,
                                 float norm_coefficient_post = 0.0f) const;

  /// In-place overload writing into caller-allocated ``outputs``. The vector
  /// must contain exactly ``3 * Xs.size()`` tensors in the layout
  /// ``{X_final_1..N, V_new_1..N, H_new_1..N}`` where each output already
  /// matches the FLOAT data type, shape and buffer size of the corresponding
  /// optimized tensor.
  void operator()(const Tensor &R, const Tensor &T, const std::vector<Tensor> &Xs,
                  const std::vector<Tensor> &Gs, const std::vector<Tensor> &Vs,
                  const std::vector<Tensor> &Hs, std::vector<Tensor> &outputs, float alpha = 0.9f,
                  float beta = 0.999f, float epsilon = 1e-6f, float norm_coefficient = 0.0f,
                  float norm_coefficient_post = 0.0f) const;

  /// Adam writes its outputs based on independent reads of multiple input
  /// tensors and never aliases an input buffer.
  static constexpr bool CanRunInPlace() noexcept { return false; }

private:
  const KernelContext &ctx_;
};

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
