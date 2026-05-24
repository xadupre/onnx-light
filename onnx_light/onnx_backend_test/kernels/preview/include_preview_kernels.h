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
// Reference implementations of the ``preview`` backend test kernels.
//
// These kernels implement reference computations for operators in the
// ``ai.onnx.preview`` domain. Each kernel is exposed as a small class whose
// constructor takes a :ref:`KernelContext` (carrying the opset against which
// the kernel must behave) and whose ``operator()`` performs the computation.
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
// ---------------------------------------------------------------------------

/// Reference implementation of ``ai.onnx.preview::FlexAttention`` (v1).
///
/// Computes ``Y = Softmax((Q @ K^T) * scale, axis=-1) @ V`` over rank-4
/// (batched, multi-head) FLOAT inputs. Supports Grouped Query Attention
/// (GQA): when ``q_num_heads != kv_num_heads`` each K/V head is shared by
/// a contiguous group of query heads, i.e. query head ``h`` uses K/V head
/// ``floor(h / (q_num_heads / kv_num_heads))``; ``q_num_heads`` must be a
/// multiple of ``kv_num_heads``.
///
/// The optional ``score_mod`` and ``prob_mod`` modifier subgraphs of the
/// upstream operator are not modeled by this reference kernel — it
/// implements the un-modified baseline that backends are expected to
/// reproduce when neither modifier is provided.
///
/// Only FLOAT tensors are supported.
class FlexAttention {
public:
  explicit FlexAttention(const KernelContext &ctx) : ctx_(ctx) {}

  /// Computes the attention output for the given Q, K, V tensors using the
  /// default scaling factor ``1 / sqrt(head_size)``.
  Tensor operator()(const Tensor &Q, const Tensor &K, const Tensor &V) const;

  /// Computes the attention output for the given Q, K, V tensors using an
  /// explicit ``scale`` value (matching the ``scale`` attribute of the
  /// operator).
  Tensor operator()(const Tensor &Q, const Tensor &K, const Tensor &V, float scale) const;

  /// In-place overload writing into a caller-allocated ``output`` tensor.
  /// ``output`` must already be a FLOAT tensor whose shape equals
  /// ``(batch_size, q_num_heads, q_seq_len, v_head_size)`` and whose
  /// ``data`` buffer has been sized to match.
  void operator()(const Tensor &Q, const Tensor &K, const Tensor &V, float scale,
                  Tensor &output) const;

  /// FlexAttention computes a fresh output buffer from independent reads of
  /// Q, K, V and never aliases an input buffer.
  static constexpr bool CanRunInPlace() noexcept { return false; }

private:
  KernelContext ctx_;
};

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
