// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernel_context.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/simple_tensor.h"

#include <functional>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels {
// Re-exports the runtime types moved to ``onnx_core::runtime`` so
// kernel implementations below can keep referring to them
// unqualified, matching pre-move code.
using namespace ::onnx_light::core::runtime;

namespace kernel {
using ::onnx_light::core::runtime::DefaultOpset;
using ::onnx_light::core::runtime::KernelBase;
using ::onnx_light::core::runtime::KernelContext;
using ::onnx_light::core::runtime::OpsetId;

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
/// (batched, multi-head) floating-point inputs (FLOAT or DOUBLE; FLOAT16 and
/// BFLOAT16 are supported via internal promotion to FLOAT32). Supports
/// Grouped Query Attention (GQA): when ``q_num_heads != kv_num_heads`` each
/// K/V head is shared by a contiguous group of query heads, i.e. query head
/// ``h`` uses K/V head ``floor(h / (q_num_heads / kv_num_heads))``;
/// ``q_num_heads`` must be a multiple of ``kv_num_heads``.
///
/// The optional ``score_mod`` modifier subgraph is supported via the
/// ``ScoreModFn`` callback overload: callers that have a way to evaluate
/// the ``score_mod`` subgraph (e.g. a graph-executor) can pass a callable
/// that receives the pre-softmax score tensor of shape
/// ``(batch_size, q_num_heads, q_seq_len, kv_seq_len)`` and rewrites it
/// in place; the kernel then runs ``Softmax`` on the rewritten scores
/// before the final ``probs @ V`` matmul. Overloads without a callback
/// retain the baseline behavior (``score_mod`` treated as identity).
///
/// The optional ``prob_mod`` modifier subgraph is supported via the
/// ``ProbModFn`` callback overload: callers that have a way to evaluate
/// the ``prob_mod`` subgraph (e.g. a graph-executor) can pass a callable
/// that receives the post-softmax probability tensor of shape
/// ``(batch_size, q_num_heads, q_seq_len, kv_seq_len)`` and rewrites it
/// in place; the kernel then uses the rewritten probabilities for the
/// final ``probs @ V`` matmul. Overloads without a callback retain the
/// baseline behavior (``prob_mod`` treated as identity).
///
/// FLOAT and DOUBLE tensors are supported (and FLOAT16/BFLOAT16 via internal
/// promotion to FLOAT32). ``Q``, ``K`` and ``V`` must share the same element
/// type.
class FlexAttention : public KernelBase {
public:
  static constexpr const char *name = "onnx_kernels:CPU:ai.onnx.preview:FlexAttention";
  void Run(RuntimeContext &rt) override;
  using KernelBase::KernelBase;

  /// Callback used to apply the ``score_mod`` modifier subgraph to the
  /// pre-softmax score tensor of shape
  /// ``(batch_size, q_num_heads, q_seq_len, kv_seq_len)``. The callback
  /// receives a mutable reference to a tensor sharing ``Q``'s element type
  /// and must rewrite
  /// its contents in place while preserving ``data_type`` and ``shape``;
  /// the kernel validates these invariants after the call.
  using ScoreModFn = std::function<void(Tensor &)>;

  /// Callback used to apply the ``prob_mod`` modifier subgraph to the
  /// post-softmax probability tensor of shape
  /// ``(batch_size, q_num_heads, q_seq_len, kv_seq_len)``. The callback
  /// receives a mutable reference to a tensor sharing ``Q``'s element type
  /// and must rewrite
  /// its contents in place while preserving ``data_type`` and ``shape``;
  /// the kernel validates these invariants after the call.
  using ProbModFn = std::function<void(Tensor &)>;

  /// Computes the attention output for the given Q, K, V tensors using the
  /// default scaling factor ``1 / sqrt(head_size)``.
  Tensor operator()(const Tensor &Q, const Tensor &K, const Tensor &V,
                    RuntimeContext *rt = nullptr) const;

  /// Computes the attention output for the given Q, K, V tensors using an
  /// explicit ``scale`` value (matching the ``scale`` attribute of the
  /// operator).
  Tensor operator()(const Tensor &Q, const Tensor &K, const Tensor &V, float scale,
                    RuntimeContext *rt = nullptr) const;

  /// Computes the attention output and applies the ``prob_mod`` callback
  /// to the post-softmax probability tensor before computing
  /// ``Y = probs @ V``. When ``prob_mod`` is an empty ``std::function``
  /// this is equivalent to the overload without a ``prob_mod``
  /// callback.
  Tensor operator()(const Tensor &Q, const Tensor &K, const Tensor &V, float scale,
                    const ProbModFn &prob_mod, RuntimeContext *rt = nullptr) const;

  /// Computes the attention output and applies the ``score_mod`` callback
  /// to the pre-softmax score tensor before the softmax, and the
  /// ``prob_mod`` callback to the post-softmax probability tensor before
  /// computing ``Y = probs @ V``. When either callback is an empty
  /// ``std::function`` it is treated as identity.
  Tensor operator()(const Tensor &Q, const Tensor &K, const Tensor &V, float scale,
                    const ScoreModFn &score_mod, const ProbModFn &prob_mod,
                    RuntimeContext *rt = nullptr) const;

  /// In-place overload writing into a caller-allocated ``output`` tensor.
  /// ``output`` must already share ``Q``'s element type and have shape
  /// ``(batch_size, q_num_heads, q_seq_len, v_head_size)`` with its
  /// ``data`` buffer sized to match.
  void operator()(const Tensor &Q, const Tensor &K, const Tensor &V, float scale, Tensor &output,
                  RuntimeContext *rt = nullptr) const;

  /// In-place overload with ``prob_mod`` support. See the returning
  /// overload taking a ``ProbModFn`` for the semantics of ``prob_mod``;
  /// ``output`` has the same preconditions as the in-place overload
  /// without a callback.
  void operator()(const Tensor &Q, const Tensor &K, const Tensor &V, float scale,
                  const ProbModFn &prob_mod, Tensor &output, RuntimeContext *rt = nullptr) const;

  /// In-place overload with ``score_mod`` and ``prob_mod`` support. See
  /// the returning overloads for the semantics of the callbacks;
  /// ``output`` has the same preconditions as the in-place overload
  /// without a callback.
  void operator()(const Tensor &Q, const Tensor &K, const Tensor &V, float scale,
                  const ScoreModFn &score_mod, const ProbModFn &prob_mod, Tensor &output,
                  RuntimeContext *rt = nullptr) const;

  /// FlexAttention computes a fresh output buffer from independent reads of
  /// Q, K, V and never aliases an input buffer.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

} // namespace kernel
} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels
