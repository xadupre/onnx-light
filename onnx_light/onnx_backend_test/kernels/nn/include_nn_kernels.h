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

/// Single-direction (``"forward"``) one-layer RNN on FLOAT tensors using
/// the ``Tanh`` activation. Implements the upstream ONNX ``RNN`` formula
///
///   ``H_t = tanh(X_t @ W^T + W_b + H_{t-1} @ R^T + R_b)``
///
/// for ``layout=0`` only (``X.shape = [seq_length, batch_size,
/// input_size]``; ``W.shape = [1, hidden_size, input_size]``;
/// ``R.shape = [1, hidden_size, hidden_size]``; optional ``B.shape =
/// [1, 2 * hidden_size]`` (``[Wb, Rb]``); optional ``initial_h.shape =
/// [1, batch_size, hidden_size]``, defaulting to zeros). The ``activations``
/// attribute, if present, must be either empty or the single value
/// ``"Tanh"``; ``direction`` must be ``"forward"`` (the default);
/// ``sequence_lens`` is not supported (every batch must share the same
/// sequence length); ``clip`` is not supported.
///
/// The two outputs are produced together: ``Y`` has shape
/// ``[seq_length, 1, batch_size, hidden_size]`` and is the concatenation of
/// every per-time-step hidden state; ``Y_h`` has shape
/// ``[1, batch_size, hidden_size]`` and equals the last time step of ``Y``.
class RNN {
public:
  explicit RNN(const KernelContext &ctx) : ctx_(ctx) {}

  /// Returns the pair ``(Y, Y_h)``. ``b`` may be a default-constructed
  /// (empty-shape) ``Tensor`` to indicate the optional ``B`` input is
  /// missing; same convention for ``initial_h``.
  std::pair<Tensor, Tensor> operator()(const Tensor &x, const Tensor &w, const Tensor &r,
                                       const Tensor &b = Tensor{},
                                       const Tensor &initial_h = Tensor{}) const;

  /// Output shape generally differs from the input shape, so storage
  /// cannot in general be shared.
  static constexpr bool CanRunInPlace() noexcept { return false; }

private:
  const KernelContext &ctx_;
};

/// Reference implementation of ``ai.onnx::Attention`` (v23 / v24).
///
/// Computes scaled dot-product attention on rank-4 FLOAT inputs. The
/// computation is
///
///   ``Y = Softmax((Q @ K^T) * scale + attn_mask, axis=-1) @ V``
///
/// where ``attn_mask`` is treated as 0 when not provided. Supports Grouped
/// Query Attention (GQA): when ``q_num_heads != kv_num_heads`` each K/V
/// head is shared by a contiguous group of query heads, i.e. query head
/// ``h`` uses K/V head ``floor(h / (q_num_heads / kv_num_heads))``;
/// ``q_num_heads`` must be a multiple of ``kv_num_heads``.
///
/// The optional ``past_key``/``past_value`` and ``present_*`` outputs of
/// the upstream operator, the ``softcap`` attribute, the
/// ``qk_matmul_output`` output and the ``nonpad_kv_seqlen`` input (v24)
/// are not modeled by this reference kernel. Only the primary output ``Y``
/// is produced, mirroring the un-modified baseline that backends should
/// reproduce when none of those optional features are used.
///
/// Only rank-4 FLOAT tensors are supported.
class Attention {
public:
  explicit Attention(const KernelContext &ctx) : ctx_(ctx) {}

  /// Computes the attention output for the given Q, K, V tensors using the
  /// default scaling factor ``1 / sqrt(head_size)``.
  Tensor operator()(const Tensor &Q, const Tensor &K, const Tensor &V) const;

  /// Computes the attention output for the given Q, K, V tensors using an
  /// explicit ``scale`` value (matching the ``scale`` attribute of the
  /// operator).
  Tensor operator()(const Tensor &Q, const Tensor &K, const Tensor &V, float scale) const;

  /// Computes the attention output with an explicit ``scale`` and an
  /// optional FLOAT attention bias ``attn_mask`` whose shape must be
  /// broadcastable to ``(batch_size, q_num_heads, q_seq_len, kv_seq_len)``.
  /// Pass a default-constructed ``Tensor`` (or ``nullptr`` via the
  /// in-place overload) to omit the mask. Bool/integer masks and the
  /// ``is_causal`` attribute can be applied by the caller by precomputing
  /// the FLOAT bias.
  Tensor operator()(const Tensor &Q, const Tensor &K, const Tensor &V, float scale,
                    const Tensor &attn_mask) const;

  /// In-place overload writing into a caller-allocated ``output`` tensor.
  /// ``output`` must already be a FLOAT tensor whose shape equals
  /// ``(batch_size, q_num_heads, q_seq_len, v_head_size)`` and whose
  /// ``data`` buffer has been sized to match. ``attn_mask`` is optional
  /// (pass ``nullptr`` to omit).
  void operator()(const Tensor &Q, const Tensor &K, const Tensor &V, float scale,
                  const Tensor *attn_mask, Tensor &output) const;

  /// Attention computes a fresh output buffer from independent reads of
  /// Q, K, V and never aliases an input buffer.
  static constexpr bool CanRunInPlace() noexcept { return false; }

private:
  const KernelContext &ctx_;
};

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
