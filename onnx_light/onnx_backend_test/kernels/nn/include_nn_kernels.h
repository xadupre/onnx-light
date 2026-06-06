// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/simple_tensor.h"

#include <cstdint>
#include <string>
#include <utility>
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
class AveragePool : public KernelBase {
public:
  using KernelBase::KernelBase;

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
};

/// Global average pooling on a FLOAT tensor laid out as ``(N, C, D1, ..., Dk)``.
/// The output shape is ``(N, C, 1, 1, ..., 1)`` — each spatial dimension is
/// reduced to 1 by computing the mean over all elements in that dimension.
class GlobalAveragePool : public KernelBase {
public:
  using KernelBase::KernelBase;

  /// Returns a FLOAT output tensor of shape ``(N, C, 1, 1, ..., 1)``.
  Tensor operator()(const Tensor &x) const;

  /// Output shape differs from the input shape, so storage cannot be shared.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Global max pooling on a FLOAT tensor laid out as ``(N, C, D1, ..., Dk)``.
/// The output shape is ``(N, C, 1, 1, ..., 1)`` — each spatial dimension is
/// reduced to 1 by computing the maximum over all elements in that dimension.
class GlobalMaxPool : public KernelBase {
public:
  using KernelBase::KernelBase;

  /// Returns a FLOAT output tensor of shape ``(N, C, 1, 1, ..., 1)``.
  Tensor operator()(const Tensor &x) const;

  /// Output shape differs from the input shape, so storage cannot be shared.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Global Lp pooling on a FLOAT tensor laid out as ``(N, C, D1, ..., Dk)``.
/// The output shape is ``(N, C, 1, 1, ..., 1)`` — each spatial dimension is
/// reduced to 1 by computing the Lp norm over all elements in that dimension.
/// The default value of ``p`` is 2 (L2 norm). When ``p == 1`` this is L1
/// pooling; ``p == 2`` (default) gives the RMS/L2 pooling defined by ONNX.
class GlobalLpPool : public KernelBase {
public:
  using KernelBase::KernelBase;

  /// Returns a FLOAT output tensor of shape ``(N, C, 1, 1, ..., 1)``.
  /// ``p`` is the Lp norm exponent (default 2).
  Tensor operator()(const Tensor &x, int64_t p = 2) const;

  /// Output shape differs from the input shape, so storage cannot be shared.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// N-D Lp pooling on a FLOAT tensor laid out as ``(N, C, D1, ..., Dk)``.
/// Implements the ONNX ``LpPool`` operator: for every output position, the
/// kernel computes ``(sum |x_i|^p)^(1/p)`` over the input values inside the
/// pooling window. Positions outside ``[0, in_dim)`` (i.e. in the padded
/// region) contribute ``0`` to the sum (matching the upstream ONNX
/// reference, where the input is zero-padded before pooling).
///
/// ``kernel_shape`` must have ``k`` entries; ``strides``, ``pads`` and
/// ``dilations`` (lengths ``k``, ``2 * k`` and ``k`` respectively) default
/// to all-ones / all-zeros / all-ones when omitted. ``auto_pad`` defaults
/// to ``NOTSET`` (use explicit ``pads``); when set to ``SAME_UPPER``,
/// ``SAME_LOWER`` or ``VALID`` the ``pads`` argument must be empty and the
/// begin/end padding is computed from the input shape. ``p`` defaults to
/// ``2`` (L2 norm).
class LpPool : public KernelBase {
public:
  using KernelBase::KernelBase;

  /// All attributes explicit. ``strides`` may be empty (treated as all 1),
  /// ``pads`` may be empty (treated as all 0) and ``dilations`` may be empty
  /// (treated as all 1).
  Tensor operator()(const Tensor &x, const std::vector<int64_t> &kernel_shape,
                    const std::vector<int64_t> &strides = {}, const std::vector<int64_t> &pads = {},
                    int64_t p = 2, bool ceil_mode = false,
                    const std::vector<int64_t> &dilations = {},
                    const std::string &auto_pad = "NOTSET") const;

  /// Output shape generally differs from the input shape, so the output
  /// buffer cannot in general alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return false; }
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
class BatchNormalization : public KernelBase {
public:
  using KernelBase::KernelBase;

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
};

/// Reference implementation of ``InstanceNormalization`` (opset 1, 6, 22).
///
/// Normalizes a FLOAT input ``X`` of shape ``(N, C, D1, ..., Dk)`` (rank
/// >= 2) per-instance, per-channel:
///
/// ``Y = scale * (X - mean) / sqrt(var + epsilon) + B``
///
/// where ``mean`` and ``var`` are computed over the spatial dimensions
/// ``D1, ..., Dk`` for every ``(n, c)`` pair. ``scale`` and ``B`` are 1-D
/// FLOAT tensors of length ``C``.
class InstanceNormalization : public KernelBase {
public:
  using KernelBase::KernelBase;

  /// Returns the output ``Y``. ``epsilon`` defaults to 1e-5f, the upstream
  /// default.
  Tensor operator()(const Tensor &x, const Tensor &scale, const Tensor &bias,
                    float epsilon = 1e-5f) const;

  void operator()(const Tensor &x, const Tensor &scale, const Tensor &bias, Tensor &output,
                  float epsilon = 1e-5f) const;

  /// Output ``Y`` has the same shape as ``X`` so the output buffer may
  /// alias the input ``X`` buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Reference implementation of ``GroupNormalization`` (opset 18, 21).
///
/// Groups the ``C`` channel dimension of a FLOAT input ``X`` of shape
/// ``(N, C, D1, ..., Dk)`` (rank >= 2) into ``num_groups`` equal-sized
/// groups, normalizes each ``(n, group)`` block over its channels and
/// spatial dims, then applies a per-channel affine ``scale * x + bias``:
///
/// ``Y = scale * (X - mean) / sqrt(var + epsilon) + bias``
///
/// The kernel implements the opset-21 contract where ``scale`` and ``bias``
/// are 1-D FLOAT tensors of length ``C``. ``num_groups`` must divide ``C``.
class GroupNormalization : public KernelBase {
public:
  using KernelBase::KernelBase;

  /// Returns the output ``Y``. ``epsilon`` defaults to 1e-5f, the upstream
  /// default.
  Tensor operator()(const Tensor &x, const Tensor &scale, const Tensor &bias, int64_t num_groups,
                    float epsilon = 1e-5f) const;

  void operator()(const Tensor &x, const Tensor &scale, const Tensor &bias, int64_t num_groups,
                  Tensor &output, float epsilon = 1e-5f) const;

  /// Output ``Y`` has the same shape as ``X`` so the output buffer may
  /// alias the input ``X`` buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Reference implementation of ``RMSNormalization`` (opset 23).
///
/// Normalizes a FLOAT input ``X`` of arbitrary rank ``r`` by dividing it by
/// the root-mean-square computed over the last ``r - axis`` dimensions, then
/// multiplies the normalized result by ``scale`` (with unidirectional
/// broadcasting):
///
/// ``Y = X / sqrt(mean(X * X, axes=normalized_axes) + epsilon) * scale``
///
/// where ``normalized_axes = [axis, ..., r - 1]``. ``scale`` must be
/// broadcastable to the normalized shape (i.e. ``X.shape[axis:]``).
class RMSNormalization : public KernelBase {
public:
  using KernelBase::KernelBase;

  /// Returns the output ``Y``. ``axis`` defaults to ``-1`` and ``epsilon``
  /// to ``1e-5f`` to match the upstream defaults.
  Tensor operator()(const Tensor &x, const Tensor &scale, int64_t axis = -1,
                    float epsilon = 1e-5f) const;

  void operator()(const Tensor &x, const Tensor &scale, Tensor &output, int64_t axis = -1,
                  float epsilon = 1e-5f) const;

  /// Output ``Y`` has the same shape as ``X`` so the output buffer may
  /// alias the input ``X`` buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Reference implementation of ``MeanVarianceNormalization`` (opset 9, 13).
///
/// Normalizes each element as
/// ``Y = (X - mean(X, axes)) / sqrt(mean((X - mean(X, axes))^2, axes))``.
/// The default axes are ``[0, 2, 3]``.
class MeanVarianceNormalization : public KernelBase {
public:
  using KernelBase::KernelBase;

  Tensor operator()(const Tensor &x, const std::vector<int64_t> &axes = {0, 2, 3}) const;
  void operator()(const Tensor &x, Tensor &output,
                  const std::vector<int64_t> &axes = {0, 2, 3}) const;

  /// Output ``Y`` has the same shape as ``X`` so the output buffer may
  /// alias the input ``X`` buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Flattens an input tensor of any element type into a 2-D matrix. Given an
/// input of shape ``(d_0, d_1, ..., d_n)`` and integer attribute ``axis``,
/// the output has shape ``(d_0 * d_1 * ... * d_(axis-1), d_axis * ... * d_n)``.
/// ``axis`` defaults to 1 and may be negative (counted from the back). When
/// ``axis == 0`` the outer dimension is 1 and the inner dimension is the
/// product of every input dimension. When ``axis == rank`` the outer
/// dimension is the product of every input dimension and the inner
/// dimension is 1.
///
/// The output reuses the input data buffer byte-for-byte (Flatten is a pure
/// view operation), so every element type supported by
/// :cpp:func:`PackedByteSize` is accepted. The output dtype always matches
/// the input dtype.
class Flatten : public KernelBase {
public:
  using KernelBase::KernelBase;

  /// Returns the 2-D output tensor. ``axis`` follows ONNX semantics.
  Tensor operator()(const Tensor &input, int64_t axis = 1) const;
  void operator()(const Tensor &input, int64_t axis, Tensor &output) const;

  /// Output element count equals the input element count, so storage can
  /// be shared in-place.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Reference implementation of ``Dropout`` (opset 12+ behavior).
///
/// ``Dropout`` takes an input tensor ``data`` and optional scalar ``ratio`` /
/// ``training_mode`` controls. In inference mode (``training_mode=false``),
/// ``output`` is a copy of ``data`` and ``mask`` (when requested) is all ones.
/// In training mode, each element is kept with probability ``1-ratio`` and
/// scaled by ``1/(1-ratio)``.
///
/// This backend test kernel currently supports FLOAT and DOUBLE tensors for
/// ``data``. ``ratio`` must be in ``[0, 1)``.
class Dropout : public KernelBase {
public:
  using KernelBase::KernelBase;

  std::pair<Tensor, Tensor> operator()(const Tensor &data, float ratio = 0.5f,
                                       bool training_mode = false, int64_t seed = kNoSeed) const;

  Tensor operator()(const Tensor &data, float ratio, bool training_mode, Tensor &mask,
                    int64_t seed = kNoSeed) const;

  static constexpr int64_t kNoSeed = -1;
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Reference implementation of ``LRN`` (Local Response Normalization).
///
/// Implements
///
///   ``Y[n, c, d1, ..., dk] = X[n, c, d1, ..., dk] /``
///   ``(bias + alpha / size * square_sum[n, c, d1, ..., dk]) ** beta``
///
/// where ``square_sum[n, c, d1, ..., dk]`` is the sum of squares over the
/// channel window
/// ``i in [max(0, c - floor((size-1)/2)), min(C-1, c + ceil((size-1)/2))]``.
///
/// The kernel accepts FLOAT inputs of rank ``>= 2`` laid out as
/// ``(N, C, D1, ..., Dk)``. ``size`` must be a strictly positive integer.
/// ``alpha`` defaults to ``0.0001``, ``beta`` to ``0.75`` and ``bias`` to
/// ``1.0`` (the ONNX defaults). The output shape matches the input shape.
class LRN : public KernelBase {
public:
  using KernelBase::KernelBase;

  /// Returns a FLOAT output tensor whose shape matches ``x``.
  Tensor operator()(const Tensor &x, int64_t size, float alpha = 0.0001f, float beta = 0.75f,
                    float bias = 1.0f) const;

  /// Output shape matches input shape, so storage may alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Reference implementation of ``LpNormalization``.
///
/// Computes ``output = input / Lp_norm(input, axis)`` element-wise along the
/// given ``axis`` (negative values count from the end). When the Lp norm
/// along a slice is zero, the corresponding output slice is set to zero to
/// avoid division by zero.
///
/// The kernel accepts FLOAT inputs of any rank ``>= 1``. ``axis`` defaults
/// to ``-1`` (the last axis). ``p`` must be either ``1`` (L1 norm) or
/// ``2`` (L2 norm) and defaults to ``2``. The output shape matches the
/// input shape.
class LpNormalization : public KernelBase {
public:
  using KernelBase::KernelBase;

  /// Returns a FLOAT output tensor whose shape matches ``x``.
  Tensor operator()(const Tensor &x, int64_t axis = -1, int64_t p = 2) const;

  /// Output shape matches input shape, so storage may alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// N-D max pooling on a FLOAT tensor laid out as ``(N, C, D1, ..., Dk)``.
/// Implements the ONNX ``MaxPool`` operator restricted to ``storage_order=0``
/// (row-major). ``kernel_shape`` must have ``k`` entries; ``strides``,
/// ``pads`` and ``dilations`` (lengths ``k``, ``2 * k`` and ``k``
/// respectively) default to all-ones / all-zeros / all-ones when omitted.
/// ``auto_pad`` defaults to ``NOTSET`` (use explicit ``pads``); when set to
/// ``SAME_UPPER``, ``SAME_LOWER`` or ``VALID`` the ``pads`` argument must be
/// empty and the begin/end padding is computed from the input shape.
///
/// The :cpp:func:`operator()` overload returns only the primary output ``Y``.
/// :cpp:func:`WithIndices` additionally produces the optional ``Indices``
/// tensor (``int64`` flat indices into the un-padded input buffer, mirroring
/// the upstream ONNX semantics).
class MaxPool : public KernelBase {
public:
  using KernelBase::KernelBase;

  /// Returns the primary output ``Y`` (FLOAT tensor of the pooled values).
  Tensor operator()(const Tensor &x, const std::vector<int64_t> &kernel_shape,
                    const std::vector<int64_t> &strides = {}, const std::vector<int64_t> &pads = {},
                    bool ceil_mode = false, const std::vector<int64_t> &dilations = {},
                    int64_t storage_order = 0, const std::string &auto_pad = "NOTSET") const;

  /// Returns ``(Y, Indices)`` where ``Indices`` is an ``int64`` tensor with
  /// the same shape as ``Y`` containing the flat indices into the un-padded
  /// input ``(N, C, D1, ..., Dk)`` buffer.
  std::pair<Tensor, Tensor>
  WithIndices(const Tensor &x, const std::vector<int64_t> &kernel_shape,
              const std::vector<int64_t> &strides = {}, const std::vector<int64_t> &pads = {},
              bool ceil_mode = false, const std::vector<int64_t> &dilations = {},
              int64_t storage_order = 0, const std::string &auto_pad = "NOTSET") const;

  /// Output shape generally differs from the input shape.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// N-D ``MaxUnpool`` on a FLOAT tensor with ``int64`` indices. Implements the
/// ONNX ``MaxUnpool`` operator: each value of ``x`` is scattered into a
/// zero-initialized output buffer at the linear index given by ``indices``
/// (interpreted as the flat ``(N, C, D1, ..., Dk)`` offset of the output).
///
/// Two overloads are provided: one that derives the output shape from the
/// ``kernel_shape``, ``strides`` and ``pads`` attributes, and one that takes
/// an explicit rank-1 ``int64`` ``output_shape`` tensor (if specified, the
/// ``pads`` attribute is ignored, matching the ONNX spec).
class MaxUnpool : public KernelBase {
public:
  using KernelBase::KernelBase;

  /// Two-input form: derives the output shape from ``kernel_shape``,
  /// ``strides`` and ``pads`` (defaults: 1 and 0).
  Tensor operator()(const Tensor &x, const Tensor &indices,
                    const std::vector<int64_t> &kernel_shape,
                    const std::vector<int64_t> &strides = {},
                    const std::vector<int64_t> &pads = {}) const;

  /// Three-input form: ``output_shape`` is a rank-1 ``int64`` tensor giving
  /// the full output shape ``(N, C, D1, ..., Dk)``.
  Tensor operator()(const Tensor &x, const Tensor &indices, const Tensor &output_shape,
                    const std::vector<int64_t> &kernel_shape,
                    const std::vector<int64_t> &strides = {},
                    const std::vector<int64_t> &pads = {}) const;

  /// Output shape differs from input shape.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// 2-D ``MaxRoiPool`` kernel restricted to FLOAT inputs/outputs. Implements
/// the ONNX ``MaxRoiPool`` operator: for each RoI ``[batch_id, x1, y1, x2,
/// y2]`` the kernel scales the corner coordinates by ``spatial_scale``,
/// rounds them to the nearest integer feature-map cell, divides the
/// resulting region into a ``pooled_shape[0] x pooled_shape[1]`` grid of
/// bins, and produces one max value per (channel, bin) pair. Empty bins
/// (e.g. when the scaled RoI is smaller than the requested pooled output)
/// emit ``0`` — mirroring the upstream reference implementation in
/// ``onnx/reference/ops/op_max_roi_pool.py``.
class MaxRoiPool : public KernelBase {
public:
  /// Attributes carried by the ONNX ``MaxRoiPool`` operator.
  struct Attributes {
    /// ROI pool output shape (height, width). Required; must contain
    /// exactly two positive entries.
    std::vector<int64_t> pooled_shape;
    /// Multiplicative spatial scale factor (default 1.0).
    float spatial_scale = 1.0f;
  };

  using KernelBase::KernelBase;

  /// Returns the pooled output ``Y`` of shape
  /// ``(num_rois, C, pooled_shape[0], pooled_shape[1])``.
  Tensor operator()(const Tensor &x, const Tensor &rois, const Attributes &attrs) const;

  /// In-place variant that writes into a pre-allocated ``output`` tensor of
  /// the expected shape and FLOAT dtype.
  void operator()(const Tensor &x, const Tensor &rois, const Attributes &attrs,
                  Tensor &output) const;

  /// Output element layout (num_rois, C, pooled_shape[0], pooled_shape[1])
  /// differs from any input, so storage cannot be shared.
  static constexpr bool CanRunInPlace() noexcept { return false; }
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
class RNN : public KernelBase {
public:
  using KernelBase::KernelBase;

  /// Returns the pair ``(Y, Y_h)``. ``b`` may be a default-constructed
  /// (empty-shape) ``Tensor`` to indicate the optional ``B`` input is
  /// missing; same convention for ``initial_h``.
  std::pair<Tensor, Tensor> operator()(const Tensor &x, const Tensor &w, const Tensor &r,
                                       const Tensor &b = Tensor{},
                                       const Tensor &initial_h = Tensor{}) const;

  /// Output shape generally differs from the input shape, so storage
  /// cannot in general be shared.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Single-direction (``"forward"``) one-layer GRU on FLOAT tensors using
/// the default ``Sigmoid``/``Tanh`` activations. Implements the upstream
/// ONNX ``GRU`` formula
///
///   ``z_t = sigmoid(X_t @ Wz^T + H_{t-1} @ Rz^T + Wbz + Rbz)``
///   ``r_t = sigmoid(X_t @ Wr^T + H_{t-1} @ Rr^T + Wbr + Rbr)``
///   ``h_t = tanh   (X_t @ Wh^T + (r_t (.) H_{t-1}) @ Rh^T + Wbh + Rbh)``
///       (``linear_before_reset == 0``, the default)
///   ``h_t = tanh   (X_t @ Wh^T + r_t (.) (H_{t-1} @ Rh^T + Rbh) + Wbh)``
///       (``linear_before_reset != 0``)
///   ``H_t = (1 - z_t) (.) h_t + z_t (.) H_{t-1}``
///
/// for ``layout=0`` only (``X.shape = [seq_length, batch_size,
/// input_size]``; ``W.shape = [1, 3 * hidden_size, input_size]``;
/// ``R.shape = [1, 3 * hidden_size, hidden_size]``; optional ``B.shape =
/// [1, 6 * hidden_size]`` (``[Wb, Rb]`` each with 3 gate blocks in the
/// ONNX gate order ``z, r, h``); optional ``initial_h.shape =
/// [1, batch_size, hidden_size]``, defaulting to zeros). ``sequence_lens``
/// is not supported (every batch must share the same sequence length);
/// ``activations``, ``clip`` and non-``forward`` ``direction`` are not
/// supported.
///
/// The two outputs are produced together: ``Y`` has shape
/// ``[seq_length, 1, batch_size, hidden_size]`` and is the concatenation of
/// every per-time-step hidden state; ``Y_h`` has shape
/// ``[1, batch_size, hidden_size]`` and equals the last time step of ``Y``.
class GRU : public KernelBase {
public:
  using KernelBase::KernelBase;

  /// Returns the pair ``(Y, Y_h)``. ``b`` may be a default-constructed
  /// (empty-shape) ``Tensor`` to indicate the optional ``B`` input is
  /// missing; same convention for ``initial_h``. ``linear_before_reset``
  /// matches the ONNX attribute of the same name (default ``0``).
  std::pair<Tensor, Tensor> operator()(const Tensor &x, const Tensor &w, const Tensor &r,
                                       const Tensor &b = Tensor{},
                                       const Tensor &initial_h = Tensor{},
                                       int64_t linear_before_reset = 0) const;

  /// Output shape generally differs from the input shape, so storage
  /// cannot in general be shared.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Single-direction (``"forward"``) one-layer LSTM on FLOAT tensors using
/// the default ``Sigmoid``/``Tanh``/``Tanh`` activations. Implements the
/// upstream ONNX ``LSTM`` formula
///
///   ``it = sigmoid(Xt @ Wi^T + Ht-1 @ Ri^T + Pi (.) Ct-1 + Wbi + Rbi)``
///   ``ft = sigmoid(Xt @ Wf^T + Ht-1 @ Rf^T + Pf (.) Ct-1 + Wbf + Rbf)``
///   ``ct = tanh   (Xt @ Wc^T + Ht-1 @ Rc^T               + Wbc + Rbc)``
///   ``Ct = ft (.) Ct-1 + it (.) ct``
///   ``ot = sigmoid(Xt @ Wo^T + Ht-1 @ Ro^T + Po (.) Ct   + Wbo + Rbo)``
///   ``Ht = ot (.) tanh(Ct)``
///
/// for ``layout=0`` only (``X.shape = [seq_length, batch_size,
/// input_size]``; ``W.shape = [1, 4 * hidden_size, input_size]``;
/// ``R.shape = [1, 4 * hidden_size, hidden_size]``; optional ``B.shape =
/// [1, 8 * hidden_size]`` (``[Wb, Rb]`` each with 4 gate blocks in the
/// ONNX gate order ``i, o, f, c``); optional ``P.shape =
/// [1, 3 * hidden_size]`` (peephole weights in gate order ``i, o, f``);
/// optional ``initial_h.shape = [1, batch_size, hidden_size]`` and
/// ``initial_c.shape = [1, batch_size, hidden_size]``, both defaulting
/// to zeros). ``sequence_lens`` is not supported (every batch must share
/// the same sequence length); ``activations``, ``clip``,
/// ``input_forget`` and non-``forward`` ``direction`` are not supported.
///
/// The two outputs are produced together: ``Y`` has shape
/// ``[seq_length, 1, batch_size, hidden_size]`` and is the concatenation of
/// every per-time-step hidden state; ``Y_h`` has shape
/// ``[1, batch_size, hidden_size]`` and equals the last time step of ``Y``.
/// The optional third output ``Y_c`` is not produced by this overload.
class LSTM : public KernelBase {
public:
  using KernelBase::KernelBase;

  /// Returns the pair ``(Y, Y_h)``. ``b``, ``initial_h``, ``initial_c``
  /// and ``p`` may each be a default-constructed (empty-shape) ``Tensor``
  /// to indicate that the corresponding optional input is missing.
  std::pair<Tensor, Tensor> operator()(const Tensor &x, const Tensor &w, const Tensor &r,
                                       const Tensor &b = Tensor{},
                                       const Tensor &initial_h = Tensor{},
                                       const Tensor &initial_c = Tensor{},
                                       const Tensor &p = Tensor{}) const;

  /// Output shape generally differs from the input shape, so storage
  /// cannot in general be shared.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference implementation of ``ai.onnx::Attention`` (v23 / v24).
///
/// Computes scaled dot-product attention. The baseline computation is
///
///   ``Y = Softmax((Q @ K^T) * scale + attn_bias, axis=-1) @ V``
///
/// where ``attn_bias`` is derived from the optional ``attn_mask`` input
/// (and/or the ``is_causal`` attribute) as described below.
///
/// Both rank-4 layouts ``(batch, num_heads, seq, head_size)`` and rank-3
/// "fused" layouts ``(batch, seq, num_heads * head_size)`` are accepted.
/// When rank-3 inputs are used the ``q_num_heads`` and ``kv_num_heads``
/// attributes must be set and ``Y`` is returned with the same rank-3
/// layout as the inputs.
///
/// Supported features (mirroring the upstream operator):
///
///   * ``scale`` attribute — when unset the implementation uses
///     ``1 / sqrt(head_size)``.
///   * ``is_causal`` attribute — applies an upper-triangular ``-inf`` mask
///     aligned to the upper-left corner of the ``(q_seq_len,
///     total_kv_seq_len)`` score matrix (accounting for any cached KV).
///   * ``softcap`` attribute — when ``> 0``, the pre-softmax scores are
///     scaled by ``softcap * tanh(s / softcap)``.
///   * ``qk_matmul_output_mode`` attribute (0/1/2/3) — selects which stage
///     of the QK pipeline is exposed in the auxiliary ``qk_matmul`` output
///     (0: raw ``(Q @ K^T) * scale``; 1: with bias; 2: after softcap; 3:
///     after softmax).
///   * Grouped Query Attention (``q_num_heads`` multiple of
///     ``kv_num_heads``).
///   * Optional ``attn_mask`` input: FLOAT (rank 2-4, broadcasted to
///     ``(batch, q_num_heads, q_seq_len, total_kv_seq_len)`` and added as
///     a bias) or BOOL (rank 2-4, ``true`` means "attend"; ``false``
///     becomes ``-inf``). When the mask's last dimension is shorter than
///     ``total_kv_seq_len`` it is padded with ``-inf`` / ``false``.
///   * Optional ``past_key``/``past_value`` rank-4 FLOAT inputs that are
///     concatenated along the sequence axis with ``K``/``V`` to form the
///     ``present_key``/``present_value`` outputs.
///
/// Not modeled: the v24 ``nonpad_kv_seqlen`` input and the
/// ``softmax_precision`` attribute.
class Attention : public KernelBase {
public:
  using KernelBase::KernelBase;

  /// Bundles every optional scalar/string attribute of the upstream
  /// ``ai.onnx::Attention`` operator. ``has_scale`` controls whether
  /// ``scale`` overrides the default ``1 / sqrt(head_size)``. When the
  /// inputs are rank-3 (fused layout), ``q_num_heads`` and ``kv_num_heads``
  /// must both be positive; otherwise both default to ``0`` (auto-detected
  /// from the rank-4 input shapes).
  struct Attributes {
    bool has_scale = false;        ///< When true, ``scale`` overrides the default.
    float scale = 0.0f;            ///< Multiplier applied to the ``Q @ K^T`` dot product
                                   ///< before the optional bias and softmax. When
                                   ///< ``has_scale`` is false the kernel uses
                                   ///< ``1 / sqrt(head_size)`` instead.
    bool is_causal = false;        ///< When true, applies a causal upper-triangular ``-inf`` mask.
    float softcap = 0.0f;          ///< When ``> 0``, applies ``softcap * tanh(s / softcap)``.
    int qk_matmul_output_mode = 0; ///< 0: raw; 1: + bias; 2: after softcap; 3: after softmax.
    int64_t q_num_heads = 0;       ///< Required when inputs are rank-3.
    int64_t kv_num_heads = 0;      ///< Required when inputs are rank-3.
  };

  /// Bundles every output produced by the kernel. ``Y`` is always
  /// populated; ``present_key`` / ``present_value`` mirror the upstream
  /// optional outputs and equal ``K`` / ``V`` (or their concatenation with
  /// the past tensors when those were supplied). ``qk_matmul_output``
  /// reflects the chosen ``qk_matmul_output_mode`` and is always rank-4
  /// FLOAT ``(batch, q_num_heads, q_seq_len, total_kv_seq_len)``.
  struct Result {
    Tensor Y;                ///< Primary output (same rank as Q).
    Tensor present_key;      ///< Rank-4 ``(batch, kv_num_heads, total_kv_seq_len, head_size)``.
    Tensor present_value;    ///< Rank-4 ``(batch, kv_num_heads, total_kv_seq_len, v_head_size)``.
    Tensor qk_matmul_output; ///< Auxiliary tensor (mode-dependent).
  };

  /// Default-attribute overload. ``scale = 1 / sqrt(head_size)``, no mask,
  /// no causal, no softcap, no past KV.
  Tensor operator()(const Tensor &Q, const Tensor &K, const Tensor &V) const;

  /// Explicit ``scale`` overload, no mask.
  Tensor operator()(const Tensor &Q, const Tensor &K, const Tensor &V, float scale) const;

  /// Explicit ``scale`` plus an optional FLOAT attention bias whose shape
  /// must broadcast to ``(batch, q_num_heads, q_seq_len, kv_seq_len)``.
  /// A default-constructed ``Tensor`` omits the mask.
  Tensor operator()(const Tensor &Q, const Tensor &K, const Tensor &V, float scale,
                    const Tensor &attn_mask) const;

  /// In-place overload writing into a caller-allocated ``output`` tensor.
  /// ``output`` must already be FLOAT and shape
  /// ``(batch, q_num_heads, q_seq_len, v_head_size)``.
  void operator()(const Tensor &Q, const Tensor &K, const Tensor &V, float scale,
                  const Tensor *attn_mask, Tensor &output) const;

  /// Comprehensive overload covering every supported feature. ``attn_mask``,
  /// ``past_key`` and ``past_value`` may be ``nullptr`` to mean "absent".
  Result operator()(const Tensor &Q, const Tensor &K, const Tensor &V, const Attributes &attrs,
                    const Tensor *attn_mask = nullptr, const Tensor *past_key = nullptr,
                    const Tensor *past_value = nullptr) const;

  /// Attention computes a fresh output buffer from independent reads of
  /// Q, K, V and never aliases an input buffer.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference 2-D ``DeformConv`` kernel restricted to FLOAT tensors.
///
/// Implements the deformable convolution defined in
/// https://arxiv.org/abs/1703.06211 and https://arxiv.org/abs/1811.11168,
/// matching the upstream ``ai.onnx::DeformConv`` operator (since opset 19;
/// the opset 22 update only widens ``T`` to also accept ``bfloat16`` —
/// not modeled here). Only the rank-4 (2-D image) case ``X.shape =
/// (N, C, H, W)`` is supported.
///
/// Attributes mirror the ONNX schema:
///
///   * ``kernel_shape`` (optional): inferred from ``W.shape[2:]`` when
///     omitted.
///   * ``strides`` / ``dilations`` (optional, default all ones).
///   * ``pads`` (optional, default all zeros, length ``2 * spatial_rank`` in
///     ``[h_begin, w_begin, h_end, w_end]`` order).
///   * ``group`` (default 1) — channels grouped along ``C`` / ``oC``.
///   * ``offset_group`` (default 1) — number of offset groups; ``C`` must be
///     divisible by ``offset_group``.
///
/// Inputs:
///   * ``x``    — FLOAT rank-4 ``(N, C, H, W)``.
///   * ``w``    — FLOAT rank-4 ``(oC, C/group, kH, kW)``.
///   * ``offset`` — FLOAT rank-4 ``(N, offset_group * kH * kW * 2, oH, oW)``;
///     the inner ``2`` channels are stored ``(y, x)``.
///   * ``b``     — Optional 1-D FLOAT bias of length ``oC``; an empty-shape
///     ``Tensor`` indicates absence (treated as zeros).
///   * ``mask``  — Optional FLOAT rank-4 mask
///     ``(N, offset_group * kH * kW, oH, oW)``; an empty-shape ``Tensor``
///     indicates absence (treated as ones).
class DeformConv : public KernelBase {
public:
  /// Attributes carried by the ONNX ``DeformConv`` operator. Defaults match
  /// the upstream schema (since opset 19).
  struct Attributes {
    std::vector<int64_t> kernel_shape; ///< Defaults to ``W.shape[2:]``.
    std::vector<int64_t> strides;      ///< Defaults to all ones.
    std::vector<int64_t> pads;         ///< Defaults to all zeros (length ``2 * rank``).
    std::vector<int64_t> dilations;    ///< Defaults to all ones.
    int64_t group = 1;                 ///< Number of conv groups.
    int64_t offset_group = 1;          ///< Number of offset groups.
  };

  using KernelBase::KernelBase;

  Tensor operator()(const Tensor &x, const Tensor &w, const Tensor &offset, const Tensor &b,
                    const Tensor &mask, const Attributes &attrs) const;

  void operator()(const Tensor &x, const Tensor &w, const Tensor &offset, const Tensor &b,
                  const Tensor &mask, const Attributes &attrs, Tensor &output) const;

  /// Output shape differs from any input, so storage cannot be shared.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference N-D ``Conv`` kernel restricted to FLOAT tensors.
///
/// Implements the standard convolution defined by the upstream
/// ``ai.onnx::Conv`` operator. Inputs are laid out as ``X.shape =
/// (N, C, D1, ..., Dk)``, ``W.shape = (M, C/group, k1, ..., kk)``; output
/// shape is ``(N, M, oD1, ..., oDk)``. Supports the ``kernel_shape``,
/// ``strides``, ``pads``, ``dilations``, ``group`` and ``auto_pad``
/// attributes (``NOTSET``, ``SAME_UPPER``, ``SAME_LOWER``, ``VALID``).
class Conv : public KernelBase {
public:
  /// Attributes carried by the ONNX ``Conv`` operator.
  struct Attributes {
    std::vector<int64_t> kernel_shape; ///< Defaults to ``W.shape[2:]``.
    std::vector<int64_t> strides;      ///< Defaults to all ones.
    std::vector<int64_t> pads;         ///< Defaults to all zeros (length ``2 * rank``).
    std::vector<int64_t> dilations;    ///< Defaults to all ones.
    int64_t group = 1;                 ///< Number of conv groups.
    std::string auto_pad = "NOTSET"; ///< ``NOTSET`` / ``SAME_UPPER`` / ``SAME_LOWER`` / ``VALID``.
  };

  using KernelBase::KernelBase;

  /// Returning overload. ``B`` may be a default-constructed (empty-shape)
  /// ``Tensor`` to indicate the optional bias is missing.
  Tensor operator()(const Tensor &x, const Tensor &w, const Tensor &b,
                    const Attributes &attrs) const;

  /// In-place overload writing into a caller-allocated FLOAT ``output``.
  void operator()(const Tensor &x, const Tensor &w, const Tensor &b, const Attributes &attrs,
                  Tensor &output) const;

  /// Output shape generally differs from the input, so storage cannot be shared.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference N-D ``ConvInteger`` kernel.
///
/// Inputs ``x`` and ``w`` are 8-bit integer tensors (``int8`` or ``uint8``)
/// and the output ``y`` is ``int32``. The optional ``x_zero_point`` must be
/// a scalar with the same dtype as ``x``; the optional ``w_zero_point`` may
/// be a scalar or a 1-D length-``M`` tensor (per-output-channel) with the
/// same dtype as ``w``. Empty-shape ``Tensor`` indicates absence.
class ConvInteger : public KernelBase {
public:
  /// Attributes carried by the ONNX ``ConvInteger`` operator.
  struct Attributes {
    std::vector<int64_t> kernel_shape;
    std::vector<int64_t> strides;
    std::vector<int64_t> pads;
    std::vector<int64_t> dilations;
    int64_t group = 1;
    std::string auto_pad = "NOTSET";
  };

  using KernelBase::KernelBase;

  /// Returning overload.
  Tensor operator()(const Tensor &x, const Tensor &w, const Tensor &x_zero_point,
                    const Tensor &w_zero_point, const Attributes &attrs) const;

  /// In-place overload writing into a caller-allocated INT32 ``output``.
  void operator()(const Tensor &x, const Tensor &w, const Tensor &x_zero_point,
                  const Tensor &w_zero_point, const Attributes &attrs, Tensor &output) const;

  /// Output dtype differs from input dtype, so storage cannot be shared.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference N-D ``ConvTranspose`` kernel restricted to FLOAT tensors.
///
/// Implements the upstream ``ai.onnx::ConvTranspose``. Input layout is
/// ``X.shape = (N, C, D1, ..., Dk)`` and ``W.shape = (C, M/group, k1, ..., kk)``.
/// When ``output_shape`` is provided the per-axis padding is derived per
/// the upstream spec; otherwise the output spatial dim is
/// ``stride[i] * (iD[i] - 1) + output_padding[i] +
///  ((k[i]-1)*dil[i]+1) - pads[start] - pads[end]``.
class ConvTranspose : public KernelBase {
public:
  /// Attributes carried by the ONNX ``ConvTranspose`` operator.
  struct Attributes {
    std::vector<int64_t> kernel_shape;
    std::vector<int64_t> strides;
    std::vector<int64_t> pads;
    std::vector<int64_t> dilations;
    std::vector<int64_t> output_padding;
    std::vector<int64_t> output_shape;
    int64_t group = 1;
    std::string auto_pad = "NOTSET";
  };

  using KernelBase::KernelBase;

  /// Returning overload.
  Tensor operator()(const Tensor &x, const Tensor &w, const Tensor &b,
                    const Attributes &attrs) const;

  /// In-place overload writing into a caller-allocated FLOAT ``output``.
  void operator()(const Tensor &x, const Tensor &w, const Tensor &b, const Attributes &attrs,
                  Tensor &output) const;

  /// Output shape generally differs from the input, so storage cannot be shared.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference N-D ``Col2Im`` kernel restricted to FLOAT tensors.
/// Implements the upstream ``ai.onnx::Col2Im`` operator (since opset 18),
/// rearranging column blocks back into a multi-dimensional batched image.
///
/// Inputs:
///   * ``input`` — FLOAT rank-3 tensor of shape
///     ``(N, C * product(block_shape), L)`` where ``L`` is the number of
///     blocks enumerated in lexicographic order.
///   * ``image_shape`` — INT64 1-D tensor of length ``n_spatial`` giving
///     the spatial dimensions ``(d_1, ..., d_n)`` of the output image.
///   * ``block_shape`` — INT64 1-D tensor of length ``n_spatial`` giving
///     the (un-dilated) block shape.
///
/// Attributes mirror the ONNX schema:
///   * ``dilations`` (optional, defaults to all ones).
///   * ``pads``      (optional, defaults to all zeros, length
///     ``2 * n_spatial`` in ``[d1_begin, ..., dn_begin, d1_end, ..., dn_end]``
///     order).
///   * ``strides``   (optional, defaults to all ones).
///
/// Output: FLOAT rank-``2 + n_spatial`` tensor of shape
/// ``(N, C, image_shape[0], ..., image_shape[n_spatial - 1])``. Overlapping
/// block contributions are summed (matching PyTorch's ``fold``).
class Col2Im : public KernelBase {
public:
  /// Attributes carried by the ONNX ``Col2Im`` operator.
  struct Attributes {
    std::vector<int64_t> dilations; ///< Defaults to all ones.
    std::vector<int64_t> pads;      ///< Defaults to all zeros (length ``2 * n_spatial``).
    std::vector<int64_t> strides;   ///< Defaults to all ones.
  };

  using KernelBase::KernelBase;

  Tensor operator()(const Tensor &input, const Tensor &image_shape, const Tensor &block_shape,
                    const Attributes &attrs) const;

  void operator()(const Tensor &input, const Tensor &image_shape, const Tensor &block_shape,
                  const Attributes &attrs, Tensor &output) const;

  /// Output shape differs from input shape, so storage cannot be shared.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference implementation of ``ai.onnx::RotaryEmbedding`` (v23).
///
/// Applies the rotary positional embedding (RoPE) defined in
/// https://arxiv.org/pdf/2104.09864 to the input tensor ``X``. The kernel
/// accepts both the rank-3 layout ``(batch_size, sequence_length,
/// hidden_size)`` (in which case ``num_heads`` must be provided and
/// ``hidden_size = num_heads * head_size``) and the rank-4 layout
/// ``(batch_size, num_heads, sequence_length, head_size)``. The output has
/// the same shape and dtype as ``X``.
///
/// ``cos_cache`` and ``sin_cache`` are either rank-2
/// ``(max_position_id_plus_1, head_size / 2)`` tables indexed by the
/// optional ``position_ids`` input or rank-3
/// ``(batch_size, sequence_length, head_size / 2)`` tensors used directly
/// when ``position_ids`` is omitted. When ``rotary_embedding_dim`` is
/// non-zero the last dimension of the caches must equal
/// ``rotary_embedding_dim / 2`` and rotation is applied only to the first
/// ``rotary_embedding_dim`` channels of each head; the remaining channels
/// are passed through unchanged.
///
/// When ``interleaved`` is true, even/odd channels of each head are paired
/// instead of the (default) splitting into two contiguous halves.
class RotaryEmbedding : public KernelBase {
public:
  /// Attributes carried by the ONNX ``RotaryEmbedding`` operator. All
  /// defaults match the upstream schema.
  struct Attributes {
    bool interleaved = false;         ///< Pair (0,1),(2,3),... instead of (0,h/2),(1,h/2+1),...
    int64_t rotary_embedding_dim = 0; ///< 0 means "rotate all of head_size".
    int64_t num_heads = 0;            ///< Required when ``X`` is rank-3.
  };

  using KernelBase::KernelBase;

  /// Returning overload. ``position_ids`` may be a default-constructed
  /// (empty-shape) ``Tensor`` to indicate the optional input is missing.
  Tensor operator()(const Tensor &X, const Tensor &cos_cache, const Tensor &sin_cache,
                    const Tensor &position_ids, const Attributes &attrs) const;

  /// In-place overload writing into a caller-allocated output of the same
  /// shape and dtype as ``X``.
  void operator()(const Tensor &X, const Tensor &cos_cache, const Tensor &sin_cache,
                  const Tensor *position_ids, const Attributes &attrs, Tensor &output) const;

  /// Output shape matches input shape; storage could theoretically alias
  /// ``X`` but the in-place overload above does not exploit it.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
