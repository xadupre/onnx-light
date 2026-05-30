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
// Reference implementations of the ``reduction`` backend test kernels.
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
// ``ReduceSum`` mirrors the ONNX ``ReduceSum`` operator at opset 13, where
// ``axes`` is an optional second input tensor (int64) rather than an
// attribute. The kernel supports FLOAT tensors and the standard ``keepdims``
// and ``noop_with_empty_axes`` attributes. When ``axes`` is omitted (or
// empty), the kernel either reduces over all dimensions (the default,
// ``noop_with_empty_axes == false``) or performs an identity copy
// (``noop_with_empty_axes == true``).
//
// Each kernel class also exposes a ``static constexpr bool CanRunInPlace()``
// query indicating whether the output tensor's data buffer may alias one of
// the input tensors' buffers. ``ReduceSum``'s output shape generally differs
// from its input, so storage cannot be shared with an input.
// ---------------------------------------------------------------------------

/// Returns the indices of the maximum (``ArgMax``) or minimum (``ArgMin``)
/// values of a FLOAT input ``data`` along a single ``axis``.
///
/// The output is always an INT64 tensor whose shape matches ``data`` with
/// the reduced ``axis`` either dropped (``keepdims = false``) or replaced
/// by ``1`` (``keepdims = true``, the default).
///
/// ``axis`` follows ONNX semantics and accepts values in
/// ``[-rank(data), rank(data) - 1]``. ``select_last_index`` (default
/// ``false``, ONNX opset 12+) selects the index of the last occurrence of
/// the extremum when multiple positions are tied; otherwise the first
/// occurrence is selected.
class ArgReduce : public KernelBase {
public:
  enum class Mode { kMax, kMin };

  ArgReduce(const KernelContext &ctx, Mode mode) : KernelBase(ctx), mode_(mode) {}

  Tensor operator()(const Tensor &data, int64_t axis = 0, bool keepdims = true,
                    bool select_last_index = false) const;
  void operator()(const Tensor &data, int64_t axis, bool keepdims, bool select_last_index,
                  Tensor &output) const;

  /// The INT64 output buffer never aliases the FLOAT input buffer.
  static constexpr bool CanRunInPlace() noexcept { return false; }

private:
  Mode mode_;
};

/// ``ArgMax`` wrapper around :class:`ArgReduce` configured for the maximum.
class ArgMax : public ArgReduce {
public:
  explicit ArgMax(const KernelContext &ctx) : ArgReduce(ctx, ArgReduce::Mode::kMax) {}
};

/// ``ArgMin`` wrapper around :class:`ArgReduce` configured for the minimum.
class ArgMin : public ArgReduce {
public:
  explicit ArgMin(const KernelContext &ctx) : ArgReduce(ctx, ArgReduce::Mode::kMin) {}
};

/// Sum reduction of a FLOAT input ``data`` along the dimensions listed in the
/// optional ``axes`` int64 tensor. If ``axes`` is omitted (or empty), the
/// kernel reduces over all dimensions unless ``noop_with_empty_axes`` is true
/// in which case it performs an identity copy.
class ReduceSum : public KernelBase {
public:
  using KernelBase::KernelBase;

  /// ``axes`` omitted: reduces over all dimensions of ``data`` (the default
  /// when ``noop_with_empty_axes`` is false) or returns a copy of ``data``
  /// when ``noop_with_empty_axes`` is true.
  Tensor operator()(const Tensor &data, bool keepdims = true,
                    bool noop_with_empty_axes = false) const;
  void operator()(const Tensor &data, bool keepdims, bool noop_with_empty_axes,
                  Tensor &output) const;

  /// Explicit ``axes``: an int64 tensor whose elements are the dimensions to
  /// reduce. Negative axis values are accepted (ONNX semantics: ``axis`` in
  /// ``[-rank(data), rank(data) - 1]``).
  Tensor operator()(const Tensor &data, const Tensor &axes, bool keepdims = true,
                    bool noop_with_empty_axes = false) const;
  void operator()(const Tensor &data, const Tensor &axes, bool keepdims, bool noop_with_empty_axes,
                  Tensor &output) const;

  /// Output shape generally differs from the input shape, so the output
  /// buffer cannot in general alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Shared FLOAT kernel for ``ReduceMax`` and ``ReduceMin``.
class ReduceMinMax : public KernelBase {
public:
  enum class Mode { kMax, kMin };

  ReduceMinMax(const KernelContext &ctx, Mode mode) : KernelBase(ctx), mode_(mode) {}

  Tensor operator()(const Tensor &data, bool keepdims = true,
                    bool noop_with_empty_axes = false) const;
  void operator()(const Tensor &data, bool keepdims, bool noop_with_empty_axes,
                  Tensor &output) const;

  Tensor operator()(const Tensor &data, const Tensor &axes, bool keepdims = true,
                    bool noop_with_empty_axes = false) const;
  void operator()(const Tensor &data, const Tensor &axes, bool keepdims, bool noop_with_empty_axes,
                  Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return false; }

private:
  Mode mode_;
};

/// ``ReduceMax`` wrapper around :class:`ReduceMinMax` configured for maximum.
class ReduceMax : public ReduceMinMax {
public:
  explicit ReduceMax(const KernelContext &ctx) : ReduceMinMax(ctx, ReduceMinMax::Mode::kMax) {}
};

/// ``ReduceMin`` wrapper around :class:`ReduceMinMax` configured for minimum.
class ReduceMin : public ReduceMinMax {
public:
  explicit ReduceMin(const KernelContext &ctx) : ReduceMinMax(ctx, ReduceMinMax::Mode::kMin) {}
};

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
