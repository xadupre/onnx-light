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
// Reference implementations of the ``traditionalml`` backend test kernels
// (``ai.onnx.ml`` domain).
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
// Each kernel class also exposes a ``static constexpr bool CanRunInPlace()``
// query indicating whether the output tensor's data buffer may alias one of
// the input tensors' buffers. ``LabelEncoder`` reports ``false`` because in
// the general case the input and output element types differ.
// ---------------------------------------------------------------------------

/// Maps each element of the input tensor through a pair of parallel
/// ``keys``/``values`` arrays — the reference behaviour of the ``ai.onnx.ml``
/// ``LabelEncoder`` operator (since opset 4 in the ``ai.onnx.ml`` domain).
///
/// For every input element ``x[i]``, the output element ``y[i]`` is
/// ``values[k]`` where ``k`` is the index of the first ``keys[k]`` that
/// matches ``x[i]``; if no key matches, ``y[i]`` is ``default_value``.
///
/// The output tensor has the same shape as the input tensor. The kernel
/// supports the following ``(KeyT, ValueT)`` element-type combinations via
/// explicit template instantiations:
///
///   * ``(int64_t, int64_t)``
///   * ``(int64_t, float)``
///   * ``(float,   int64_t)``
///   * ``(float,   float)``
///
/// ``keys.size()`` must match ``values.size()``. The kernel throws
/// ``std::invalid_argument`` if the input element type does not match
/// ``KeyT`` or, for the in-place overload, if the preallocated output's
/// type/shape do not match the resolved value type and the input shape.
class LabelEncoder {
public:
  explicit LabelEncoder(const KernelContext &ctx) : ctx_(ctx) {}

  template <typename KeyT, typename ValueT>
  Tensor operator()(const Tensor &x, const std::vector<KeyT> &keys,
                    const std::vector<ValueT> &values, ValueT default_value) const;

  template <typename KeyT, typename ValueT>
  void operator()(const Tensor &x, const std::vector<KeyT> &keys, const std::vector<ValueT> &values,
                  ValueT default_value, Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return false; }

private:
  KernelContext ctx_;
};

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
