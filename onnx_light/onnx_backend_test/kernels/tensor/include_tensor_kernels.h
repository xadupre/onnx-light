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
// Reference implementations of the ``tensor`` backend test kernels.
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
// the input tensors' buffers. ``Concat`` returns ``false`` because its
// output is strictly larger (along ``axis``) than any single input and
// therefore cannot share storage with one.
// ---------------------------------------------------------------------------

/// Concatenates a list of tensors along ``axis`` (since opset 13). All input
/// tensors must share the same data type and the same shape except along the
/// concatenation axis. ``axis`` may be negative, in which case it counts from
/// the back of the input rank.
class Concat {
public:
  explicit Concat(const KernelContext &ctx) : ctx_(ctx) {}
  Tensor operator()(const std::vector<Tensor> &inputs, int64_t axis) const;
  void operator()(const std::vector<Tensor> &inputs, int64_t axis, Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return false; }

private:
  KernelContext ctx_;
};

/// Performs element-wise type conversion of an input tensor ``x`` to the
/// data type specified by ``to`` (a ``TensorProto::DataType`` value,
/// mirroring the ``Cast`` operator's required ``to`` attribute). The output
/// shape always matches the input shape.
///
/// The reference implementation supports the numeric element types in the
/// backend test library — ``FLOAT``, ``DOUBLE``, ``INT32``, ``INT64``,
/// ``INT8``, ``UINT8``, ``INT16``, ``UINT16`` and ``BOOL`` — as well as
/// ``STRING`` in either direction (numeric ↔ STRING uses the canonical
/// decimal representation). Other dtypes will cause the kernel to throw
/// ``std::invalid_argument``: this is sufficient for the backend test
/// cases registered today and keeps the implementation small.
/// Out-of-range floating-point values when casting to an integer dtype
/// follow C++ ``static_cast`` semantics, which matches the behaviour
/// exercised by the upstream ``test_cast_FLOAT_to_*`` node tests for the
/// supported conversions.
class Cast {
public:
  explicit Cast(const KernelContext &ctx) : ctx_(ctx) {}
  Tensor operator()(const Tensor &x, int32_t to) const;
  void operator()(const Tensor &x, int32_t to, Tensor &output) const;

  /// Output element type may differ from the input element type, so storage
  /// can not be shared in general.
  static constexpr bool CanRunInPlace() noexcept { return false; }

private:
  KernelContext ctx_;
};

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
