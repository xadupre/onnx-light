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
// Reference implementations of the ``text`` backend test kernels.
//
// Each kernel is exposed as a small class whose constructor takes a
// :ref:`KernelContext` (carrying the opset against which the kernel must
// behave) and whose ``operator()`` performs the computation.
//
// Two flavors of ``operator()`` are provided:
//
//   * The returning overload (``Tensor operator()(...) const``) allocates a
//     fresh ``Tensor`` whose ``string_data`` buffer is owned by the returned
//     value.
//   * The in-place overload (``void operator()(..., Tensor &output) const``)
//     writes results into a caller-supplied output tensor whose
//     ``string_data`` vector has already been sized. The caller is
//     responsible for setting ``output.data_type`` to
//     ``TensorProto::DataType::STRING``, ``output.shape`` to the broadcasted
//     shape and ``output.string_data`` to the broadcasted element count; the
//     kernel validates these attributes and throws ``std::invalid_argument``
//     on mismatch.
//
// ``StringConcat`` mirrors the ONNX ``StringConcat`` operator (since opset
// 20 in the ai.onnx domain): it concatenates two ``tensor(string)`` inputs
// element-wise with NumPy-style bidirectional broadcasting. ``output[i] =
// x[i] + y[i]`` after broadcasting both inputs to a common shape.
//
// Each kernel class also exposes a ``static constexpr bool CanRunInPlace()``
// query indicating whether the output tensor's ``string_data`` buffer may
// alias either input's buffer. ``StringConcat`` writes new strings whose
// bytes depend on both inputs, so aliasing an input is not permitted.
// ---------------------------------------------------------------------------

/// Element-wise concatenation of two ``tensor(string)`` inputs with
/// NumPy-style bidirectional broadcasting.
class StringConcat {
public:
  explicit StringConcat(const KernelContext &ctx) : ctx_(ctx) {}

  Tensor operator()(const Tensor &x, const Tensor &y) const;
  void operator()(const Tensor &x, const Tensor &y, Tensor &output) const;

  /// Output bytes depend on both inputs, so the output buffer cannot
  /// safely alias either input buffer.
  static constexpr bool CanRunInPlace() noexcept { return false; }

private:
  const KernelContext &ctx_;
};

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
