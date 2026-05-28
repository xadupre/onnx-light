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
// Reference implementations of the ``generator`` backend test kernels.
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
// ``Constant`` mirrors the ONNX ``Constant`` operator: it has no inputs and
// produces an output tensor whose data type, shape and bytes are taken from
// the operator's ``value`` attribute. The kernel does not parse attributes
// itself — it consumes the already-decoded ``value`` tensor, which keeps
// this reference implementation independent from any attribute-decoding
// machinery while still exercising the operator's value-producing
// semantics.
//
// Each kernel class also exposes a ``static constexpr bool CanRunInPlace()``
// query indicating whether the output tensor's data buffer may alias the
// supplied ``value`` tensor's buffer. ``Constant`` simply copies the
// ``value`` bytes into the output, so aliasing with the value buffer is
// permitted.
// ---------------------------------------------------------------------------

/// Returns a copy of the ``value`` attribute of the ``Constant`` op.
class Constant {
public:
  explicit Constant(const KernelContext &ctx) : ctx_(ctx) {}
  Tensor operator()(const Tensor &value) const;
  void operator()(const Tensor &value, Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return true; }

private:
  const KernelContext &ctx_;
};

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
