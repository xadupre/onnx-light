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
// Reference implementations of the ``controlflow`` backend test kernels.
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
// ``If`` mirrors the ONNX ``If`` operator: it selects one of two precomputed
// branch values based on a scalar BOOL condition. The kernel does not
// execute the branch subgraphs itself — it consumes their already-evaluated
// outputs, which keeps this reference implementation independent from any
// graph-executor machinery while still exercising the operator's selection
// semantics.
//
// Each kernel class also exposes a ``static constexpr bool CanRunInPlace()``
// query indicating whether the output tensor's data buffer may alias one of
// the input tensors' buffers. ``If`` simply copies the selected branch into
// the output, so aliasing with that branch's buffer is permitted.
// ---------------------------------------------------------------------------

/// Selects ``then_value`` when the scalar BOOL ``cond`` is true,
/// otherwise returns ``else_value``. Both branch values must share the
/// same data type and shape.
class If {
public:
  explicit If(const KernelContext &ctx) : ctx_(ctx) {}
  Tensor operator()(const Tensor &cond, const Tensor &then_value, const Tensor &else_value) const;
  void operator()(const Tensor &cond, const Tensor &then_value, const Tensor &else_value,
                  Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return true; }

private:
  const KernelContext &ctx_;
};

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
