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
// behave) and whose ``operator()`` performs the computation. Every call
// returns a fresh ``Tensor`` whose data buffer is owned by the returned
// value.
//
// ``If`` mirrors the ONNX ``If`` operator: it selects one of two precomputed
// branch values based on a scalar BOOL condition. The kernel does not
// execute the branch subgraphs itself — it consumes their already-evaluated
// outputs, which keeps this reference implementation independent from any
// graph-executor machinery while still exercising the operator's selection
// semantics.
// ---------------------------------------------------------------------------

/// Selects ``then_value`` when the scalar BOOL ``cond`` is true,
/// otherwise returns ``else_value``. Both branch values must share the
/// same data type and shape.
class If {
public:
  explicit If(const KernelContext &ctx) : ctx_(ctx) {}
  Tensor operator()(const Tensor &cond, const Tensor &then_value, const Tensor &else_value) const;

private:
  KernelContext ctx_;
};

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
