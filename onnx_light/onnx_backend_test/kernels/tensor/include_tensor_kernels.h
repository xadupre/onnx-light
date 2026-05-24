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
// behave) and whose ``operator()`` performs the computation. Every call
// returns a fresh ``Tensor`` whose data buffer is owned by the returned
// value.
// ---------------------------------------------------------------------------

/// Concatenates a list of tensors along ``axis`` (since opset 13). All input
/// tensors must share the same data type and the same shape except along the
/// concatenation axis. ``axis`` may be negative, in which case it counts from
/// the back of the input rank.
class Concat {
public:
  explicit Concat(const KernelContext &ctx) : ctx_(ctx) {}
  Tensor operator()(const std::vector<Tensor> &inputs, int64_t axis) const;

private:
  KernelContext ctx_;
};

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
