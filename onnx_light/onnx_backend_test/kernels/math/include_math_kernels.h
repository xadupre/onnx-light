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
// Reference implementations of the ``math`` backend test kernels.
//
// Each kernel is exposed as a small class whose constructor takes a
// :ref:`KernelContext` (carrying the opset against which the kernel must
// behave) and whose ``operator()`` performs the computation. Every call
// returns a fresh ``Tensor`` whose data buffer is owned by the returned
// value.
// ---------------------------------------------------------------------------

/// Element-wise absolute value.
class Abs {
public:
  explicit Abs(const KernelContext &ctx) : ctx_(ctx) {}
  Tensor operator()(const Tensor &x) const;

  const KernelContext &context() const { return ctx_; }

private:
  KernelContext ctx_;
};

/// Element-wise addition with NumPy-style broadcasting.
class Add {
public:
  explicit Add(const KernelContext &ctx) : ctx_(ctx) {}
  Tensor operator()(const Tensor &x, const Tensor &y) const;

  const KernelContext &context() const { return ctx_; }

private:
  KernelContext ctx_;
};

/// BlackmanWindow function evaluated at ``size`` integer samples. When
/// ``periodic`` is true the window is computed as if of length ``size+1`` and
/// the last sample is discarded (matches NumPy/ONNX conventions).
class BlackmanWindow {
public:
  explicit BlackmanWindow(const KernelContext &ctx) : ctx_(ctx) {}
  Tensor operator()(const Tensor &size, bool periodic = true) const;

  const KernelContext &context() const { return ctx_; }

private:
  KernelContext ctx_;
};

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
