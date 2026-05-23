// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/tensor.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

// ---------------------------------------------------------------------------
// Reference implementations of the ``math`` backend test kernels. Each kernel
// returns a fresh ``Tensor`` whose data buffer is owned by the returned value.
// ---------------------------------------------------------------------------

/// Element-wise absolute value.
Tensor Abs(const Tensor &x);

/// Element-wise addition with NumPy-style broadcasting.
Tensor Add(const Tensor &x, const Tensor &y);

/// BlackmanWindow function evaluated at ``size`` integer samples. When
/// ``periodic`` is true the window is computed as if of length ``size+1`` and
/// the last sample is discarded (matches NumPy/ONNX conventions).
Tensor BlackmanWindow(const Tensor &size, bool periodic = true);

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
