// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/test_case.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

/**
 * Construction-time context passed to backend test kernel classes.
 *
 * Kernels are implemented as classes whose constructor takes a single
 * ``KernelContext`` argument. The context bundles the opset against which
 * the kernel must behave so the same kernel class can specialize its
 * computation (or perform opset-specific validation) without changing
 * its call sites.
 *
 * Only the ``opset`` field is exposed today; new construction-time inputs
 * (for example a device descriptor or an allocator) can be added later as
 * additional fields without breaking existing kernel classes.
 */
struct KernelContext {
  /// Opset against which the kernel must behave (domain + version).
  OpsetId opset;

  KernelContext() = default;
  explicit KernelContext(OpsetId opset_) : opset(std::move(opset_)) {}
};

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
