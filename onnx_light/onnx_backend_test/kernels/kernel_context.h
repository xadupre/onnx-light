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

/**
 * Base class for every backend test kernel.
 *
 * Each concrete kernel class derives from ``KernelBase`` so it inherits
 * ownership of the construction-time ``KernelContext`` reference. Derived
 * kernels access the context through the protected ``ctx_`` member and
 * typically inherit ``KernelBase``'s constructor via ``using
 * KernelBase::KernelBase;``, which preserves the ``explicit`` qualifier on
 * the single-argument constructor.
 *
 * Centralizing the context member here keeps every kernel class consistent,
 * makes it trivial to extend the construction-time interface (e.g. by
 * adding new fields to ``KernelContext``), and avoids a repeated
 * boilerplate ``const KernelContext &ctx_;`` member in each kernel.
 */
class KernelBase {
public:
  explicit KernelBase(const KernelContext &ctx) : ctx_(ctx) {}

protected:
  const KernelContext &ctx_;
};

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
