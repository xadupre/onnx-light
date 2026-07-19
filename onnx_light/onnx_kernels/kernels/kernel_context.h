// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/test_case.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

class RawBufferAllocator;

namespace kernel {

/**
 * Lightweight opset identifier used by the backend test library.
 *
 * Mirrors the (domain, version) pair carried by ``OperatorSetIdProto`` but
 * keeps the public API of this library independent from the proto type so
 * test cases can be declared without touching the proto wire format.
 */
struct OpsetId {
  std::string domain;
  int64_t version = 0;

  OpsetId() = default;
  OpsetId(std::string domain_, int64_t version_) : domain(std::move(domain_)), version(version_) {}
};

/// Builds an :ref:`OpsetId` for the default ai.onnx domain (empty string).
inline OpsetId DefaultOpset(int64_t version) { return OpsetId(std::string(), version); }

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

  /// Allocator kernels must use to acquire the raw byte storage of the
  /// tensors they return, so no result buffer is allocated outside the
  /// runtime context. ``nullptr`` when the owning
  /// :cpp:class:`RuntimeContext` has no allocator attached (the default),
  /// in which case results fall back to inline ``std::vector`` storage.
  /// Kept in sync by :cpp:func:`RuntimeContext::set_allocator`.
  RawBufferAllocator *allocator = nullptr;

  KernelContext() = default;
  explicit KernelContext(OpsetId opset_, RawBufferAllocator *allocator_ = nullptr)
      : opset(std::move(opset_)), allocator(allocator_) {}
};

/**
 * Base class for every backend test kernel.
 *
 * Each concrete kernel class derives from ``KernelBase`` so it inherits
 * ownership of the construction-time ``KernelContext``. Derived kernels access
 * the context through the protected ``ctx_`` member and typically inherit
 * ``KernelBase``'s constructor via ``using KernelBase::KernelBase;``, which
 * preserves the ``explicit`` qualifier on the single-argument constructor.
 *
 * ``ctx_`` is stored by value so that kernels are safely copy-constructible:
 * lazy test-case lambdas capture kernels by value and may outlive the
 * ``KernelContext`` local variable used at registration time.
 *
 * Centralizing the context member here keeps every kernel class consistent,
 * makes it trivial to extend the construction-time interface (e.g. by
 * adding new fields to ``KernelContext``), and avoids a repeated
 * boilerplate ``KernelContext ctx_;`` member in each kernel.
 */
class KernelBase {
public:
  explicit KernelBase(const KernelContext &ctx) : ctx_(ctx) {}

protected:
  KernelContext ctx_;
};

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
