// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#ifndef ONNX_LIGHT_ONNX_CORE_RUNTIME_KERNEL_CONTEXT_H
#define ONNX_LIGHT_ONNX_CORE_RUNTIME_KERNEL_CONTEXT_H

// Note: this header intentionally does NOT include
// ``onnx_core/backend_test/test_case.h``, even though that header also needs
// ``KernelContext``/``OpsetId``: ``test_case.h`` itself includes this
// file, and pulling it in here would create a circular ``#include`` that
// silently truncates whichever of the two happens to be entered first
// (relying on include order for correctness).
#include "onnx_core/compute/raw_buffer_allocator.h"

#include <cstdint>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {

// Forward declaration: kernels only hold a ``const NodeProto *`` and never
// dereference the proto's definition here, so the full ``onnx_proto`` header is
// intentionally not pulled into this low-level runtime header.
class NodeProto;

namespace core::runtime {

using ::onnx_light::core::runtime::RawBufferAllocator;

// Forward declaration: ``KernelBase::Run`` takes ``RuntimeContext &`` by
// reference, so a forward declaration is sufficient and avoids a circular
// include (``runtime_context.h`` includes this header).
class RuntimeContext;

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
  /// Initialized from the allocator supplied to :cpp:class:`RuntimeContext`.
  RawBufferAllocator *allocator = nullptr;

  KernelContext() = default;
  explicit KernelContext(OpsetId opset_, RawBufferAllocator *allocator_ = nullptr)
      : opset(std::move(opset_)), allocator(allocator_) {}
};

/**
 * Base class for every kernel.
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
 * ``KernelBase`` is also the runtime dispatch interface: it exposes a virtual
 * :cpp:func:`Run` that the runtime (:cpp:func:`RunNode` /
 * :cpp:class:`RuntimeSession`) calls once per node. Each dispatch-registered
 * kernel overrides :cpp:func:`Run` to read the node's current inputs from
 * ``rt.tensors()``, invoke its own ``operator()`` and store the produced
 * outputs back. The node the kernel runs for is attached once (during kernel
 * resolution) via :cpp:func:`set_node`; the owning graph / execution plan
 * outlives the kernel, so the stored pointer stays valid for the kernel's life.
 * The default :cpp:func:`Run` throws — only kernels registered in the dispatch
 * table (and the control-flow / custom kernels) override it.
 *
 * Centralizing the context member here keeps every kernel class consistent,
 * makes it trivial to extend the construction-time interface (e.g. by
 * adding new fields to ``KernelContext``), and avoids a repeated
 * boilerplate ``KernelContext ctx_;`` member in each kernel.
 */
class KernelBase {
public:
  explicit KernelBase(const KernelContext &ctx) : ctx_(ctx) {}
  KernelBase(const KernelBase &) = default;
  KernelBase &operator=(const KernelBase &) = default;
  virtual ~KernelBase() = default;

  /// Attaches the node this kernel runs for. Called once at kernel-resolution
  /// time; the node outlives the kernel.
  void set_node(const NodeProto &node) { node_ = &node; }

  /// Records the identity string of the concrete kernel class this instance
  /// implements (e.g. ``"onnx_kernels:CPU:ai.onnx:Abs"``). Set once, at
  /// kernel-resolution time, by the factory that instantiates the kernel (the
  /// dispatch-table factory forwards the kernel class's ``static constexpr
  /// name``; control-flow, model-local-function and custom-kernel adapters set
  /// their own identity). ``name`` must have static storage duration.
  void set_kernel_name(const char *name) noexcept { kernel_name_ = name; }

  /// Returns the identity string of the concrete kernel class this instance
  /// implements, or ``nullptr`` when the kernel was constructed directly
  /// (e.g. in tests) without being resolved through the dispatch path.
  const char *kernel_name() const noexcept { return kernel_name_; }

  /// Runs the kernel for its node (see :cpp:func:`set_node`) against the
  /// current state of ``rt``: reads the node's current inputs, computes and
  /// writes the outputs back. Safe to call repeatedly. The default throws;
  /// every dispatch-registered kernel overrides it.
  virtual void Run(RuntimeContext &rt);

  /// Appends the identity strings of the kernels this instance uses to ``out``,
  /// skipping any already present in ``seen``. The default records this
  /// kernel's own :cpp:func:`kernel_name` (falling back to nothing when the
  /// kernel was constructed outside the dispatch path and has no name).
  /// Control-flow kernels (``If`` / ``Loop`` / ``Scan`` / ``SequenceMap``)
  /// override this to also recurse into the subgraph kernels their owned
  /// sessions instantiated, resolving them against ``rt`` the same way
  /// :cpp:func:`RuntimeSession::UsedKernels` does. This is how nested
  /// control-flow subgraphs contribute their kernels to
  /// :cpp:func:`RuntimeSession::UsedKernels`.
  virtual void CollectUsedKernels(RuntimeContext &rt, std::vector<std::string> &out,
                                  std::unordered_set<std::string> &seen) const;

protected:
  KernelContext ctx_;

  /// The node this kernel was built for, or ``nullptr`` when the kernel is used
  /// directly (e.g. in tests) without being resolved through the dispatch path.
  const NodeProto *node_ = nullptr;

  /// Identity string of the concrete kernel class, set by the instantiating
  /// factory (see :cpp:func:`set_kernel_name`); ``nullptr`` until then.
  const char *kernel_name_ = nullptr;
};

} // namespace core::runtime
} // namespace ONNX_LIGHT_NAMESPACE

#endif // ONNX_LIGHT_ONNX_CORE_RUNTIME_KERNEL_CONTEXT_H
