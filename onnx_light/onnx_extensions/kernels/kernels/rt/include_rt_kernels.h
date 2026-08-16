// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"

#include <cstdint>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels {
// Re-exports the runtime types moved to ``onnx_core::runtime`` so
// kernel implementations below can keep referring to them
// unqualified, matching pre-move code.
using namespace ::onnx_light::core::runtime;

namespace kernel {
using ::onnx_light::core::runtime::DefaultOpset;
using ::onnx_light::core::runtime::KernelBase;
using ::onnx_light::core::runtime::KernelContext;
using ::onnx_light::core::runtime::OpsetId;

/// Reference implementation of the light-only ``ai.rt::DelayedInitializer`` op.
class DelayedInitializer : public KernelBase {
public:
  static constexpr const char *name = "onnx_kernels:CPU:ai.rt:DelayedInitializer";
  void Run(RuntimeContext &rt) override;
  /// Inherits the context-only constructor so the dispatch factory can
  /// build the kernel; ``Run`` parses the ONNX attributes and constructs the
  /// fully-initialized instance it delegates to.
  using KernelBase::KernelBase;
  struct Attributes {
    onnx_kernels::Shape shape;
    int32_t dtype = 0;
    std::string load_device;
    std::string runtime_device;
    std::string filename;
    int64_t offset = 0;
  };

  /// Initializes the delayed-initializer kernel and eagerly loads bytes when requested.
  DelayedInitializer(const KernelContext &ctx, Attributes attrs);

  /// Returns the initialized tensor, loading bytes at execution time when needed.
  Tensor operator()(RuntimeContext *rt = nullptr) const;

  static constexpr bool CanRunInPlace() noexcept { return false; }

private:
  static Tensor LoadBytes(const Attributes &attrs, RawBufferAllocator *allocator = nullptr);
  static void LoadBytesInto(const Attributes &attrs, uint8_t *destination, size_t byte_count);
  static int64_t ComputeElementCount(const onnx_kernels::Shape &shape);

  Attributes attrs_;
  Tensor loaded_bytes_;
};

} // namespace kernel
} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels
