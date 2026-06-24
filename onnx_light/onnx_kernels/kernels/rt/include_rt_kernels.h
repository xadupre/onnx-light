// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_kernels/kernels/kernel_context.h"
#include "onnx_kernels/simple_tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

/// Reference implementation of the light-only ``ai.rt::DelayedInitializer`` op.
class DelayedInitializer : public KernelBase {
public:
  struct Attributes {
    std::vector<int64_t> shape;
    int32_t dtype = 0;
    std::string load_device;
    std::string runtime_device;
    std::string filename;
    int64_t offset = 0;
  };

  DelayedInitializer(const KernelContext &ctx, Attributes attrs);

  Tensor operator()() const;

  static constexpr bool CanRunInPlace() noexcept { return false; }

private:
  static std::vector<uint8_t> LoadBytes(const Attributes &attrs);
  static int64_t ComputeElementCount(const std::vector<int64_t> &shape);

  Attributes attrs_;
  std::vector<uint8_t> loaded_bytes_;
};

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
