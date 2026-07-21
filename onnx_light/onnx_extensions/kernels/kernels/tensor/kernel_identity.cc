// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"

#include "onnx_core/runtime/runtime_context.h"
#include <algorithm>
#include <cstdint>
#include <cstring>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

Tensor Identity::operator()(const Tensor &input, RuntimeContext *rt) const {
  Tensor output = MakeOutputTensor(input.data_type, input.shape, input.size_bytes(),
                                   rt ? rt->allocator() : nullptr);
  output.name = input.name;
  if (input.data_type == static_cast<int32_t>(DataType::STRING)) {
    output.string_data = input.string_data;
  } else if (input.size_bytes() != 0) {
    std::memcpy(output.mutable_bytes(), input.bytes(), input.size_bytes());
  }
  return output;
}

void Identity::operator()(const Tensor &input, Tensor &output) const {
  EXT_ENFORCE_INVALID(output.data_type == input.data_type,
                      "kernel::Identity: preallocated output dtype mismatch.");
  EXT_ENFORCE_INVALID(output.shape == input.shape,
                      "kernel::Identity: preallocated output shape mismatch.");
  if (input.data_type == static_cast<int32_t>(DataType::STRING)) {
    EXT_ENFORCE_INVALID(output.string_data.size() == input.string_data.size(),
                        "kernel::Identity: preallocated string output size mismatch.");
    output.string_data = input.string_data;
  } else {
    EXT_ENFORCE_INVALID(output.size_bytes() == input.size_bytes(),
                        "kernel::Identity: preallocated output byte-size mismatch.");
    if (input.size_bytes() != 0) {
      std::memcpy(output.mutable_bytes(), input.bytes(), input.size_bytes());
    }
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
